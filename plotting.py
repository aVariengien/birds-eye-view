from core import (
    ChunkCollection,
    Pipeline,
    OpenAITextProcessor,
    OpenAIEmbeddor,
    UMAPReductor,
    Chunk,
    DotProductLabelor,
    EmbeddingSearch,
)
from file_loading import wrap_str
import colorcet as cc
from bokeh.plotting import figure  # type: ignore
from bokeh.models import ColumnDataSource, HoverTool, Button, TapTool, CustomJS, Div  # type: ignore
from bokeh.layouts import column, row  # type: ignore
from bokeh.plotting import figure
from bokeh.models import ( # type: ignore
    ColumnDataSource, HoverTool, Button, TapTool, CustomJS, Div,
    ColorBar, LinearColorMapper, CategoricalColorMapper, Select, Legend, LabelSet, Label
)
from bokeh.layouts import column, row
from bokeh.palettes import Category20, Category10  # type: ignore
from bokeh.transform import factor_cmap  # type: ignore
from bokeh.plotting import figure, output_file, save
from markdownify import markdownify as md  # type: ignore
import urllib.parse
import os
from bokeh.plotting import figure

from bokeh.layouts import column, row
from bokeh.palettes import Category20, Viridis256
from bokeh.transform import factor_cmap
from markdownify import markdownify as md
from typing import List


def make_display_text(chunk: Chunk, fields_to_include: List[str]):
    """Return a fancy text to display, and the raw text to be used as hover"""

    page = ""
    title = ""
    view_source = ""
    headings = ""

    if "title" in chunk.attribs:
        title = "Title: <i>" + chunk.attribs["title"] + "</i><br><br>"

    first_words = md(chunk.display_text, convert=[])  # strip all html tags
    first_words = urllib.parse.quote(" ".join(first_words.split(" ")[:3]))
    if "url" in chunk.attribs and "http" in chunk.attribs["url"]:
        view_source = f"""<br><a href="{chunk.attribs["url"]}#:~:text={first_words}">View source</a>"""

    if "page" in chunk.attribs:
        page = f"""<br>Page {chunk.attribs["page"]}"""

    if "headings" in chunk.attribs:
        headings = "<br> <b>" + "<br>>".join(chunk.attribs["headings"]) + "</b><br><br>"

    vis_field = fields_to_include[0]
    vis_field_value = f"""<br><br>{vis_field}: {chunk.attribs[vis_field]}"""

    text = f"{title}{headings}{chunk.display_text.replace("\n", "<br>")}{view_source}{page}" #{vis_field_value}

    raw_text = wrap_str(
        md(chunk.display_text, convert=[]), skip_line_char="<br>" #  + f"   |{vis_field}: {chunk.attribs[vis_field]}"
    )  # strip html
    return text, raw_text


def create_callbacks(source, text_div, prev_lines, next_lines, n_connections):
    common_js_code = """
    function updateActiveChunk(source, text_div, prev_lines, next_lines, new_active_index, n_connections) {
        if (new_active_index !== null && new_active_index >= 0) {
            source.data['active_chunk'][0] = new_active_index;
            source.selected.indices = [new_active_index];
            var text = source.data['text'][new_active_index] || '';
            text_div.text = text + '<br><br> Value: ' + source.data['val_on_display'][new_active_index];

            // Hide all lines initially
            prev_lines.forEach(line => line.visible = false);
            next_lines.forEach(line => line.visible = false);

            var current_index = new_active_index;

            // Show previous connections
            for (var i = 0; i < n_connections; i++) {
                var prev_chunk = source.data['prev_chunk'][current_index];
                if (prev_chunk !== new_active_index) {
                    prev_lines[i].data_source.data['x'] = [source.data['x'][current_index], source.data['x'][prev_chunk]];
                    prev_lines[i].data_source.data['y'] = [source.data['y'][current_index], source.data['y'][prev_chunk]];
                    prev_lines[i].visible = true;
                    current_index = prev_chunk;
                } else {
                    break;
                }
            }

            // Reset current_index for next connections
            current_index = new_active_index;

            // Show next connections
            for (var i = 0; i < n_connections; i++) {
                var next_chunk = source.data['next_chunk'][current_index];
                if (next_chunk !== null) {
                    next_lines[i].data_source.data['x'] = [source.data['x'][current_index], source.data['x'][next_chunk]];
                    next_lines[i].data_source.data['y'] = [source.data['y'][current_index], source.data['y'][next_chunk]];
                    next_lines[i].visible = true;
                    current_index = next_chunk;
                } else {
                    break;
                }
            }
            prev_lines.forEach(line => line.data_source.change.emit());
            next_lines.forEach(line => line.data_source.change.emit());
            source.change.emit();
        }
    }
    """

    callback = CustomJS(
        args=dict(source=source, text_div=text_div, prev_lines=prev_lines, next_lines=next_lines, n_connections=n_connections),
        code=common_js_code + """
        var index = cb_data.source.selected.indices[0];
        updateActiveChunk(source, text_div, prev_lines, next_lines, index, n_connections);
        """
    )

    go_previous = CustomJS(
        args=dict(source=source, text_div=text_div, prev_lines=prev_lines, next_lines=next_lines, n_connections=n_connections),
        code=common_js_code + """
        var active_index = source.data['active_chunk'][0];
        if (active_index >= 0) {
            var prev_chunk = source.data['prev_chunk'][active_index];
            updateActiveChunk(source, text_div, prev_lines, next_lines, prev_chunk, n_connections);
            source.data['active_index'][0] = prev_chunk
        }
        """
    )

    go_next = CustomJS(
        args=dict(source=source, text_div=text_div, prev_lines=prev_lines, next_lines=next_lines, n_connections=n_connections),
        code=common_js_code + """
        var active_index = source.data['active_chunk'][0];
        if (active_index >= 0) {
            var next_chunk = source.data['next_chunk'][active_index];
            updateActiveChunk(source, text_div, prev_lines, next_lines, next_chunk, n_connections);
            source.data['active_index'][0] = next_chunk
        }
        """
    )

    return callback, go_previous, go_next

def create_navigation_buttons(go_previous, go_next):
    button_left = Button(label="⬅️", width=50)
    button_left.js_on_click(go_previous)
    button_right = Button(label="➡️", width=50)
    button_right.js_on_click(go_next)
    return button_left, button_right



def prepare_data(chunk_collection, fields_to_include):
    x, y, texts, hover_texts = [], [], [], []
    prev_chunks, next_chunks = [], []
    field_values = {field: [] for field in fields_to_include}

    for chunk in chunk_collection.chunks:
        if chunk.x is not None:
            x.append(chunk.x)
            y.append(chunk.y)
            fancy_text, raw_text = make_display_text(chunk, fields_to_include)
            texts.append(fancy_text)
            hover_texts.append(raw_text)
            for field in fields_to_include:
                field_values[field].append(chunk.attribs.get(field, ""))
            prev_chunks.append(chunk.previous_chunk_index)
            next_chunks.append(chunk.next_chunk_index)

    return x, y, texts, hover_texts, field_values, prev_chunks, next_chunks

def create_data_source(x, y, texts, hover_texts, field_values, prev_chunks, next_chunks, fields_names):
    data = dict(
        x=x, y=y, text=texts, hover_texts=hover_texts,
        prev_chunk=prev_chunks, next_chunk=next_chunks,
        active_chunk=[-1] * len(prev_chunks),
        val_on_display=field_values[fields_names[0]],
    )
    data.update(field_values)
    for field in ["emoji", "label"]:
        if field in data:
            data[field+"_dynamic"] = data[field][::] # the field that is updated with zooming in and out

    return ColumnDataSource(data=data)


def remove_elements(l, to_remove):
    l_copy = l[::]
    for x in to_remove:
        if x in l_copy:
            l_copy.remove(x)
    return l_copy

def visualize_chunks(chunk_collection, fields_to_include, n_connections=5):
    # Prepare data
    existing_fields = []
    for f in fields_to_include:
        if f in chunk_collection.chunks[0].attribs:
            existing_fields.append(f)
    fields_to_include = existing_fields
    x, y, texts, hover_texts, field_values, prev_chunks, next_chunks = prepare_data(chunk_collection, existing_fields)

    # Create data source
    source = create_data_source(x, y, texts, hover_texts, field_values, prev_chunks, next_chunks, existing_fields)

    for i,field in enumerate(fields_to_include):
        wrapped_field = wrap_str(field, max_line_len=70, skip_line_char="<br>")
        source.data[wrapped_field] = source.data[field][::]
        fields_to_include[i] = wrapped_field


    # Create figure
    p = figure(height = 600, width = 900, tools="pan,wheel_zoom,box_zoom,reset", active_scroll="wheel_zoom")
# Create glyphs and color bars for each field
    field_glyphs = {}
    color_bars = {}
    additional_glyph = {}
   
    for field in fields_to_include:
        if field == "emoji_list" or field == "label_list":
            glyph = p.scatter(
                x="x", y="y", size=10, source=source,
                fill_color='blue',
                alpha=0.1,
                visible=False
            )
            additional_glyph[field] = Div()
            field_glyphs[field] = glyph
            color_bars[field] = Div()
        elif field == "label" or field == "emoji":
            labels = LabelSet(x='x', y='y', text=field + "_dynamic",
              x_offset=-10, y_offset=-5, source=source,  visible=False)
            glyph = p.scatter(
                x="x", y="y", size=10, source=source,
                fill_color='blue',
                alpha=0.1,
                visible=False
            )
            p.add_layout(labels)
            additional_glyph[field] = labels
            field_glyphs[field] = glyph
            color_bars[field] = Div()
        else:
            if type(source.data[field][0]) == list:
                source.data[field] = [str(l) for l in source.data[field]] # TODO: keep the first elements of the  list so there's less than 20 elements

            if source.data[field] and type(source.data[field][0]) == str:
                source.data[field] = [
                    wrap_str(s, max_line_len=20, skip_line_char="\n") for s in source.data[field]
                ]
                unique_values = list(set(source.data[field]))
                if len(unique_values) > 10:
                    source.data[field] = [
                        f"{unique_values.index(s)}.{s}" if len(s) > 20 else f"{unique_values.index(s)}.{s}" for s in source.data[field]
                    ]

            if all(isinstance(val, (int, float)) for val in source.data[field]):
                color_mapper = LinearColorMapper(palette=Viridis256, low=min(source.data[field]), high=max(source.data[field]))
            else:
                unique_values = list(set(source.data[field]))
                color_mapper = CategoricalColorMapper(factors=unique_values, palette=cc.glasbey[:len(unique_values)])

            glyph = p.scatter(
                x="x", y="y", size=10, source=source,
                fill_color={'field': field, 'transform': color_mapper},
                line_color={'field': field, 'transform': color_mapper},
                alpha=0.7,
                visible=False
            )
            field_glyphs[field] = glyph

            color_bar = ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                major_label_text_font_size='8pt',
                border_line_color=None,
                location=(0, 0),
                title=field,
                visible=False
            )
            p.add_layout(color_bar, 'right')
            color_bars[field] = color_bar

            additional_glyph[field] = Div() #dummy object

    # Make the first field visible
    field_glyphs[fields_to_include[0]].visible = True
    color_bars[fields_to_include[0]].visible = True
    additional_glyph[fields_to_include[0]].visible = True
    # Add tools
    # hover = HoverTool(renderers=[circles], tooltips=[("Text", "@hover_texts")])
    # p.add_tools(hover)
    hover = HoverTool(tooltips=[("Text", "@hover_texts")])
    hover.tooltips = """
        @hover_texts | Value: @val_on_display
    """
    p.add_tools(hover)
    p.add_tools(TapTool())
    
    # Create text div
    text_div = Div(width=300, height=500, text="")

    # Create connection lines
    prev_lines = [p.line(x=[], y=[], line_color="#ffceb8", line_width=2, visible=False) for _ in range(n_connections)]
    next_lines = [p.line(x=[], y=[], line_color="#f75002", line_width=2, visible=False) for _ in range(n_connections)]

    # Create field selector
    field_selector = Select(title="Color by:", value=fields_to_include[0], options=remove_elements(fields_to_include,["emoji_list", "label_list"]))

    # Create JavaScript callback for updating visible glyph and color bar
    update_visible_elements = CustomJS(args=dict(source=source, field_glyphs=field_glyphs, color_bars=color_bars, additional_glyph=additional_glyph), code="""
        var selected_field = cb_obj.value;
        for (let i = 0; i < source.data['val_on_display'].length; i++) {
            source.data['val_on_display'][i] = source.data[selected_field][i]
        }
        for (var field in field_glyphs) {
            field_glyphs[field].visible = (field === selected_field);
            color_bars[field].visible = (field === selected_field);
            additional_glyph[field].visible = (field === selected_field);
        }

    """)

    # Call back to change the labels depending on the woom level
    original_ranges = {'x':[], 'y': []}
    callback = CustomJS(args=dict(x_range=p.x_range, y_range=p.y_range, original_ranges=original_ranges, source=source), code="""
        if (original_ranges['x'].length == 0) {
            original_ranges['x'] = [x_range.start, x_range.end];
            original_ranges['y'] = [y_range.start, y_range.end];
        }
        var x_zoom = (x_range.end - x_range.start) / (original_ranges['x'][1] - original_ranges['x'][0]);
        var y_zoom = (y_range.end - y_range.start) / (original_ranges['y'][1] - original_ranges['y'][0]);

        var average_zoom = (x_zoom + y_zoom) / 2;  
        var min_level = 0.01; // to adjust to change the range
        var max_level = 1.5;
        console.log(average_zoom);

        for (const field of ['emoji', 'label']) {         
            if (field in source.data) {
                var number_zoom_level = source.data[field+'_list'][0].split(",").length; //, is the separator

                var index_to_show = Math.floor(
                    (number_zoom_level - 1) * 
                    (average_zoom - min_level) / (max_level - min_level)
                );

                // Ensure the index is within the valid range
                index_to_show = Math.max(0, Math.min(index_to_show, number_zoom_level - 1));
                console.log(index_to_show);
                for (let i = 0; i < source.data[field].length; i++) {
                    source.data[field+"_dynamic"][i] = source.data[field+'_list'][i].split(",")[number_zoom_level-1-index_to_show];
                    
                    //console.log( source.data[field+'_list'][i].split(",")[number_zoom_level-1-index_to_show]);
                    //console.log( source.data[field+'_list'][i]);
                    //console.log(number_zoom_level);
                }
                
            }
        }
        // Your custom JavaScript code here
    """)

    # Attach the callback to both x and y ranges
    p.x_range.js_on_change('start', callback)
    p.x_range.js_on_change('end', callback)
    p.y_range.js_on_change('start', callback)
    p.y_range.js_on_change('end', callback)

    field_selector.js_on_change('value', update_visible_elements)

    # Create callbacks
    callback, go_previous, go_next = create_callbacks(source, text_div, prev_lines, next_lines, n_connections)

    # Create navigation buttons
    button_left, button_right = create_navigation_buttons(go_previous, go_next)


    # Add callback to TapTool
    tap_tool = p.select(type=TapTool)[0]
    tap_tool.callback = callback

    # Create layout
    buttons = row(button_left, button_right)
    side = column(field_selector, text_div, buttons)
    layout = row(side, p)
    #p.aspect_ratio = 1

    return layout