from birds_eye_view.core import (
    ChunkCollection,
    Pipeline,
    OpenAITextProcessor,
    OpenAIEmbeddor,
    UMAPReductor,
    Chunk,
    DotProductLabelor,
    EmbeddingSearch,
)
from birds_eye_view.file_loading import wrap_str
import colorcet as cc  # type: ignore
from bokeh.plotting import figure  # type: ignore
from bokeh.models import ColumnDataSource, HoverTool, Button, TapTool, CustomJS, Div  # type: ignore
from bokeh.layouts import column, row  # type: ignore
from bokeh.plotting import figure
from bokeh.models import (  # type: ignore
    ColumnDataSource,
    Button,
    TapTool,
    CustomJS,
    Div,
    ColorBar,
    LinearColorMapper,
    CategoricalColorMapper,
    Select,
    Legend,
    LabelSet,
    Label,
)
from bokeh.models.css import Styles
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
from typing import List, Optional, Any

from bokeh.embed import file_html # type: ignore
from bokeh.resources import CDN # type: ignore

import webbrowser
import tempfile

DEFAULT_VIS_FIELD = ["emoji", "title", "doc_position", "page", "url", "index"]

def put_field_first(x, l):
    prev = l[0]
    if x in l:
        old_x_idx = l.index(x)
        l[old_x_idx] = prev
        l[0] = x
    else:
        l.insert(0, x)

def open_html_in_browser(html_content):
    # Create a temporary file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        # Write the HTML content to the file
        f.write(html_content)
        temp_file_name = f.name

    # Open the temporary file in the default web browser
    webbrowser.open('file://' + os.path.realpath(temp_file_name))


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

    text = f"{title}{headings}{chunk.display_text.replace("\n", "<br>")}{view_source}{page}"  # {vis_field_value}

    raw_text = wrap_str(
        md(chunk.display_text, convert=[]),
        skip_line_char="<br>",  #  + f"   |{vis_field}: {chunk.attribs[vis_field]}"
    )  # strip html
    return text, raw_text


def create_callbacks(source, text_div, prev_lines, next_lines, n_connections):
    common_js_code = """
    function updateActiveChunk(source, text_div, prev_lines, next_lines, new_active_index, n_connections) {
        console.log(new_active_index);
        if (new_active_index !== null && new_active_index >= 0) {
            source.data['active_chunk'][0] = new_active_index;
            source.selected.indices = [new_active_index];
            var text = source.data['text'][new_active_index] || '';
            text_div.text = text + '<br><br>' + source.data['active_chunk'][1] + ': ' + source.data['val_on_display'][new_active_index];

            // Hide all lines initially
            prev_lines.forEach(line => line.visible = false);
            next_lines.forEach(line => line.visible = false);

            var current_index = new_active_index;
            // Show previous connections
            for (var i = 0; i < n_connections; i++) {
                var prev_chunk = source.data['prev_chunk'][current_index];
                if (prev_chunk !== new_active_index && prev_chunk !== current_index) {
                    prev_lines[i].data_source.data['x'] = [source.data['x'][current_index], source.data['x'][prev_chunk]];
                    prev_lines[i].data_source.data['y'] = [source.data['y'][current_index], source.data['y'][prev_chunk]];
                    prev_lines[i].visible = true;
                    current_index = prev_chunk;
                } else {
                    break;
                }
            }
            
            current_index = new_active_index;

            // Show next connections
            for (var i = 0; i < n_connections; i++) {
                var next_chunk = source.data['next_chunk'][current_index];
                if (next_chunk !== null && next_chunk !== current_index) {
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
        args=dict(
            source=source,
            text_div=text_div,
            prev_lines=prev_lines,
            next_lines=next_lines,
            n_connections=n_connections,
        ),
        code=common_js_code
        + """
        var index = cb_data.source.selected.indices[0];
        updateActiveChunk(source, text_div, prev_lines, next_lines, index, n_connections);
        """,
    )

    go_previous = CustomJS(
        args=dict(
            source=source,
            text_div=text_div,
            prev_lines=prev_lines,
            next_lines=next_lines,
            n_connections=n_connections,
        ),
        code=common_js_code
        + """
        var active_index = source.data['active_chunk'][0];
        if (active_index >= 0) {
            var prev_chunk = active_index - 1; //source.data['prev_chunk'][active_index];
            if (prev_chunk <= 0) {
                prev_chunk = 0;
            }
            updateActiveChunk(source, text_div, prev_lines, next_lines, prev_chunk, n_connections);
            source.data['active_index'][0] = prev_chunk
        }
        """,
    )

    go_next = CustomJS(
        args=dict(
            source=source,
            text_div=text_div,
            prev_lines=prev_lines,
            next_lines=next_lines,
            n_connections=n_connections,
        ),
        code=common_js_code
        + """
        var active_index = source.data['active_chunk'][0];
        if (active_index >= 0) {
            var next_chunk = active_index + 1; //source.data['prev_chunk'][active_index];
            if (next_chunk >= source.data['text'].length-1) {
                next_chunk = source.data['text'].length-1;
            }
            updateActiveChunk(source, text_div, prev_lines, next_lines, next_chunk, n_connections);
            source.data['active_index'][0] = next_chunk
        }
        """,
    )

    return callback, go_previous, go_next



def create_navigation_buttons(go_previous, go_next):
    button_left = Button(label="⬅️", width=50, styles=Styles(opacity="0.75"))
    button_left.js_on_click(go_previous)
    button_right = Button(label="➡️", width=50, styles=Styles(opacity="0.75"))
    button_right.js_on_click(go_next)
    return button_left, button_right


def prepare_data(
    chunk_collection, fields_to_include, highlight_first_document, document_to_show
):
    x, y, texts, hover_texts = [], [], [], []
    prev_chunks, next_chunks = [], []
    field_values = {field: [] for field in fields_to_include}

    for chunk in chunk_collection.chunks:
        if chunk.x is not None:
            if (
                highlight_first_document
                and "url" in chunk.attribs
                and chunk.attribs["url"] == document_to_show
            ) or not highlight_first_document:
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


def create_data_source(
    x, y, texts, hover_texts, field_values, prev_chunks, next_chunks, fields_names
):
    data = dict(
        x=x,
        y=y,
        text=texts,
        hover_texts=hover_texts,
        prev_chunk=prev_chunks,
        next_chunk=next_chunks,
        active_chunk=[-1] * len(prev_chunks),
        val_on_display=field_values[fields_names[0]],
    )
    data.update(field_values)
    for field in ["emoji", "keyword"]:
        if field in data:
            data[field + "_dynamic"] = data[field][
                ::
            ]  # the field that is updated with zooming in and out

    return ColumnDataSource(data=data)


def remove_elements(l, to_remove):
    l_copy = l[::]
    for x in to_remove:
        if x in l_copy:
            l_copy.remove(x)
    return l_copy


def visualize_chunks(
    chunk_collection: ChunkCollection,
    fields_to_include: Optional[List[str]] = None,
    n_connections: int=5,
    document_to_show: Optional[str]=None,
    return_html=False,
):
    """
        Visualise a chunk collection. 
        * Creates separate graphics for each field in fields_to_include
        * n_connections controls the number of lines in document order to show before / after each chunks
        * document_to_show, if None all documents are shown, if not none, only the document with this url will be displayed.
        * return_html. If True, retruns a string containing the html code. Else open a webbrowser and show the plot.
    """
    assert len(chunk_collection.chunks) > 0, "Empty chunk collection !"
    
    if fields_to_include is None:
        fields_to_include = list(
                set(
                    list(chunk_collection.chunks[0].attribs.keys())
                    + DEFAULT_VIS_FIELD
                )
            )
        for f in ["emoji_label_list"]:
            if f in fields_to_include:
                fields_to_include.remove(f)
        if "emoji" in fields_to_include:
            put_field_first("emoji", fields_to_include)

    # Prepare data
    existing_fields = []
    for f in fields_to_include:
        if f in chunk_collection.chunks[0].attribs:
            existing_fields.append(f)
    fields_to_include = existing_fields

    highlight_first_document = document_to_show is not None
    x, y, texts, hover_texts, field_values, prev_chunks, next_chunks = prepare_data(
        chunk_collection, existing_fields, highlight_first_document, document_to_show
    )

    # Create data source
    source = create_data_source(
        x,
        y,
        texts,
        hover_texts,
        field_values,
        prev_chunks,
        next_chunks,
        existing_fields,
    )

    # for i,field in enumerate(fields_to_include):
    #     if len(field) > 70:
    #         wrapped_field = wrap_str(field, max_line_len=70, skip_line_char="\n")
    #         source.data[wrapped_field] = source.data[field][::]
    #         fields_to_include[i] = wrapped_field

    # Create figure
    p = figure(
        height=700,
        width=900,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        height_policy="min",
        width_policy="max",
        max_height=1000,
        min_height=400,
        max_width=1200,
        min_width=600,
        sizing_mode="stretch_both",
    )
    # Create glyphs and color bars for each field
    field_glyphs = {}
    color_bars = {}
    additional_glyph = {}

    for field in fields_to_include:
        print(field, len(source.data[field]))
        if field == "emoji_list" or field == "keyword_list":
            glyph = p.scatter(
                x="x",
                y="y",
                size=10,
                source=source,
                fill_color="blue",
                alpha=0.1,
                visible=False,
            )
            additional_glyph[field] = Div()
            field_glyphs[field] = glyph
            color_bars[field] = Div()
        elif field == "keyword" or field == "emoji":
            labels = LabelSet(
                x="x",
                y="y",
                text=field + "_dynamic",
                x_offset=-10,
                y_offset=-5,
                source=source,
                visible=False,
            )
            glyph = p.scatter(
                x="x",
                y="y",
                size=10,
                source=source,
                fill_color="blue",
                alpha=0.1,
                visible=False,
            )
            p.add_layout(labels)
            additional_glyph[field] = labels
            field_glyphs[field] = glyph
            color_bars[field] = Div()
        else:
            if type(source.data[field][0]) == list:
                source.data[field] = [
                    str(l) for l in source.data[field]
                ]  

            if source.data[field] and type(source.data[field][0]) == str:
                source.data[field] = [
                    wrap_str(s, max_line_len=20, skip_line_char="\n")
                    for s in source.data[field]
                ]
                unique_values = list(set(source.data[field]))
                if len(unique_values) > 10:
                    source.data[field] = [
                        f"{s}" if len(s) > 20 else f"{s}" for s in source.data[field]
                    ]

            if all(isinstance(val, (int, float)) for val in source.data[field]):
                color_mapper = LinearColorMapper(
                    palette=Viridis256,
                    low=min(source.data[field]),
                    high=max(source.data[field]),
                )
            else:
                unique_values = list(set(source.data[field]))
                color_mapper = CategoricalColorMapper( #type: ignore
                    factors=unique_values, palette=cc.glasbey[: len(unique_values)]
                )

            glyph = p.scatter(
                x="x",
                y="y",
                size=10,
                source=source,
                fill_color={"field": field, "transform": color_mapper},
                line_color={"field": field, "transform": color_mapper},
                alpha=0.7,
                visible=False,
            )
            field_glyphs[field] = glyph

            color_bar = ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                major_label_text_font_size="8pt",
                border_line_color=None,
                location=(0, 0),
                title=wrap_str(field, max_line_len=80, skip_line_char="\n"),
                visible=False,
            )
            p.add_layout(color_bar, "right")
            color_bars[field] = color_bar

            additional_glyph[field] = Div()  # dummy object

    # Make the first field visible
    field_glyphs[fields_to_include[0]].visible = True
    color_bars[fields_to_include[0]].visible = True
    additional_glyph[fields_to_include[0]].visible = True
    source.data['active_chunk'][1] = fields_to_include[0]
    # Add tools
    # hover = HoverTool(renderers=[circles], tooltips=[("Text", "@hover_texts")])
    # p.add_tools(hover)
    hover = HoverTool(tooltips=[("Text", "@hover_texts")], renderers=[glyph for glyph in field_glyphs.values()])
    hover.tooltips = """
        @hover_texts | Value: @val_on_display
    """ # type: ignore
    p.add_tools(hover)
    p.add_tools(TapTool())

    # Create text div
    text_div = Div(
        width=300,
        height=500,
        text="",
        height_policy="max",
        width_policy="fixed",
        max_height=1000,
        min_height=400,
        max_width=300,
        min_width=150,
        sizing_mode="stretch_height",
    )

    # Create connection lines
    prev_lines = [
        p.line(x=[], y=[], line_color="#ffceb8", line_width=2, visible=False)
        for _ in range(n_connections)
    ]
    next_lines = [
        p.line(x=[], y=[], line_color="#f75002", line_width=2, visible=False)
        for _ in range(n_connections)
    ]

    # Create field selector
    field_selector = Select(
        title="Show:",
        value=fields_to_include[0],
        options=remove_elements(fields_to_include, ["emoji_list", "keyword_list"]),
        width=300,
        width_policy="fixed",
    )

    # Create JavaScript callback for updating visible glyph and color bar
    update_visible_elements = CustomJS(
        args=dict(
            source=source,
            field_glyphs=field_glyphs,
            color_bars=color_bars,
            additional_glyph=additional_glyph,
        ),
        code="""
        var selected_field = cb_obj.value;
        source.data['active_chunk'][1] = selected_field;
        for (let i = 0; i < source.data['val_on_display'].length; i++) {
            source.data['val_on_display'][i] = source.data[selected_field][i];
        }
        for (var field in field_glyphs) {
            field_glyphs[field].visible = (field === selected_field);
            color_bars[field].visible = (field === selected_field);
            additional_glyph[field].visible = (field === selected_field);
        }
    """,
    )

    # Call back to change the labels depending on the woom level
    original_ranges = {"x": [], "y": []}
    previous_index = {"index": -1}
    callback = CustomJS(
        args=dict(
            x_range=p.x_range,
            y_range=p.y_range,
            original_ranges=original_ranges,
            source=source,
            previous_index=previous_index,
        ),
        code="""
        if (original_ranges['x'].length == 0) {
            original_ranges['x'] = [x_range.start, x_range.end];
            original_ranges['y'] = [y_range.start, y_range.end];
        }
        var x_zoom = (x_range.end - x_range.start) / (original_ranges['x'][1] - original_ranges['x'][0]);
        var y_zoom = (y_range.end - y_range.start) / (original_ranges['y'][1] - original_ranges['y'][0]);

        var average_zoom = (x_zoom + y_zoom) / 2;  
        var min_level = 0.; // to adjust to change the range
        var max_level = 0.6; // the higher
        console.log(average_zoom);

        for (const field of ['emoji', 'keyword']) {         
            if (field in source.data) {
                var number_zoom_level = source.data[field+'_list'][0].split(",").length; //, is the separator
                
                var index_to_show = Math.floor(
                    (number_zoom_level - 1) * 
                    (average_zoom - min_level) / (max_level - min_level)
                );

                console.log(index_to_show);
                console.log(number_zoom_level);

                // Ensure the index is within the valid range
                index_to_show = Math.max(0, Math.min(index_to_show, number_zoom_level - 1));
                
                if (index_to_show != previous_index['index']) {
                    console.log(index_to_show);
                    previous_index['index'] = index_to_show;
                    for (let i = 0; i < source.data[field].length; i++) {
                        source.data[field+"_dynamic"][i] = source.data[field+'_list'][i].split(",")[number_zoom_level-1-index_to_show];
                        
                        //console.log( source.data[field+'_list'][i].split(",")[number_zoom_level-1-index_to_show]);
                        //console.log( source.data[field+'_list'][i]);
                        //console.log(number_zoom_level);
                    }
                }
            }
        }
    """,
    )

    # Attach the callback to both x and y ranges

    p.x_range.js_on_change("end", callback) #type: ignore
    p.y_range.js_on_change("end", callback) #type: ignore

    field_selector.js_on_change("value", update_visible_elements)

    # Create callbacks
    callback, go_previous, go_next = create_callbacks(
        source, text_div, prev_lines, next_lines, n_connections,
    )

    # Create navigation buttons
    button_left, button_right = create_navigation_buttons(go_previous, go_next)

    # Add callback to TapTool
    tap_tool = p.select(type=TapTool)[0]
    tap_tool.callback = callback

    # Create layout
    
    buttons = row(Div(width=100, sizing_mode="fixed"), button_left, button_right, sizing_mode="fixed", width=150, height=50)
    side = column(field_selector, text_div, buttons, sizing_mode="stretch_height", width=350)
    layout = row(side, p, sizing_mode="stretch_both")
    # p.aspect_ratio = 1

    # final processing, create html, return it or display in browser
    html = file_html(layout, CDN, "My Plot")
    if return_html:
        return html
    open_html_in_browser(html)
