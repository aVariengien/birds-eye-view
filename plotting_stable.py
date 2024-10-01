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

from bokeh.plotting import figure  # type: ignore
from bokeh.models import ColumnDataSource, HoverTool, Button, TapTool, CustomJS, Div  # type: ignore
from bokeh.layouts import column, row  # type: ignore
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TapTool,
    CustomJS,
    Div,
    ColorBar,
    LinearColorMapper,
)
from bokeh.layouts import column, row
from bokeh.palettes import Category20, Category10  # type: ignore
from bokeh.transform import factor_cmap  # type: ignore
from bokeh.plotting import figure, output_file, save
from markdownify import markdownify as md  # type: ignore
import urllib.parse
import os


def make_display_text(chunk: Chunk, vis_field: str):
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

    vis_field_value = f"""<br><br>{vis_field}: {chunk.attribs[vis_field]}"""

    text = f"{title}{headings}{chunk.display_text}{view_source}{page}{vis_field_value}"

    raw_text = (
        md(chunk.display_text, convert=[])
        + f"   |{vis_field}: {chunk.attribs[vis_field]}"
    )  # strip html
    return text, raw_text


def visualize_chunks(
    chunk_collection: ChunkCollection,
    vis_field: str,
    use_qualitative_colors: bool,
    n_connections: int = 5,
):
    x = []
    y = []
    texts = []
    hover_texts = []
    display_values = []
    prev_chunks = []
    next_chunks = []

    for i, chunk in enumerate(chunk_collection.chunks):
        if chunk.x is not None:
            x.append(chunk.x)
            y.append(chunk.y)
            fancy_text, raw_text = make_display_text(chunk, vis_field)
            texts.append(fancy_text)
            hover_texts.append(raw_text)
            display_values.append(chunk.attribs.get(vis_field, "") if vis_field else "")
            prev_chunks.append(chunk.previous_chunk_index)
            next_chunks.append(chunk.next_chunk_index)

    if type(display_values[0]) == list:
        display_values = [str(l) for l in display_values]

    if display_values and type(display_values[0]) == str:
        display_values = [
            wrap_str(s, max_line_len=20, skip_line_char="\n") for s in display_values
        ]
    # Create ColumnDataSource
    source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            text=texts,
            hover_texts=hover_texts,
            display=display_values,
            prev_chunk=prev_chunks,
            next_chunk=next_chunks,
            active_chunk=[-1] * len(prev_chunks),
        )
    )
    # Create figure
    height = 800
    width = 800
    p = figure(
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )

    # p.y_range.start = 0
    # p.y_range.end = (max(x) - min(x))*(height/width)
    if (
        all(isinstance(val, (float)) for val in display_values)
        or len(set([val for val in display_values])) > 30
    ):
        use_qualitative_colors = False

    if use_qualitative_colors:
        # Use qualitative colors for non-numeric data
        unique_values = list(set(display_values))
        color_mapper = factor_cmap(
            "display",
            palette=Category20[max(min(len(unique_values), 20), 3)],
            factors=unique_values,
        )
        circles = p.circle(
            "x", "y", size=10, source=source, color=color_mapper, alpha=0.7
        )
        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            width=8,
            location=(0, 0),
            title=wrap_str(vis_field, max_line_len=100, skip_line_char="\n"),
        )
        p.add_layout(color_bar, "right")
    else:
        # Use quantitative colors for numeric data or when qualitative is not selected
        if all(isinstance(val, (int, float)) for val in display_values):
            color_mapper = LinearColorMapper(
                palette="Viridis256", low=min(display_values), high=max(display_values)
            )
        else:
            # Fallback to index-based coloring if data is not numeric
            color_mapper = LinearColorMapper(
                palette="Viridis256", low=0, high=len(display_values) - 1
            )
            source.data["color_index"] = list(range(len(display_values)))

        circles = p.circle(
            "x",
            "y",
            size=10,
            source=source,
            color={
                "field": "color_index" if "color_index" in source.data else "display",
                "transform": color_mapper,
            },
            alpha=0.7,
        )

        color_bar = ColorBar(
            color_mapper=color_mapper,
            width=8,
            location=(0, 0),
            title=wrap_str(vis_field, max_line_len=100, skip_line_char="\n"),
        )
        p.add_layout(color_bar, "left")

    # Add hover tool
    hover = HoverTool(renderers=[circles], tooltips=[("Text", "@hover_texts")])
    p.add_tools(hover)

    # Add tap tool
    p.add_tools(TapTool())

    # Create a Div to display the full text

    text_div = Div(width=300, height=600, text="", id="active-chunk-div")

    # Create multiple lines for previous and next connections
    prev_lines = [
        p.line(x=[], y=[], line_color="#ffceb8", line_width=2, visible=False)
        for _ in range(n_connections)
    ]
    next_lines = [
        p.line(x=[], y=[], line_color="#f75002", line_width=2, visible=False)
        for _ in range(n_connections)
    ]

    common_js_code = """
    function updateActiveChunk(source, text_div, prev_lines, next_lines, new_active_index, n_connections) {
        if (new_active_index !== null && new_active_index >= 0) {
            source.data['active_chunk'][0] = new_active_index;
            source.selected.indices = [new_active_index];
            var text = source.data['text'][new_active_index] || '';
            text_div.text = text;

            // Hide all lines initially
            prev_lines.forEach(line => line.visible = false);
            next_lines.forEach(line => line.visible = false);
            console.log("hi1")

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

            console.log("hi2")

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
            console.log("hi3")
            prev_lines.forEach(line => line.data_source.change.emit());
            next_lines.forEach(line => line.data_source.change.emit());
            source.change.emit();
            console.log("hi4")
        }
    }
    """

    # JavaScript callback for click events
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
                var prev_chunk = source.data['prev_chunk'][active_index];
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
                var next_chunk = source.data['next_chunk'][active_index];
                updateActiveChunk(source, text_div, prev_lines, next_lines, next_chunk, n_connections);
                source.data['active_index'][0] = next_chunk
            }
        """,
    )
    button_left = Button(label="⬅️", width=50)
    button_left.js_on_click(go_previous)
    button_right = Button(label="➡️", width=50)
    button_right.js_on_click(go_next)

    # Add the callback to the TapTool
    tap_tool = p.select(type=TapTool)[0]
    tap_tool.callback = callback

    # Create layout with plot and text div
    buttons = row(button_left, button_right)
    side = column(text_div, buttons)
    layout = row(side, p)
    p.aspect_ratio = 1
    return layout