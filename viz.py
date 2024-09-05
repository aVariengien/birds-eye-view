import streamlit as st
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
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
import urllib.parse
from file_loading import load_files, wrap_str
from prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT, ALL_EMOJIS
from typing import Optional
from markdownify import markdownify as md  # type: ignore

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
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.palettes import Category20, Category10  # type: ignore
from bokeh.transform import factor_cmap  # type: ignore
from bokeh.plotting import figure, output_file, save
from streamlit.components.v1 import html
import os


def update_search():
    embedding_search = EmbeddingSearch(
        prompt=search_prompt,
        threshold=search_threshold,
        cache_file=cache_file,
        embedding_model=embedding_model,
    )
    st.session_state.chunk_collection.apply_step(embedding_search)
    st.session_state.vis_field = "Search:" + search_prompt
    st.success("EmbeddingSearch completed!")

# Function to create ChunkCollection
def create_chunk_collection(document_names, max_chunk, pipeline_code, cache_file):
    print(document_names)
    if document_names == st.session_state.doc_names:
        chunks = st.session_state.chunks[:max_chunk]
    else:
        st.session_state.doc_names = document_names
        st.session_state.chunks = load_files(
            document_names.split("\n"), max_chunk=None, chunk_size=chunk_size
        )
        chunks = st.session_state.chunks[:max_chunk]
    pipeline = eval(pipeline_code)
    return ChunkCollection(chunks=chunks, pipeline=pipeline)


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

    text = (
        f"{title}{headings}{chunk.display_text}{view_source}{page}{vis_field_value}"
    )

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
            active_chunk=[-1]*len(prev_chunks),
        )
    )
    # Create figure
    height = 600
    width = 1000
    p = figure(
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )

    # p.y_range.start = 0
    # p.y_range.end = (max(x) - min(x))*(height/width)

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
    prev_lines = [p.line(x=[], y=[], line_color="#ffceb8", line_width=2, visible=False) for _ in range(n_connections)]
    next_lines = [p.line(x=[], y=[], line_color="#f75002", line_width=2, visible=False) for _ in range(n_connections)]

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
            n_connections=n_connections
        ),
        code=common_js_code + """
        var index = cb_data.source.selected.indices[0];
        updateActiveChunk(source, text_div, prev_lines, next_lines, index, n_connections);
        """
    )

    go_previous = CustomJS(
        args=dict(
            source=source,
            text_div=text_div,
            prev_lines=prev_lines,
            next_lines=next_lines,
            n_connections=n_connections
        ),
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
        args=dict(
            source=source,
            text_div=text_div,
            prev_lines=prev_lines,
            next_lines=next_lines,
            n_connections=n_connections
        ),
        code=common_js_code + """
            var active_index = source.data['active_chunk'][0];
            if (active_index >= 0) {
                var next_chunk = source.data['next_chunk'][active_index];
                updateActiveChunk(source, text_div, prev_lines, next_lines, next_chunk, n_connections);
                source.data['active_index'][0] = next_chunk
            }
        """
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
    st.session_state.bokeh_plot = layout
    st.bokeh_chart(layout, use_container_width=False)
    p.aspect_ratio = 1
    

def save_bokeh_plot(plot, filename):
    if not filename.endswith('.html'):
        filename += '.html'

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    full_path = os.path.join('plots', filename)

    # Generate standalone HTML file
    html = file_html(plot, CDN, "My Plot")

    # Save the HTML to a file
    with open(full_path, 'w') as f:
        f.write(html)

    return full_path



DEFAULT_VIS_FIELD = ["title", "doc_position", "page", "url", "index"]
# Initialize session state
if "chunk_collection" not in st.session_state:
    st.session_state.chunk_collection = None
    st.session_state.chunks = None
    st.session_state.doc_names = None
    st.session_state.viz_options = DEFAULT_VIS_FIELD
    st.session_state.vis_field = DEFAULT_VIS_FIELD[0]
    st.session_state.bokeh_plot = None
elif st.session_state.chunk_collection is not None:
    st.session_state.viz_options = list(
        set(
            list(st.session_state.chunk_collection.chunks[0].attribs.keys())
            + DEFAULT_VIS_FIELD
        )
    )




# Main content
st.set_page_config(
    page_title="🦉 Bird's eye view",
    page_icon="🦉",
    layout="wide"
)

# Sidebar
st.sidebar.title("Configuration")

# File/Cache input
file_paths = st.sidebar.text_area(
    "Enter file paths or URLs (one per line)",
    """https://thezvi.substack.com/p/ai-79-ready-for-some-football
https://thezvi.substack.com/p/ai-78-some-welcome-calm
https://thezvi.substack.com/p/ai-77-a-few-upgrades
https://thezvi.substack.com/p/danger-ai-scientist-danger
https://thezvi.substack.com/p/ai-76-six-short-stories-about-openai
https://thezvi.substack.com/p/ai-75-math-is-easier
https://thezvi.substack.com/p/ai-74-gpt-4o-mini-me-and-llama-3""",
)
cache_file = st.sidebar.text_input("Cache file", "cache/cache.json")

# Pipeline code input

# OpenAITextProcessor(
#     system_prompt=MULTIPLE_EMOJI_PROMPT,
#     max_workers=10,
#     update_text=False,
#     output_key="emoji",
#     cache_file=cache_file
# ),


pipeline_code = st.sidebar.text_area(
    "Pipeline Code",
    """Pipeline([
    OpenAIEmbeddor(
        model=embedding_model, 
        cache_file=cache_file,
        batch_size=1000,
        ),
    DotProductLabelor(
        possible_labels=ALL_EMOJIS,
        nb_labels=3,
        cache_file=cache_file,
        embedding_model=embedding_model,
        key_name="emoji",
        prefix="",
    ),
    UMAPReductor(
        verbose=True,
        n_neighbors=20,
        min_dist=0.05,
        random_state=42,
        n_jobs=1,
    ),
], verbose=True)""",
)

# Max chunk slider
max_chunk = st.sidebar.slider("Max Chunks", min_value=10, max_value=2000, value=100)
if st.sidebar.checkbox("Use all chunks", value=True):
    max_chunk = None

chunk_size = st.sidebar.slider(
    "Chunk Size", min_value=50, max_value=2000, value=400, step=50
)
# Visualization options
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["text-embedding-3-large", "text-embedding-3-small"],
    index=0,
)


st.session_state.vis_field = st.sidebar.selectbox(
    "Visualization Field",
    st.session_state.viz_options,
    index=(
        0
        if st.session_state.vis_field is None
        else st.session_state.viz_options.index(st.session_state.vis_field)
    ),
    placeholder=(
        "" if st.session_state.vis_field is None else st.session_state.vis_field
    ),
)

run_pipeline = st.sidebar.button("Run Pipeline")

use_qualitative_colors = st.sidebar.checkbox("Use qualitative colors", value=True)

n_connections = st.sidebar.slider(label="Path depth",min_value=1, max_value=20, step=1, value=5)

# Text input for plot name
plot_name = st.sidebar.text_input("Plot name:", "my_plot")

# Button to save the plot
if st.sidebar.button("Save Plot"):
    if st.session_state.bokeh_plot is not None:
        try:
            saved_path = save_bokeh_plot(st.session_state.bokeh_plot, plot_name)
            st.success(f"Plot saved successfully as {saved_path}")
        except Exception as e:
            st.error(f"An error occurred while saving the plot: {str(e)}")
    else:
        st.warning("No plot available to save. Please create a plot first.")
    print(st.session_state.bokeh_plot)


# Add EmbeddingSearch configuration
st.sidebar.header("EmbeddingSearch Configuration")
search_prompt = st.sidebar.text_area(
    "Search Prompt", placeholder="Enter a prompt to show the related chunks."
)

search_threshold = st.sidebar.slider(
    "Search Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01,
    on_change=update_search,
)

assert embedding_model is not None
run_search = st.sidebar.button("Run Search")

# Main logic
if run_pipeline:
    with st.spinner(f"Processing ..."):
        st.session_state.chunk_collection = create_chunk_collection(
            file_paths, max_chunk, pipeline_code, cache_file
        )
        st.info(
            f"Succesfully created {len(st.session_state.chunk_collection.chunks)} chunks."
        )
        st.session_state.chunk_collection.process_chunks()

    st.success("Pipeline completed !")

import numpy as np

if run_search and st.session_state.chunk_collection is not None:
    update_search()


if st.session_state.chunk_collection is not None: # and refresh:
    # Visualize chunks
    assert type(st.session_state.vis_field) == str
    visualize_chunks(
        st.session_state.chunk_collection,
        st.session_state.vis_field,
        use_qualitative_colors if not run_search else False,
        n_connections,
    )
elif st.session_state.chunk_collection is None:
    st.info("Click 'Run Pipeline' to process and visualize the chunks.")

# Additional information
st.sidebar.info("Hover over points to see chunk text.")
