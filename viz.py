import streamlit as st
import plotly.graph_objects as go  # type: ignore
import plotly.express as px # type: ignore
from core import (
    ChunkCollection,
    Pipeline,
    OpenAITextProcessor,
    OpenAIEmbeddor,
    UMAPReductor,
    Chunk,
    DotProductLabelor,
)
from file_loading import load_files
from prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT, ALL_EMOJIS
from typing import Optional

from bokeh.plotting import figure # type: ignore
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, Div# type: ignore
from bokeh.layouts import column, row # type: ignore
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, Div, ColorBar, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap


# Streamlit page configuration
st.set_page_config(layout="wide")

# Initialize session state
if "chunk_collection" not in st.session_state:
    st.session_state.chunk_collection = None
if "pdf" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.doc_names = None

# Sidebar
st.sidebar.title("Configuration")

# File/Cache input
file_paths = st.sidebar.text_area("Enter file paths or URLs (one per line)", "data/plurality.pdf\ndata/another_file.pdf")
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
    DotProductLabelor(
        possible_labels=ALL_EMOJIS,
        nb_labels=3,
        cache_file=cache_file,
        embedding_model="text-embedding-3-small",
        key_name="emoji",
        prefix="",
    ),
    OpenAIEmbeddor(
        model="text-embedding-3-small", 
        cache_file=cache_file,
        batch_size=1000,
        ),
    UMAPReductor(
        verbose=True,
        n_neighbors=40,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    ),
], verbose=True)""",
)

# Max chunk slider
max_chunk = st.sidebar.slider("Max Chunks", min_value=10, max_value=2000, value=100)
if st.sidebar.checkbox("Use all chunks"):
    max_chunk = None

chunk_size = st.sidebar.slider("Chunk Size", min_value=10, max_value=2000, value=400)
# Visualization options
vis_options = ["emoji", "page", "chunk_id", "title", "url", "index"]
vis_field = st.sidebar.selectbox("Visualization Field", vis_options)

# Run pipeline button
run_pipeline = st.sidebar.button("Run Pipeline")

# Main content
st.title("Chunk Map Visualization")


# Function to create ChunkCollection
def create_chunk_collection(document_names, max_chunk, pipeline_code, cache_file):
    print(document_names)
    if document_names == st.session_state.doc_names:
        chunks = st.session_state.chunks[:max_chunk]
    else:
        st.session_state.doc_names = document_names
        st.session_state.chunks = load_files(document_names.split("\n"), max_chunk=None, chunk_size=chunk_size)
        chunks = st.session_state.chunks[:max_chunk]
    pipeline = eval(pipeline_code)
    return ChunkCollection(chunks=chunks, pipeline=pipeline)

use_qualitative_colors = st.sidebar.checkbox("Use qualitative colors")


def visualize_chunks(chunk_collection: ChunkCollection, vis_field: Optional[str], use_qualitative_colors):
    # Prepare data
    x = [chunk.x for chunk in chunk_collection.chunks if chunk.x is not None]
    y = [chunk.y for chunk in chunk_collection.chunks if chunk.y is not None]
    texts = [chunk.display_text for chunk in chunk_collection.chunks if chunk.x is not None]
    
    if vis_field == "chunk_id":
        display_values = [i for i, chunk in enumerate(chunk_collection.chunks) if chunk.x is not None]
    elif vis_field != "":
        display_values = [chunk.attribs.get(vis_field, "") for chunk in chunk_collection.chunks if chunk.x is not None]
    else:
        display_values = [""] * len(x)

    # Create ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        text=texts,
        display=display_values
    ))

    # Create figure
    p = figure(width=800, height=600, tools="pan,wheel_zoom,box_zoom,reset", active_scroll="wheel_zoom")
    
    if use_qualitative_colors:
        # Use qualitative colors for non-numeric data
        unique_values = list(set(display_values))
        color_mapper = factor_cmap('display', palette=Category10[10], factors=unique_values)
        circles = p.circle('x', 'y', size=10, source=source, color=color_mapper, alpha=0.7)
        color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
    else:
        # Use quantitative colors for numeric data or when qualitative is not selected
        if all(isinstance(val, (int, float)) for val in display_values):
            color_mapper = LinearColorMapper(palette="Viridis256", low=min(display_values), high=max(display_values))
        else:
            # Fallback to index-based coloring if data is not numeric
            color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=len(display_values)-1)
            source.data['color_index'] = list(range(len(display_values)))
            
        circles = p.circle('x', 'y', size=10, source=source, color={'field': 'color_index' if 'color_index' in source.data else 'display', 'transform': color_mapper}, alpha=0.7)
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
    # Add hover tool
    hover = HoverTool(renderers=[circles], tooltips=[("Text", "@text")])
    p.add_tools(hover)

    # Add tap tool
    p.add_tools(TapTool())

    # Create a Div to display the full text
    text_div = Div(width=200, height=600)

    # JavaScript callback for click events
    callback = CustomJS(args=dict(source=source, text_div=text_div), code="""
        var index = cb_data.source.selected.indices[0];
        var text = source.data['text'][index];
        var display_val = source.data['display'][index];
        text_div.text = '<div style="background-color: white; padding: 10px; border: 1px solid black;"> Val:'+display_val+'<br><br>' + text + '</div>';
    """)

    # Add the callback to the TapTool
    tap_tool = p.select(type=TapTool)[0]
    tap_tool.callback = callback

    # Create layout with plot and text div
    layout = row(text_div, p)
    st.bokeh_chart(layout)

    

# Main logic
if run_pipeline:
    with st.spinner(f"Processing ..."):
        st.session_state.chunk_collection = create_chunk_collection(
            file_paths, max_chunk, pipeline_code, cache_file
        )
        st.info(f"Succesfully created {len(st.session_state.chunk_collection.chunks)} chunks.")
        st.session_state.chunk_collection.process_chunks()

    st.success("Pipeline completed !")

if st.session_state.chunk_collection is not None:
    # Visualize chunks
    visualize_chunks(st.session_state.chunk_collection, vis_field, use_qualitative_colors)
    # fig = visualize_chunks(st.session_state.chunk_collection, vis_field)
    # st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click 'Run Pipeline' to process and visualize the chunks.")

# Additional information
st.sidebar.info("Hover over points to see chunk text.")
