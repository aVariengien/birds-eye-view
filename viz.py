import streamlit as st
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from birds_eye_view.core import (
    ChunkCollection,
    Pipeline,
    OpenAITextProcessor,
    OpenAIEmbeddor,
    UMAPReductor,
    Chunk,
    DotProductLabelor,
    HierachicalLabelMapper,
    EmbeddingSearch,
)
from birds_eye_view.plotting import visualize_chunks
from birds_eye_view.file_loading import load_files, wrap_str
import numpy as np
from birds_eye_view.prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT, ALL_EMOJIS
from typing import Optional

from streamlit.components.v1 import html # type: ignore
import os

import pickle
from bokeh.resources import CDN # type: ignore
from bokeh.embed import file_html # type: ignore
import re

def get_body_content(html_string):
    # Use regular expression to find the content between <body> and </body> tags
    body_pattern = re.compile(r'<body[^>]*>(.*?)</body>', re.DOTALL | re.IGNORECASE)
    match = body_pattern.search(html_string)

    if match:
        # Return the content of the body tag
        return """<div class="test!!" style="position: relative; display: block; left: 0px; top: 0px; width: 1000px; height: 800px; margin: 0px;">     <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.3.min.js"></script>
    <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.4.3.min.js"></script>
    <script type="text/javascript">
        Bokeh.set_log_level("info");
    </script>""" + match.group(1).strip() + "</div>"
    else:
        # Return None if no body tag is found
        return None

# Ensure the 'saved_collections' directory exists
if not os.path.exists("saved_collections"):
    os.makedirs("saved_collections")


def save_bokeh_plot(plot, filename):
    if not filename.endswith(".html"):
        filename += ".html"

    # Ensure the 'plots' directory exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    full_path = os.path.join("plots", filename)

    # Generate standalone HTML file
    html = file_html(plot, CDN, "My Plot")

    # Save the HTML to a file
    with open(full_path, "w") as f:
        f.write(html)

    return full_path


# Function to save the chunk collection
def save_chunk_collection(collection, filename):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    full_path = os.path.join("saved_collections", filename)
    with open(full_path, "wb") as f:
        pickle.dump(collection, f)
    return full_path


# Function to load the chunk collection
def load_chunk_collection(filename):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    full_path = os.path.join("saved_collections", filename)
    with open(full_path, "rb") as f:
        return pickle.load(f)


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
    print("Loaded files!")
    return ChunkCollection(chunks=chunks, pipeline=pipeline)




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
st.set_page_config(page_title="🦉 Bird's eye view", page_icon="🦉", layout="wide")

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
cache_file = st.sidebar.text_input("Cache file", "cache/new.json")

if not os.path.exists(cache_file):
    with open(cache_file, 'a'):
        os.utime(cache_file, None)
# Pipeline code input

# OpenAITextProcessor(
#     system_prompt=MULTIPLE_EMOJI_PROMPT,
#     max_workers=10,
#     update_text=False,
#     output_key="emoji",
#     cache_file=cache_file
# ),


    # DotProductLabelor(
    #     possible_labels=ALL_EMOJIS,
    #     nb_labels=3,
    #     cache_file=cache_file,
    #     embedding_model=embedding_model,
    #     key_name="emoji",
    #     prefix="",
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
    HierachicalLabelMapper(
        number_levels=10,
        key_name="emoji",
    )
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
    index= 0,
)

if st.session_state.viz_options is not None:
    first_element = st.session_state.viz_options[0]
    new_first_idx = st.session_state.viz_options.index(st.session_state.vis_field)
    st.session_state.viz_options[0] = st.session_state.vis_field
    st.session_state.viz_options[new_first_idx] = first_element

run_pipeline = st.sidebar.button("Run Pipeline")

# Additional information
st.sidebar.info("Hover over points to see chunk text. Click to highlight a chunk.")


use_qualitative_colors = st.sidebar.checkbox("Use qualitative colors", value=True)

n_connections = st.sidebar.slider(
    label="Path depth", min_value=1, max_value=20, step=1, value=5
)

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


if run_search and st.session_state.chunk_collection is not None:
    update_search()

highlight_first_document = st.sidebar.checkbox("Check to only show the first document, but use the other docs in creating the embeddings.")

if st.sidebar.button("Refresh") or st.session_state.chunk_collection is not None:  # and refresh:
    # Visualize chunks
    assert type(st.session_state.vis_field) == str
    plot = visualize_chunks(
        st.session_state.chunk_collection,
        st.session_state.viz_options, # st.session_state.vis_field viz_options
        #use_qualitative_colors if not run_search else False,
        n_connections,
        highlight_first_document=highlight_first_document,
        document_to_show = file_paths.split("\n")[0],
    )
    st.session_state.bokeh_plot = plot
    html(file_html(plot, CDN, "My Plot"), height=550)
    # print(" +++++ ")
    # print(file_html(plot, CDN, "My Plot"))
    #st.bokeh_chart(plot, use_container_width=False)
elif st.session_state.chunk_collection is None:
    st.info("Click 'Run Pipeline' to process and visualize the chunks.")


st.sidebar.header("Manage Chunk Collection")
save_name = st.sidebar.text_input("Enter name to save:", "my_collection")

if st.sidebar.button("💾 Save Collection"):
    if (
        "chunk_collection" in st.session_state
        and st.session_state.chunk_collection is not None
    ):
        try:
            saved_path = save_chunk_collection(
                st.session_state.chunk_collection, save_name
            )
            st.success(f"Collection saved successfully as {saved_path}")
        except Exception as e:
            st.error(f"An error occurred while saving the collection: {str(e)}")
    else:
        st.warning("No chunk collection available to save.")

# Load section
load_name = st.sidebar.text_input("Enter name to load:", "my_collection")

if st.sidebar.button("⬆️ Load Collection"):
    try:
        loaded_collection = load_chunk_collection(load_name)
        st.session_state.chunk_collection = loaded_collection
        st.success(f"Collection '{load_name}' loaded successfully!")
        st.rerun()
    except FileNotFoundError:
        st.error(
            f"File '{load_name}.pkl' not found in the 'saved_collections' directory."
        )
    except Exception as e:
        st.error(f"An error occurred while loading the collection: {str(e)}")


# Text input for plot name
st.sidebar.header("📊 Save Interactive Plot")
plot_name = st.sidebar.text_input("Plot name:", "my_plot")

# Button to save the plot
if st.sidebar.button("📊 Save Plot"):
    if st.session_state.bokeh_plot is not None:
        try:
            saved_path = save_bokeh_plot(st.session_state.bokeh_plot, plot_name)
            st.success(f"Plot saved successfully as {saved_path}")
        except Exception as e:
            st.error(f"An error occurred while saving the plot: {str(e)}")
    else:
        st.warning("No plot available to save. Please create a plot first.")
