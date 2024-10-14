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

# Ensure the 'saved_collections' directory exists
if not os.path.exists("saved_collections"):
    os.makedirs("saved_collections")


def put_field_first(x, l):
    prev = l[0]
    if x in l:
        old_x_idx = l.index(x)
        l[old_x_idx] = prev
        l[0] = x
    else:
        l.insert(0, x)


def save_bokeh_plot(plot, filename):
    if not filename.endswith(".html"):
        filename += ".html"

    # Ensure the 'plots' directory exists
    if not os.path.exists("plots"):
        os.makedirs("plots")

    full_path = os.path.join("plots", filename)

    # Save the HTML to a file
    with open(full_path, "w") as f:
        f.write(plot)

    return full_path


# Function to save the chunk collection
def save_chunk_collection(collection, filename):
    if not filename.endswith(".json"):
        filename += ".json"
    full_path = os.path.join("saved_collections", filename)
    collection.save(full_path)
    return full_path


# Function to load the chunk collection
def load_chunk_collection(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    full_path = os.path.join("saved_collections", filename)

    return ChunkCollection.load_from_file(full_path)


def update_search():
    embedding_search = EmbeddingSearch(
        prompt=search_prompt,
        threshold=search_threshold,
        embedding_model=embedding_model,
    )
    st.session_state.chunk_collection.apply_step(embedding_search)
    st.session_state.vis_field = "Search:" + search_prompt
    st.success("EmbeddingSearch completed!")
    st.session_state.run_search = True


# Function to create ChunkCollection
def create_chunk_collection(document_names, max_chunk, pipeline_code):
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




DEFAULT_VIS_FIELD = ["emoji", "title", "doc_position", "page", "url", "index"]
# Initialize session state
if "chunk_collection" not in st.session_state:
    st.session_state.chunk_collection = None
    st.session_state.chunks = None
    st.session_state.doc_names = None
    st.session_state.viz_options = DEFAULT_VIS_FIELD
    st.session_state.vis_field = DEFAULT_VIS_FIELD[0]
    st.session_state.bokeh_plot = None

# Main content
st.set_page_config(page_title="🦉 Bird's eye view", page_icon="🦉", layout="wide")

# Sidebar
st.sidebar.title("🦉 Bird's eye view")
st.sidebar.markdown("*Take a look at your documents from above.*")

# File/Cache input
file_paths = st.sidebar.text_area(
    "Enter file paths (e.g. data/paper.pdf) or URLs (one per line)",
    """https://thezvi.substack.com/p/ai-79-ready-for-some-football""",
)

run_pipeline = st.sidebar.button("⏩️ Run Pipeline")

# Additional information
st.sidebar.info("Hover over points to see chunk text. Click to highlight a chunk.")

# Add EmbeddingSearch configuration
st.sidebar.header("Embedding Search")
st.sidebar.markdown("*Enter a search query to highlight the chunks that relate to it.*")
search_prompt = st.sidebar.text_area(
    "Search Prompt", placeholder="Enter a prompt to show the related chunks.", on_change=update_search
)

st.session_state.run_search = st.sidebar.button("🔎 Run Search")


with st.sidebar.expander("Advanced parameters", expanded=False, icon="⚙️"):
    cache_dir = st.text_input("Cache folder", "cache/")

    pipeline_code = st.text_area(
        "Pipeline Code",
        """Pipeline([
        OpenAIEmbeddor(
            model=embedding_model, 
            cache_dir=cache_dir,
            batch_size=1000,
            ),
        DotProductLabelor(
            possible_labels=ALL_EMOJIS,
            nb_labels=3,
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
            max_number_levels=10,
            key_name="emoji",
        )
    ], verbose=True)""",
    )
    if not os.path.exists(cache_dir):
        with open(cache_dir, 'a'):
            os.utime(cache_dir, None)

    # Max chunk slider
    max_chunk = st.slider("Max Chunks", min_value=10, max_value=2000, value=100)
    if st.checkbox("Use all chunks", value=True):
        max_chunk = None

    chunk_size = st.slider(
        "Chunk Size", min_value=50, max_value=2000, value=400, step=50
    )
    # Visualization options
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-large", "text-embedding-3-small"],
        index=0,
    )

    st.session_state.vis_field = st.selectbox(
        "Visualization Field",
        st.session_state.viz_options,
        index= 0,
    )

    if st.session_state.viz_options is not None:
        put_field_first(st.session_state.vis_field, st.session_state.viz_options)
        
    n_connections = st.slider(
        label="Path depth", min_value=1, max_value=20, step=1, value=5
    )
    highlight_first_document = st.checkbox("Check to only show the first document, but use the other docs in creating the embeddings.")
    
    search_threshold = st.slider(
        "Search Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        on_change=update_search,
    )
with st.sidebar.expander("Save and load", expanded=False, icon="💾"):
    st.header("Manage Chunk Collection")

    save_name = st.text_input("Enter name to save:", "my_collection")

    if st.button("💾 Save Collection"):
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
    load_name = st.text_input("Enter name to load:", "my_collection")

    if st.button("⬆️ Load Collection"):
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
    st.header("📊 Save Interactive Plot")
    plot_name = st.text_input("Plot name:", "my_plot")

    # Button to save the plot
    if st.button("📊 Save Plot"):
        if st.session_state.bokeh_plot is not None:
            try:
                saved_path = save_bokeh_plot(st.session_state.bokeh_plot, plot_name)
                st.success(f"Plot saved successfully as {saved_path}")
            except Exception as e:
                st.error(f"An error occurred while saving the plot: {str(e)}")
        else:
            st.warning("No plot available to save. Please create a plot first.")




# Main logic
if run_pipeline:
    with st.spinner(f"Processing ..."):
        st.session_state.chunk_collection = create_chunk_collection(
            file_paths, max_chunk, pipeline_code, 
        )
        st.info(
            f"Succesfully created {len(st.session_state.chunk_collection.chunks)} chunks."
        )
        st.session_state.chunk_collection.process_chunks()

    st.success("Pipeline completed !")
    st.rerun()


if st.session_state.run_search and st.session_state.chunk_collection is not None:
    update_search()


if st.sidebar.button("Refresh") or st.session_state.chunk_collection is not None:  # and refresh:
    # Visualize chunks
    assert type(st.session_state.vis_field) == str

    if st.session_state.chunk_collection is not None:
        st.session_state.viz_options = list(
            set(
                list(st.session_state.chunk_collection.chunks[0].attribs.keys())
                + DEFAULT_VIS_FIELD
            )
        )
    else:
        st.session_state.viz_options = list(
            DEFAULT_VIS_FIELD
        )

    put_field_first("emoji", st.session_state.viz_options)
    if st.session_state.run_search:
        search_field = "Search:" + search_prompt
        put_field_first(search_field, st.session_state.viz_options)


    html_str = visualize_chunks(
        st.session_state.chunk_collection,
        st.session_state.viz_options, # st.session_state.vis_field viz_options
        n_connections,
        document_to_show = file_paths.split("\n")[0] if highlight_first_document else None,
        return_html=True
    )
    st.session_state.bokeh_plot = html_str
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 96px;
                    padding-bottom: 0rem;
                    padding-left: 40px;
                    padding-right: 20px;
                }
               .css-1d391kg {
                    padding-top: 0rem;
                    padding-right: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                }
                .ea3mdgi5 
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
               iframe {
                    height: 80vh !important;
                }
        </style>
        """, unsafe_allow_html=True)
    html(html_str, height=600)
elif st.session_state.chunk_collection is None:
    st.info("Click 'Run Pipeline' to process and visualize the chunks.")