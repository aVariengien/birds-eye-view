import streamlit as st
import plotly.graph_objects as go  # type: ignore
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
vis_options = ["emoji", "page", "chunk_id"]
vis_field = st.sidebar.selectbox("Visualization Field", vis_options)

# Run pipeline button
run_pipeline = st.sidebar.button("Run Pipeline")

# Main content
st.title("Chunk Map Visualization")


# Function to create ChunkCollection
def create_chunk_collection(document_names, max_chunk, pipeline_code, cache_file):
    if document_names == st.session_state.doc_names:
        chunks = st.session_state.chunks[:max_chunk]
    else:
        st.session_state.doc_names = document_names
        st.session_state.chunks = load_files(document_names, max_chunk=None, chunk_size=chunk_size)
        chunks = st.session_state.chunks[:max_chunk]
    pipeline = eval(pipeline_code)
    return ChunkCollection(chunks=chunks, pipeline=pipeline)


# Function to visualize chunks
def visualize_chunks(chunk_collection: ChunkCollection, vis_field: Optional[str]):
    x = [chunk.x for chunk in chunk_collection.chunks if chunk.x is not None]
    y = [chunk.y for chunk in chunk_collection.chunks if chunk.y is not None]
    texts = [
        chunk.display_text for chunk in chunk_collection.chunks if chunk.x is not None
    ]

    if vis_field == "chunk_id":
        display_values = [
            i for i, chunk in enumerate(chunk_collection.chunks) if chunk.x is not None
        ]
    elif vis_field != "":
        display_values = [
            chunk.attribs.get(vis_field, "")
            for chunk in chunk_collection.chunks
            if chunk.x is not None
        ]

    if type(display_values[0]) in [int, float]:
        colors = display_values[::]
        display_values = [""] * len(colors) #type:ignore
    else:
        colors = list(range(len(x)))

    print(type(display_values[0]))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Chunk Index"),
                ),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=display_values,
                textfont=dict(size=10),
                hovertext=[f"#{i}<br>" + t for i,t in enumerate(texts)],
                hoverinfo="text",
            ),
        ]
    )

    fig.update_layout(
        title="Chunk Map Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        hovermode="closest",
        height=800,
    )
    return fig


# Main logic
if run_pipeline:
    with st.spinner("Processing chunks..."):
        st.session_state.chunk_collection = create_chunk_collection(
            file_paths, max_chunk, pipeline_code, cache_file
        )
        st.session_state.chunk_collection.process_chunks()

    st.success("Pipeline completed!")
    
    # n = 0
    # # # Update display text for each chunk
    # for i, chunk in enumerate(st.session_state.chunk_collection.chunks):
    #     chunk.display_text = f"<b>Chunk #{i}</b><br>" + chunk.text
    #     if "emoji" in chunk.attribs:
    #         for emoji in chunk.attribs["emoji"].split(" "):
    #             if emoji not in ALL_EMOJIS:
    #                 print(f"error !! {chunk.attribs["emoji"]}")
    #                 chunk.attribs["emoji"] = ""
    #                 n += 1
    # print(n)

if st.session_state.chunk_collection is not None:
    # Visualize chunks
    fig = visualize_chunks(st.session_state.chunk_collection, vis_field)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Click 'Run Pipeline' to process and visualize the chunks.")

# Additional information
st.sidebar.info("Hover over points to see chunk text.")
