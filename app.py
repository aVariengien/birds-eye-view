import streamlit as st
from birds_eye_view.core import *
from birds_eye_view.plotting import visualize_chunks
from birds_eye_view.file_loading import load_files
from birds_eye_view.prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT, ALL_EMOJIS

from streamlit.components.v1 import html # type: ignore
import os

from bokeh.resources import CDN # type: ignore

import platform

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
        api_key=api_key
    )
    st.session_state.chunk_collection.apply_step(embedding_search)
    st.session_state.vis_field = "Search:" + search_prompt
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


def is_streamlit_cloud():
    # Check for environment variables that are specific to Streamlit Cloud
    return platform.processor() == ""


# Welcome
if st.session_state.chunk_collection is None:
    st.title("🦉 Bird's Eye View")
    st.markdown("""
    Welcome to Bird's Eye View!
    Start by loading documents or use a pre-loaded collections below 👇.

    """)

    # Buttons to load default collections
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📚 Load MMLU"):
            st.session_state.chunk_collection = load_chunk_collection("mmlu.json")
            st.rerun()
        st.markdown("7000 multiple-choice questions from the MMLU benchmark, a comprehensive LLM test covering 57 subjects.")


    with col2:
        if st.button("⛵️ Load *The Voyage of the Beagle*"):
            st.session_state.chunk_collection = load_chunk_collection("beagle.json")
            st.rerun()
        st.markdown("*The Voyage of the Beagle* by Charles Darwin, a 300-page travel diary of his 1831-1836 expedition in South America that made the biologist famous.")



# Sidebar
st.sidebar.title("🦉 Bird's eye view")
st.sidebar.markdown("*Take a look at your documents from above.*")

if os.getenv("OPENAI_API_KEY") is not None:
    api_key = os.getenv("OPENAI_API_KEY")
else:
    try:
        if "OpenAI_key" not in st.secrets:
            api_key = st.sidebar.text_input("Enter your OpenAI API key:")
        else:
            api_key = st.secrets["OpenAI_key"]
    except:
        if os.getenv("OPENAI_API_KEY") is None:
            api_key = st.sidebar.text_input("Enter your OpenAI API key:")


# File/Cache input
file_paths = st.sidebar.text_area(
    "Enter URLs (one per line)" + is_streamlit_cloud()*" or file paths (if hosted locally)",
    """""",
)

run_pipeline = st.sidebar.button("⏩️ Run Pipeline")

# Additional information
st.sidebar.info("Hover over points to see chunk text. Click to highlight a chunk. Use the arrows to navigate to the previous/next chunk.")

# Add EmbeddingSearch configuration

with st.sidebar:
    with st.form(key="Fuzzy Search"):
        st.header("Embedding Search")
        st.markdown("*Enter a search query to highlight the chunks that relate to it.*")

        search_prompt = st.text_area(
            "Search Prompt", placeholder="Enter a prompt to show the related chunks.", 
        )

        st.session_state.run_search = st.form_submit_button("🔎 Run Search")


with st.sidebar.expander("Advanced parameters", expanded=False, icon="⚙️"):
    cache_dir = st.text_input("Cache folder", "cache/")

    if is_streamlit_cloud():
        pipeline_code = """Pipeline([OpenAIEmbeddor(
                model=embedding_model, 
                batch_size=2000,
                api_key=api_key
                ),
            DotProductLabelor(
                possible_labels=ALL_EMOJIS,
                nb_labels=3,
                embedding_model=embedding_model,
                key_name="emoji",
                prefix="",
                api_key=api_key
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
        ], verbose=True)"""
    else:
        pipeline_code = st.text_area("Pipeline Code", value="""Pipeline([OpenAIEmbeddor(
                model=embedding_model, 
                batch_size=2000,
                api_key=api_key,
                cache_dir="bev_cache",
                ),
            DotProductLabelor(
                possible_labels=ALL_EMOJIS,
                nb_labels=3,
                embedding_model=embedding_model,
                key_name="emoji",
                prefix="",
                api_key=api_key
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
        ], verbose=True)""")
    if not os.path.exists(cache_dir):
        with open(cache_dir, 'a'):
            os.utime(cache_dir, None)

    # Max chunk slider
    max_chunk = st.slider("Max Chunks", min_value=10, max_value=2000, value=100)
    if st.checkbox("Use all chunks", value=True):
        max_chunk = None

    chunk_size = st.slider(
        "Chunk Size", min_value=50, max_value=2000, value=600, step=50
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
        label="Path depth", min_value=0, max_value=20, step=1, value=5
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

@st.cache_data
def save(_collection, name):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    print("Hello")
    return _collection.save(return_data=True)

with st.sidebar.expander("Save and load", expanded=False, icon="💾"):
    st.header("Manage Chunk Collection")

    if st.session_state.chunk_collection is not None:
        st.download_button(
            label="💾 Download collection",
            data=save(st.session_state.chunk_collection, "hello"),
            file_name="chunk_collection.json",
        )
    
    uploaded_file = st.file_uploader("⬆️ Load Collection")
    if uploaded_file is not None:
        # To read file as bytes:
        st.session_state.chunk_collection = ChunkCollection.load_from_file(uploaded_file)

    # Text input for plot name
    if st.session_state.bokeh_plot is not None:
        st.download_button(
            label="📊 Download Interactive Plot",
            data=st.session_state.bokeh_plot,
            file_name=f"birds_eye_view_plot.html",
        )



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


    fields_to_show = st.session_state.viz_options[::]
    fields_to_show.remove("emoji_label_list")
    html_str = visualize_chunks(
        st.session_state.chunk_collection,
        fields_to_show, # st.session_state.vis_field viz_options
        n_connections=n_connections,
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