# %%
import importlib
import birds_eye_view.plotting
importlib.reload(birds_eye_view.plotting)
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

# %%
importlib.reload(birds_eye_view.file_loading)
url = "data/mmlu_test.json"
chunks = birds_eye_view.file_loading.load_files(file_paths=[url], max_chunk=100)


# %%
cache_file = "beagle.json"
embedding_model = "text-embedding-3-large"

pipeline = Pipeline([
    OpenAIEmbeddor(
        model=embedding_model, 
        cache_file=cache_file,
        batch_size=2000,
        ),
    DotProductLabelor(
        nb_labels=3,
        embedding_model=embedding_model,
        key_name="emoji",
    ),
    UMAPReductor(
        verbose=True,
        n_neighbors=20,
        min_dist=0.05,
        random_state=None,
        n_jobs=8,
    ),
    HierachicalLabelMapper(
        max_number_levels=10,
        key_name="emoji",
    )
], verbose=True)


# %%

collection = ChunkCollection(pipeline=pipeline, chunks=chunks)
collection.process_chunks()

# %%

all_chunk = birds_eye_view.file_loading.load_files(file_paths=[url], max_chunk=None)
big_collection = ChunkCollection(pipeline=pipeline, chunks=all_chunk)
big_collection.process_chunks()

# %%

beagle_chunks = birds_eye_view.file_loading.load_files(file_paths=["https://www.gutenberg.org/cache/epub/944/pg944-images.html"], max_chunk=None)
medium_collection = ChunkCollection(pipeline=pipeline, chunks=beagle_chunks)
medium_collection.process_chunks()
# %%
importlib.reload(birds_eye_view.core)

from birds_eye_view.core import HierachicalLabelMapper

new_h_mapper = HierachicalLabelMapper(
    max_number_levels=10,
    key_name="emoji",
)

collection.apply_step(new_h_mapper)
big_collection.apply_step(new_h_mapper)
medium_collection.apply_step(new_h_mapper)

# %%

importlib.reload(birds_eye_view.plotting)
keys = list(collection.chunks[0].attribs.keys())
birds_eye_view.plotting.visualize_chunks(collection, fields_to_include=keys)

birds_eye_view.plotting.visualize_chunks(big_collection, fields_to_include=keys)
birds_eye_view.plotting.visualize_chunks(medium_collection, fields_to_include=keys)

# %%

chunk = chunks[100]

chunk.og_text


# %%
