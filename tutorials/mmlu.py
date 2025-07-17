# %%
import birds_eye_view.plotting
from birds_eye_view.core import (
    ChunkCollection,
    Pipeline,
    OpenAIEmbeddor,
    UMAPReductor,
    DotProductLabelor,
    HierachicalLabelMapper,
    EmbeddingSearch,
)
from birds_eye_view.plotting import visualize_chunks
from birds_eye_view.file_loading import load_files
import random as rd
from datasets import load_dataset
import json
from tqdm import tqdm


# %% [markdown]
# # Visualizing MMLU with bird eye view

# The goal of this notebook is to show a map of the [MMLU benckmark](https://en.wikipedia.org/wiki/MMLU). It is _the_ dataset used to measure the general knowledge of language models. However, one rarely have the chance to know what it's made of, beyond a few scattered examples and high-level description like "a dataset of multiple choice questions from high-school tests".
#
# We will use Bird eye view to create a meaningful map of MMLU, visualize tousands of data sample as a whole, and interact with the data to get a global intuitive understanding of its composition. What are the topic present? How hard are the question? How much do they rely on reasoning vs general knowledge?
#
# This notebook will guide you through data loading, and pipeline creating with bird's eye view.

# %% [markdown]
# We start by loading the MMLU dataset from Huggingface, and formating to a list of dictionnary.
# %%


def export_mmlu_to_json(output_file: str, splits=["test"]):
    """Possible splits are ['train', 'test', 'validation', 'dev', 'auxiliary_train']"""
    # Load the dataset
    ds = load_dataset("cais/mmlu", "all")

    # Prepare the data for JSON export
    json_data = []

    # Process each split
    for split in splits:
        if split not in ds:
            print(f"Split '{split}' not found in the dataset. Skipping.")
            continue

        split_data = ds[split]

        for i, item in enumerate(tqdm(split_data, desc=f"Processing {split} split")):
            chunk = {
                "text": f"Question: {item['question']}<br>"
                f"A: {item['choices'][0]}<br>"
                f"B: {item['choices'][1]}<br>"
                f"C: {item['choices'][2]}<br>"
                f"D: {item['choices'][3]}<br>"
                f"Answer: {item['choices'][item['answer']]}\n",
                "attribs": {
                    "subject": item["subject"],
                    "split": split,
                    "index": i,
                    "doc_position": i / len(split_data),
                    "correct_answer": item["choices"][item["answer"]],
                    "answer_index": item["answer"],
                    "no_line": "true"
                },
            }
            json_data.append(chunk)

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"MMLU dataset exported to {output_file}")
    print(f"Total number of questions exported: {len(json_data)}")


# %% We load and save to a json file.

export_mmlu_to_json("../data/mmlu_test.json", splits=["test"])

# %%[markdown]
# ## Importing data to bird's eye view
# Now comes the bird eye view part. We start by loading the freshly created json file into a list of chunks.
#
# The `load_files` function can take as input local path to .json, .pdf, .txt or .html files. It also supports loading webpages form url. It will parse them, and chunk then to return a list of `Chunk`.
#
# Here, no need for parsing nor chunking, the work was done in the function above.

# %% We load the file

path = "../data/mmlu_test.json"
chunks = load_files(file_paths=[path])

print(len(chunks))
# %% To reduce the computation load, we sample a random subset to half the number of chunks.
rd.seed(42)
chunks = rd.sample(chunks, len(chunks) // 2)  # sub sample the chunks

print(len(chunks))

# %%[markdown]

# ## Defining the `Pipeline`

# The `Pipeline` object defines how to get from a set of texts to a richely anotated map. Here's the role of each `PipelineStep`.

# * `OpenAIEmbeddor` uses [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings) to get an embedding for each chunk. It can use a cache relying on `hdf5` files to limit the numbers of API request.
# * `DotProductLabelor` compute dot product between the embeddings of a set of labels, and the chunk embeddings. For each chunk, it selects the labels with the highest dot product to label the chunk. Because we chose the key `"emoji"`, the backend will retreive a long list of 1500 emojis to be used as labels, but you can totally define your own list !
# * `UMAPReductor` reduced the dimension of the embdding from thousands of dimensions to a 2D space using the [UMAP algorithm](https://umap-learn.readthedocs.io/en/latest/).
# * `HierachicalLabelMapper` incorporate the position information from the `UMAPReductor` and the labels from teh `DotProductLabelor` to create a series of map labeling with varying densities depending on the zoom level.

#
# You can also define your own `PipelineStep`! Check the `core.py` file for the implementation.
# %% Define the pipeline


cache_file = "cache/"
embedding_model = "text-embedding-3-large"

pipeline = Pipeline(
    [
        OpenAIEmbeddor(
            model=embedding_model,
            cache_dir=cache_file,
            batch_size=2000,
        ),
        DotProductLabelor(
            nb_labels=3,
            embedding_model=embedding_model,
            key_name="emoji",
            cache_dir="../cache/emoji"
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
        ),
    ],
    verbose=True,
)


# %%[markdown]
# ## Let's put it together in a ChunkCollection!

# A `ChunkCollection` combine a pipeline and a list of chunk. Let's make the GPUs go brrr!

# %%

collection = ChunkCollection(pipeline=pipeline, chunks=chunks)
collection.process_chunks()

# %%[markdown]
# ## Time to visualize
# We use the parameter `n_connections=0` to remove the links between points. This is useful when the chunks come from an ordered document, like a book that has been sliced into a list of chunks. But here, there is no meaningful orders among the chunks.

# %%

visualize_chunks(collection, n_connections=0)
# %%[markdown]
# ## Run search

# We'll create an `EmbeddingSearch` to compute the dot product between a prompt and all the point of the map. This complement the emoji view by heklping use locate specific content on the map.
# 
# The `threshold` parameter is used to sharpen the results of the search and disgard background noise in the results. The raw results are always available in the `RawSearch:{prompt}` field in the plot.

# %%

prompt = "A multiple choice question that talks about cats."
search = EmbeddingSearch(prompt=prompt, embedding_model=embedding_model, threshold=0.2)
collection.apply_step(search)

# %%

visualize_chunks(collection, fist_field=f"Search:{prompt}", n_connections=0)
# %%[markdown]
# ## Load and save

# The `save` method will create a json file that can be use to load the chunks later. It doesn't save the pipeline parameters.
# %%
collection.save(filename="../saved_collections/mmlu_tutorial.json")


# %%

new_collection = ChunkCollection.load_from_file("../saved_collections/mmlu_tutorial.json")

# %%
