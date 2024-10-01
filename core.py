# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pprint import pprint
from os import getenv
import numpy
from attrs import define, field
from typing import List, Callable, Dict, Tuple, Set, Optional, Any, Literal, Union
import numpy as np
import random as rd
import umap  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
import concurrent.futures
from prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT
import threading
import json
import markdown # type: ignore
import time
import math


def wrap_str(s: str, max_line_len=100, skip_line_char="<br>"):
    """Add skip line every max_line_len characters. Ensure that no word is cut in the middle."""
    words = s.split(" ")
    wrapped_str = ""
    line_len = 0
    for word in words:
        if line_len + len(word) > max_line_len:
            wrapped_str += skip_line_char
            line_len = 0
        wrapped_str += word + " "
        line_len += len(word) + 1
    return wrapped_str


def make_display_text(text: str):
    return markdown.markdown(text)

def sharpen(value, threshold, sharpness=10):
    """
    Crush values below the threshold towards 0 and boost values above the threshold towards 1.
    
    Args:
    - values (array-like): Array of input values to transform.
    - threshold (float): The threshold value for the transformation.
    - sharpness (float): Controls the steepness of the transition (higher values = sharper transition).
    
    Returns:
    - transformed_values (numpy array): The transformed array.
    """
    value = np.clip(value, 0, 1)
    
    # Apply a modified logistic function
    logistic = 1 / (1 + np.exp(-sharpness * (value - threshold)))
    
    # Scale the logistic function to meet the requirements
    delta = (logistic - 0.5) * np.minimum(value, 1 - value) * 2
    transformed_value = value + delta
    
    # Ensure output values are between 0 and 1
    transformed_value = np.clip(transformed_value, 0, 1)
    
    return transformed_value

@define
class Chunk:
    og_text: str = field()  # the original text the atom has been created with
    x: Optional[float] = field(default=None)
    y: Optional[float] = field(default=None)
    embedding: Optional[np.array] = field(default=None, eq=False)
    text: str = field(
        default=""
    )  # the text after / during the pre-embedding processing
    display_text: str = field(default="")
    attribs: dict = field(factory=dict)  # other attributes
    id: int = field(default=-1)
    previous_chunk_index: int = field(default=-1)
    next_chunk_index: int = field(default=-1)

    def __attrs_post_init__(self):
        self.text = self.og_text
        self.display_text = make_display_text(self.text)

class PipelineStep:
    """A mother class for all process steps."""

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        raise NotImplementedError(
            "The process method must be implemented by subclasses of PipelineStep."
        )


@define
class OpenAITextProcessor(PipelineStep):
    """A TextProcessor to process text using OpenAI API with caching functionality."""

    system_prompt: str = field(default=DENOISING_PROMPT)
    model: str = field(default="gpt-4-turbo-preview")
    client: OpenAI = field(factory=lambda: OpenAI())
    max_workers: int = field(default=20)  # To adjust based on API rate limit
    output_key: str = field(
        default=""
    )  # The dict keys of attribs where the output of the processor should go
    update_text: bool = field(
        default=True
    )  # Whether to update the text attribute of the chunk with the output
    cache_file: Optional[str] = field(default=None)
    cache: Dict[str, Dict[str, str]] = field(factory=dict)
    cache_lock: threading.Lock = field(factory=threading.Lock)

    def __attrs_post_init__(self):
        if self.cache_file:
            self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            print(
                f"Cache file {self.cache_file} not found. Starting with an empty cache."
            )
        except json.JSONDecodeError:
            print(
                f"Error decoding cache file {self.cache_file}. Starting with an empty cache."
            )

    def save_cache(self):
        if self.cache_file:
            with self.cache_lock:
                with open(self.cache_file, "w") as f:
                    json.dump(self.cache, f)

    def process_chunk(self, chunk: Chunk) -> Chunk:
        cache_key = f"{self.model}-{self.system_prompt}"
        with self.cache_lock:
            if cache_key in self.cache and chunk.text in self.cache[cache_key]:
                print("Cache hit!")
                response = self.cache[cache_key][chunk.text]
                return self.update_chunk(chunk, response)

        try:
            print("Computing cache for text!")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": chunk.text},
                ],
            )
            response = str(completion.choices[0].message.content)

            # Update cache
            with self.cache_lock:
                if cache_key not in self.cache:
                    self.cache[cache_key] = {}
                self.cache[cache_key][chunk.text] = response

        except Exception as e:
            print(f"Error processing chunk: {e}")
            return chunk
        return self.update_chunk(chunk, response)

    def update_chunk(self, chunk: Chunk, response: str) -> Chunk:
        if self.update_text:  # Update text if required
            chunk.text = response
            chunk.display_text = make_display_text(chunk.text)

        # Update attribs based on output_keys
        if self.output_key != "":
            chunk.attribs[self.output_key] = response

        return chunk

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk): chunk for chunk in chunks
            }
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    _ = future.result()
                except Exception as e:
                    print(f"Chunk processing failed: {e}")

        self.save_cache()
        return chunks


@define
class OpenAIEmbeddor(PipelineStep):
    """An embeddor based on OpenAI embedding API with caching functionality."""

    cache_file: Optional[str] = field(default=None)
    cache: Dict[str, Dict[str, List[float]]] = field(factory=dict)
    model: str = field(default="text-embedding-3-small")
    client: OpenAI = field(factory=lambda: OpenAI())
    batch_size: int = field(default=4000)  # Process in batches to avoid API limits
    cache_lock: threading.Lock = field(factory=threading.Lock)

    def __attrs_post_init__(self):
        if self.cache_file:
            self.load_cache()

    def compute_embeddings(self, texts: List[str]) -> List[Any]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            t1 = time.time()
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
            except:
                print(len(batch))
                print( " ==== ")
                print(batch)
                print(" ==== ")
                print(response.json())
                raise ValueError("bad request!")
            print(f"Time for request: {time.time() - t1}")
            batch_embeddings = [np.array(data.embedding) for data in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def load_cache(self):
        t1 = time.time()
        try:
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            print(
                f"Cache file {self.cache_file} not found. Starting with an empty cache."
            )
        except json.JSONDecodeError:
            print(
                f"Error decoding cache file {self.cache_file}. Starting with an empty cache."
            )
        print(f"Time to load: {time.time() - t1}, {self.cache_file}")

    def save_cache(self):
        if self.cache_file:
            with self.cache_lock:
                with open(self.cache_file, "w") as f:
                    json.dump(self.cache, f)

    def get_embeddings(self, texts: List[str]) -> List[np.array]:
        all_embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if self.model in self.cache and text in self.cache[self.model]:
                all_embeddings.append(np.array(self.cache[self.model][text]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            print(f"Caching {len(texts_to_embed)} ...")
            new_embeddings = self.compute_embeddings(texts_to_embed)
            for text, embedding in zip(texts_to_embed, new_embeddings):
                if self.model not in self.cache:
                    self.cache[self.model] = {}
                self.cache[self.model][text] = embedding.tolist()  # type: ignore

            for i, embedding in zip(indices_to_embed, new_embeddings):
                all_embeddings.insert(i, embedding)

            self.save_cache()
        else:
            print("No caching needed!")

        return all_embeddings

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

def dist(chunk1: Chunk, chunk2: Chunk) -> float:
    assert chunk1.x is not None
    assert chunk1.y is not None
    assert chunk2.x is not None
    assert chunk2.y is not None
    return math.sqrt((chunk1.x - chunk2.x)**2 +  (chunk1.y - chunk2.y)**2)

def most_frequent(l: List[str]) -> str:
    counts: dict[str, int] = {} 
    most_freq = ""
    highest_count = 0
    for x in l:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
        if counts[x] > highest_count:
            highest_count = counts[x]
            most_freq = x
    return most_freq

@define
class HierachicalLabelMapper(PipelineStep):
    """
        A pipeline step that take the result of a DotProductLabelor, and makes progressive agregation of the labels through local majority voting.
    """
    number_levels: int = field()
    key_name: str = field()

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        for chunk in chunks:
            assert chunk.x is not None, "You need to reduce the dimension before the Mapper"
            assert chunk.y is not None
            assert self.key_name in chunk.attribs, "You need to use a DotProductLabelor before using the Mapper."

        C = 5 # a constant controlling the evolution of the radius for each step. 
        exponent = 2.5

        # estimating the average distance between two points
        average_distance = 0.0
        N = 300
        for i in range(N):
            chunk1, chunk2 = rd.sample(chunks, 2)
            average_distance += dist(chunk1, chunk2)
        average_distance = average_distance / N

        hierarchical_labels: dict[int, List[str]] = {}
        for i in range(len(chunks)):
            hierarchical_labels[i] = []

        step_dividors = [ math.pow(((i/self.number_levels)*C),exponent) + 1 for i in range(self.number_levels)]
        print(step_dividors)

        for step in range(self.number_levels):
            #nb_points = int(len(chunks) * step / self.number_levels + 1)
            # TODO: change the way the radius is computed
            radius = average_distance / step_dividors[step] if step < self.number_levels -1 else 0
            # for the last step, radius is zero

            indices = set(list(range(len(chunks))))
            
            # while there still indices
            while len(indices) > 0:
                # choose a local representative among the remainers
                rd_point_idx = rd.choice(list(indices)) 
                center = chunks[rd_point_idx]
                labels_neighbors = []
                indices.remove(rd_point_idx)
                for i, chunk in enumerate(chunks):
                    if dist(chunk, center) < radius:
                        labels_neighbors.extend(chunk.attribs[self.key_name+ "_list"].split(","))
                        if i in indices:
                            indices.remove(i)
                            # remove the points in the suronding 

                # compute the most frequent label in the neighborhood, among _all_ the point in the radius
                
                if len(labels_neighbors) == 0:
                    hierarchical_labels[rd_point_idx].append(center.attribs[self.key_name+ "_list"].split(",")[0])
                else:
                    hierarchical_labels[rd_point_idx].append(most_frequent(labels_neighbors))

            for k in hierarchical_labels.keys():
                if len(hierarchical_labels[k]) < step:
                    hierarchical_labels[k].append(" ")
        
        for i, chunk in enumerate(chunks):
            chunk.attribs[self.key_name] = chunk.attribs[self.key_name + "_list"].replace(",", "")
            chunk.attribs[self.key_name + "_list"] = ",".join(hierarchical_labels[i])
            
            #print(",".join(hierarchical_labels[i]))

        return chunks


@define
class DotProductLabelor(PipelineStep):
    possible_labels: List[str] = field()
    nb_labels: int = field()
    cache_file: str = field()
    embedding_model: str = field()
    key_name: str = field()
    prefix: str = field(default="")

    def __attrs_post_init__(self):
        assert self.key_name in ["emoji", "keyword"]
        self.embedder = OpenAIEmbeddor(cache_file=self.cache_file, model=self.embedding_model)

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        # 1. Embed the label strings
        label_texts = [f"{self.prefix}{label}" for label in self.possible_labels]
        label_embeddings = self.embedder.get_embeddings(label_texts)

        # 2. Embed all chunk texts
        chunk_embeddings = self.embedder.get_embeddings([chunk.text for chunk in chunks])

        # 3. Compute dot products between label embeddings and chunk embeddings
        dot_products = np.dot(np.array(chunk_embeddings), np.array(label_embeddings).T)

        dot_products = dot_products - np.mean(dot_products, axis=0) # center the dot products

        # 4. For each chunk, find top 3 labels and add to attribs
        for i, chunk in enumerate(chunks):
            top_indices = np.argsort(dot_products[i])[-self.nb_labels:][::-1]
            top_labels = [self.possible_labels[idx] for idx in top_indices]
            chunk.attribs[self.key_name+ "_list"] = ",".join(top_labels)
            chunk.attribs[self.key_name] = top_labels[0]

        return chunks

@define
class EmbeddingSearch(PipelineStep):
    prompt: str = field()
    cache_file: str = field()
    embedding_model: str = field()
    threshold: Optional[float] = field(default=None)
    embedder: Optional[OpenAIEmbeddor] = field(default=None)

    def __attrs_post_init__(self):
        self.embedder = OpenAIEmbeddor(cache_file=None, model=self.embedding_model)

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        # Compute prompt embedding
        assert self.embedder is not None
        t1 = time.time()
        if f"RawSearch:{self.prompt}" not in chunks[0].attribs:
            prompt_embedding = self.embedder.get_embeddings([self.prompt])[0]
            chunk_embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
            
            if not chunk_embeddings:
                raise ValueError("No embeddings found in chunks. Make sure to run the embedding step first.")
            
            # Compute dot products
            dot_products = np.dot(np.array(chunk_embeddings), prompt_embedding)
            
            # Add scores to chunk attributes
            for chunk, score in zip(chunks, dot_products):
                chunk.attribs[f"Search:{self.prompt}"] = sharpen(float(score), threshold=self.threshold)
                chunk.attribs[f"RawSearch:{self.prompt}"] = float(score)
        else:
            for chunk in chunks:
                chunk.attribs[f"Search:{self.prompt}"] = sharpen(float(chunk.attribs[f"RawSearch:{self.prompt}"]), threshold=self.threshold)
        print(f"Time to compute embedding {time.time() - t1}")
        return chunks

@define
class UMAPReductor(PipelineStep):
    """A reductor based on UMAP to reduce the dimension of the embeddings. (high dim vect -> 2D vect)"""

    n_neighbors: int = field(default=40)
    n_jobs: int = field(default=1)
    min_dist: float = field(default=0.1)
    n_components: int = field(default=2)
    random_state: int = field(default=42)
    verbose: bool = field(default=False)

    def __attrs_post_init__(self):
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            random_state=self.random_state,
            verbose=self.verbose,
            n_jobs=self.n_jobs
        )

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        # Extract embeddings from chunks
        embeddings = [
            chunk.embedding for chunk in chunks if chunk.embedding is not None
        ]

        if not embeddings:
            raise ValueError(
                "No embeddings found in chunks. Make sure to run the embedding step first."
            )

        # Stack embeddings into a 2D numpy array
        embedding_matrix = np.stack(embeddings)  # type: ignore

        # Perform dimensionality reduction
        reduced_embeddings = self.reducer.fit_transform(embedding_matrix)

        # Update chunks with reduced embeddings
        for chunk, reduced_embedding in zip(chunks, reduced_embeddings):
            chunk.x, chunk.y = reduced_embedding

        return chunks


@define
class Pipeline:
    steps: List[PipelineStep] = field(factory=list)
    verbose: bool = field(default=False)

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        for i, step in enumerate(self.steps):
            
            if self.verbose:
                print(f"Running step #{i} - {step.__class__}")
            t1 = time.time()
            chunks = step.process(chunks)
            t2 = time.time()
            if self.verbose:
                print(f"Step #{i} - {step.__class__} completed in {t2-t1}s.")

        return chunks


@define
class ChunkCollection:
    chunks: List[Chunk] = field(
        factory=list
    )  # Is exported as [{og_text: "example", attribs: {"page":2, "title": "bla"}}, ...]
    pipeline: Pipeline = field(factory=Pipeline)

    def __attrs_post_init__(self):
        for i, chunk in enumerate(self.chunks):
            if "no_line" in chunk.attribs:
                chunk.next_chunk_index = i+1
                chunk.previous_chunk_index = i
            else:
                if i < len(self.chunks)-1:
                    chunk.next_chunk_index = i+1
                else:
                    chunk.next_chunk_index = i
                if i>0:
                    chunk.previous_chunk_index = i-1
                else:
                    chunk.previous_chunk_index = i
            chunk.id = i

    def __getstate__(self):
        return {'chunks': self.chunks}

    def __setstate__(self, state):
        self.chunks = state['chunks']

    def process_chunks(self) -> None:
        self.chunks = self.pipeline.process(self.chunks)
        self.__attrs_post_init__()
    def apply_step(self, step: PipelineStep):
        self.chunks = step.process(self.chunks)
        self.__attrs_post_init__()

# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# def crush_and_boost(values, threshold, sharpness=10):
#     """
#     Crush values below the threshold towards 0 and boost values above the threshold towards 1.
    
#     Args:
#     - values (array-like): Array of input values to transform.
#     - threshold (float): The threshold value for the transformation.
#     - sharpness (float): Controls the steepness of the transition (higher values = sharper transition).
    
#     Returns:
#     - transformed_values (numpy array): The transformed array.
#     """
#     # Apply a logistic function with a custom threshold
#     values = np.clip(values, 0, 1)
    
#     # Apply a modified logistic function
#     logistic = 1 / (1 + np.exp(-sharpness * (values - threshold)))
    
#     # Scale the logistic function to meet the requirements
#     delta = (logistic - 0.5) * np.minimum(values, 1 - values) * 2
#     transformed_values = values + delta
    
#     # Ensure output values are between 0 and 1
#     transformed_values = np.clip(transformed_values, 0, 1)
    
#     return transformed_values

# # Define the range of input values
# x = np.linspace(0, 1, 500)

# # Parameters for the transformation
# threshold = 0.15
# sharpness = 10

# # Apply the transformation
# y = crush_and_boost(x, threshold, sharpness)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label=f'Sigmoid-like function (threshold={threshold}, sharpness={sharpness})')
# plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
# plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
# plt.axhline(1, color='black', linestyle='--', linewidth=0.5)
# plt.title('Crush and Boost Function')
# plt.xlabel('Input Value')
# plt.ylabel('Transformed Value')
# plt.legend()
# plt.grid(True)
# plt.show()

# # %%
