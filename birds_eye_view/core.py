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
from birds_eye_view.prompts import DENOISING_PROMPT, MULTIPLE_EMOJI_PROMPT, ALL_EMOJIS
import threading
import json
import markdown # type: ignore
import time
import math
from sklearn.neighbors import RadiusNeighborsTransformer #type: ignore
import h5py
from h5py import File
import os
import base64

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
    embedding: Optional[np.ndarray] = field(default=None, eq=False)
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
    cache_dir: Optional[str] = field(default=None)
    model: str = field(default="text-embedding-3-small")
    api_key: Optional[str] = field(default=None)
    client: OpenAI = field(factory=lambda: OpenAI(api_key="dummy"))
    batch_size: int = field(default=4000)

    # New attributes for HDF5 and index
    hdf5_file: Optional[File] = field(default=None)
    index: Dict[str, Dict[str, int]] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.api_key is not None and len(self.api_key) > 0:
            self.client.api_key = self.api_key
        else:
            self.client.api_key = getenv("OPENAI_API_KEY")
    def load_cache(self):
        t1 = time.time()
        if self.cache_dir:
            hdf5_path = os.path.join(self.cache_dir, f"{self.model}_embeddings.h5")
            index_path = os.path.join(self.cache_dir, f"{self.model}_index.json")

            # Load or create HDF5 file
            self.hdf5_file = h5py.File(hdf5_path, 'a')

            # Load or create index
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}

        print(f"Time to load: {time.time() - t1}")

    def save_index(self):
        if self.cache_dir:
            index_path = os.path.join(self.cache_dir, f"{self.model}_index.json")
            with open(index_path, 'w') as f:
                json.dump(self.index, f)

    def compute_embeddings(self, texts: List[str]) -> List[Any]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            t1 = time.time()
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
            except Exception as e:
                print(f"Error in API request: {e}")
                raise ValueError(f"Error in embedding API request: {e}")
            print(f"Time for request: {time.time() - t1}")
            batch_embeddings = [np.array(data.embedding) for data in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        self.load_cache()
        all_embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            if self.cache_dir and text in self.index.get(self.model, {}):
                assert type(self.hdf5_file) == File
                embedding_index = self.index[self.model][text]
                embedding = self.hdf5_file[f"{self.model}/{embedding_index}"][:]
                all_embeddings.append(embedding)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        if texts_to_embed:
            print(f"Computing {len(texts_to_embed)} embeddings...")
            new_embeddings = self.compute_embeddings(texts_to_embed)
            for i, embedding in zip(indices_to_embed, new_embeddings):
                all_embeddings.insert(i, embedding)

            if self.cache_dir:
                assert type(self.hdf5_file) == File
                if self.model not in self.hdf5_file:
                    self.hdf5_file.create_group(self.model)

                if self.model not in self.index:
                    self.index[self.model] = {}

                current_index = len(self.index[self.model])

                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self.hdf5_file[f"{self.model}/{current_index}"] = embedding
                    self.index[self.model][text] = current_index
                    current_index += 1

                self.hdf5_file.flush()
                self.save_index()
        else:
            print("No caching needed!")

        return all_embeddings

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    def __del__(self):
        if self.hdf5_file:
            self.hdf5_file.close()

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
    max_number_levels: int = field()
    key_name: str = field()

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        assert self.max_number_levels > 3
        for chunk in chunks:
            assert chunk.x is not None, "You need to reduce the dimension before the Mapper"
            assert chunk.y is not None
            assert self.key_name in chunk.attribs, "You need to use a DotProductLabelor before using the Mapper."

        C = 9 # a constant controlling the evolution of the radius for each step. 
        exponent = 3

        # estimating the average distance between two points
        distances = []
        N = 1000
        for i in range(N):
            chunk1, chunk2 = rd.sample(chunks, 2)
            distances.append(dist(chunk1, chunk2))
        map_diameter = np.percentile(distances, 90)

        hierarchical_labels: dict[int, List[str]] = {}
        for i in range(len(chunks)):
            hierarchical_labels[i] = []

        #step_dividors = [ math.pow(((i/self.max_number_levels)*C),exponent) + 6 for i in range(self.max_number_levels)]
        max_zoom = 0.6
        label_per_length = 5 # density of the label shown on the screen

        min_distance = 0.0 # a tail to ensure the small details are rendered
        tail_length = 1
        radiuses = [(i*max_zoom/self.max_number_levels) *  map_diameter / (2*label_per_length) for i in range(1, self.max_number_levels-tail_length)][::-1]
        last_radius = radiuses[-1]
        for j in range(tail_length, 0, -1):
            radiuses.append((last_radius-min_distance)*(j/(tail_length+1)) + min_distance)
        radiuses.append(0.0)
    

        points = np.array([[chunk.x, chunk.y] for chunk in chunks])

        last_turn = False
        step = 0
        while step < self.max_number_levels and not last_turn:
            radius = radiuses[step] if step < self.max_number_levels -1 else 0 # for the last step, radius is zero
            if last_turn:
                radius = 0

            graph = RadiusNeighborsTransformer(radius=radius).fit_transform(points)

            indices = set(list(range(len(chunks))))
            
            # while there still indices
            nb_turns = 0
            while len(indices) > 0:
                nb_turns += 1
                # choose a local representative among the remainers
                rd_point_idx = rd.choice(list(indices)) 
                center = chunks[rd_point_idx]
                labels_neighbors = []
                indices.remove(rd_point_idx)

                _, non_zero_idx = graph[rd_point_idx].nonzero()

                for i in non_zero_idx:
                    labels_neighbors.extend(chunks[i].attribs[self.key_name+ "_label_list"].split(","))
                    if i in indices:
                        indices.remove(i)
                        # remove the points in the neighborhood 

                # compute the most frequent label in the neighborhood, among _all_ the point in the radius
                
                if len(labels_neighbors) == 0:
                    hierarchical_labels[rd_point_idx].append(center.attribs[self.key_name+ "_label_list"].split(",")[0])
                else:
                    hierarchical_labels[rd_point_idx].append(most_frequent(labels_neighbors))

            for k in hierarchical_labels.keys():
                if len(hierarchical_labels[k]) <= step:
                    hierarchical_labels[k].append(" ")

            # if we do almost as much turns as the number of chunks, the radius is too small. We should stop, we are at the minimal resolution
            # if nb_turns > 0.8*len(chunks): # TODO uncomment
            #     last_turn = True
            step +=1
        
        for i, chunk in enumerate(chunks):
            chunk.attribs[self.key_name] = chunk.attribs[self.key_name + "_label_list"].replace(",", "")
            display_list = []
            for label in hierarchical_labels[i][:(-tail_length-1)]:
                for k in range(tail_length+1):
                    display_list.append(label)
            for label in hierarchical_labels[i][-tail_length-1:]:
                display_list.append(label)
            chunk.attribs[self.key_name + "_list"] = ",".join(display_list)
            
            #print(",".join(hierarchical_labels[i]))

        return chunks


@define
class DotProductLabelor(PipelineStep):
    nb_labels: int = field()
    embedding_model: str = field()
    key_name: str = field()
    api_key: Optional[str] = field(default=None)
    possible_labels: Optional[List[str]] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    no_cache: bool = field(default=False)
    prefix: str = field(default="")

    def __attrs_post_init__(self):
        assert self.key_name in ["emoji", "keyword"]
        if self.cache_dir is None and not self.no_cache:
            self.cache_dir = "cache/"+self.key_name #TODO change when we change the cache file system
            # by default the name of the cache is the keyname.
        
        if self.possible_labels is None and self.key_name == "emoji":
            self.possible_labels = ALL_EMOJIS
        
        self.embedder = OpenAIEmbeddor(cache_dir=self.cache_dir, model=self.embedding_model, api_key=self.api_key)

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        assert self.possible_labels is not None, "No possible_labels set."
        assert len(chunks) > 0
        assert chunks[0].embedding is not None, "Chunk embeddings needs to be computed ahead."
        # 1. Embed the label strings
        label_texts = [f"{self.prefix}{label}" for label in self.possible_labels]
        label_embeddings = self.embedder.get_embeddings(label_texts)

        # 2. Embed all chunk texts
        chunk_embeddings = [chunk.embedding for chunk in chunks]

        # 3. Compute dot products between label embeddings and chunk embeddings
        dot_products = np.dot(np.array(chunk_embeddings), np.array(label_embeddings).T)

        dot_products = dot_products - np.mean(dot_products, axis=0) # center the dot products

        # 4. For each chunk, find top 3 labels and add to attribs
        for i, chunk in enumerate(chunks):
            top_indices = np.argsort(dot_products[i])[-self.nb_labels:][::-1]
            top_labels = [self.possible_labels[idx] for idx in top_indices]
            chunk.attribs[self.key_name+ "_label_list"] = ",".join(top_labels)
            chunk.attribs[self.key_name] = top_labels[0]

        return chunks

@define
class EmbeddingSearch(PipelineStep):
    prompt: str = field()
    embedding_model: str = field()
    threshold: Optional[float] = field(default=None)
    embedder: Optional[OpenAIEmbeddor] = field(default=None)
    api_key: Optional[str] = field(default=None)

    def __attrs_post_init__(self):
        self.embedder = OpenAIEmbeddor(cache_dir=None, model=self.embedding_model, api_key=self.api_key)

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        # Compute prompt embedding
        assert self.embedder is not None
        t1 = time.time()
        if f"RawSearch:{self.prompt}" not in chunks[0].attribs:
            print(self.embedder.get_embeddings([self.prompt]))
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

def default_pipeline_factory() -> Pipeline:
    return Pipeline([
        OpenAIEmbeddor(
            model="text-embedding-3-large", 
            cache_dir=None,
            batch_size=2000,
            ),
        DotProductLabelor(
            nb_labels=3,
            embedding_model="text-embedding-3-large",
            key_name="emoji",
            no_cache=False,
        ),
        UMAPReductor(
            verbose=True,
            n_neighbors=20,
            min_dist=0.05,
        ),
        HierachicalLabelMapper(
            max_number_levels=10,
            key_name="emoji",
        )
    ], verbose=True)


@define
class ChunkCollection:
    chunks: List[Chunk] = field( # TODO fix the type here.
        factory=list
    )  # Is exported as [{og_text: "example", attribs: {"page":2, "title": "bla"}}, ...]
    pipeline: Pipeline = field(factory=default_pipeline_factory)

    def __attrs_post_init__(self):
        assert len(self.chunks) > 0, "You have no chunk in your collection!"
        assert 'Chunk' in str(type(self.chunks[0])), f"Bad chunk type! {type(self.chunks[0])}"

        for i, chunk in enumerate(self.chunks):
            if "no_line" in chunk.attribs: # TODO: fix it, so far, still lines. See changes made to the plotting function
                chunk.next_chunk_index = i
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

    def save(self, filename: Optional[str] = None, return_data: bool = False):
        """
        Save the ChunkCollection to a file.
        """
        def chunk_to_dict(chunk: Chunk) -> dict:
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            chunk_dict = {
                'og_text': chunk.og_text,
                'x': convert_to_serializable(chunk.x),
                'y': convert_to_serializable(chunk.y),
                'text': chunk.text,
                'display_text': chunk.display_text,
                'attribs': {k: convert_to_serializable(v) for k, v in chunk.attribs.items()},
                'id': chunk.id,
                'previous_chunk_index': chunk.previous_chunk_index,
                'next_chunk_index': chunk.next_chunk_index,
            }
            if chunk.embedding is not None:
                # Convert numpy array to bytes, then to base64
                embedding_bytes = chunk.embedding.astype(np.float16).tobytes()
                embedding_base64 = base64.b64encode(embedding_bytes).decode('utf-8')
                chunk_dict['embedding'] = embedding_base64
            return chunk_dict

        serialized_chunks = [chunk_to_dict(chunk) for chunk in self.chunks]

        if return_data:
            return json.dumps(serialized_chunks)
        else:
            assert type(filename) == str
            with open(filename, 'w') as f:
                json.dump(serialized_chunks, f)
        

    @classmethod
    def load_from_list(cls, l: List[Any], pipeline: Optional[Pipeline] = None) -> 'ChunkCollection':
        assert len(l) > 0

        if type(l[0]) == str:
            dicts = [{"text" : s} for s in l]
        else:
            dicts = l
        
        new_chunks = []
        for i, chunk in enumerate(dicts):
            assert "text" in chunk, "Your chunk must have a text"
            new_chunk = Chunk(og_text=chunk["text"])
            for field in ["previous_chunk_index", "next_chunk_index", "display_text"]:
                if field in chunk:
                    new_chunk.__setattr__(field, chunk[field])
            for k in chunk.keys():
                if k not in ["previous_chunk_index", "next_chunk_index", "display_text", "text"]:
                    new_chunk.attribs[k] = chunk[k]
            new_chunks.append(new_chunk)
        
        if pipeline is None:
            pipeline = default_pipeline_factory()
        return cls(chunks=new_chunks, pipeline=pipeline)

    @classmethod
    def load_from_file(cls, source: str | Any) -> 'ChunkCollection':
        """
        Load a ChunkCollection from a file.
        """
        if type(source) == str:
            with open(source, 'r') as f:
                serialized_chunks = json.load(f)
        else:
            serialized_chunks = json.load(source) # type: ignore

        def dict_to_chunk(chunk_dict: dict) -> Chunk:
            embedding = None
            if 'embedding' in chunk_dict:
                # Convert base64 to bytes, then to numpy array
                embedding_bytes = base64.b64decode(chunk_dict['embedding'])
                embedding = np.frombuffer(embedding_bytes, dtype=np.float16)

            return Chunk(
                og_text=chunk_dict['og_text'],
                x=chunk_dict.get('x'),
                y=chunk_dict.get('y'),
                embedding=embedding,
                text=chunk_dict['text'],
                display_text=chunk_dict['display_text'],
                attribs=chunk_dict['attribs'],
                id=chunk_dict['id'],
                previous_chunk_index=chunk_dict['previous_chunk_index'],
                next_chunk_index=chunk_dict['next_chunk_index']
            )

        chunks = [dict_to_chunk(chunk_dict) for chunk_dict in serialized_chunks]
        return cls(chunks=chunks)