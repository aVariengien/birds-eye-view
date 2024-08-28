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
from core import (
    Chunk,
    ChunkCollection,
    OpenAIEmbeddor,
    OpenAITextProcessor,
    UMAPReductor,
    Pipeline,
    wrap_str,
)
from file_loading import import_pdf

# %%
file_name = "data/plurality.pdf"
cache_file = "cache/plurality.json"

# Create the ChunkCollection object
chunk_collection = ChunkCollection(
    chunks=import_pdf(file_name, max_chunk=10, chunk_size=800),
    pipeline=Pipeline(
        [
            # OpenAITextProcessor(prompt=DENOISING_PROMPT, update_text=True, output_key=""),
            OpenAITextProcessor(
                system_prompt=MULTIPLE_EMOJI_PROMPT,
                max_workers=10,
                update_text=False,
                output_key="emoji",
                caching_file=cache_file
            ),
            OpenAIEmbeddor(model="text-embedding-3-small", caching_file=cache_file),
            UMAPReductor(verbose=True),
        ],
        verbose=True,
    ),
)

# %%

chunk_collection.process_chunks()

def visualize_chunks(chunk_collection: ChunkCollection):
    # Extract x, y coordinates and texts from chunks
    x = [chunk.x for chunk in chunk_collection.chunks if chunk.x is not None]
    y = [chunk.y for chunk in chunk_collection.chunks if chunk.y is not None]
    texts = [
        chunk.display_text for chunk in chunk_collection.chunks if chunk.x is not None
    ]
    emojis = [
        chunk.attribs["emoji"]
        for chunk in chunk_collection.chunks
        if chunk.x is not None
    ]

    # Create a color scale based on chunk indices
    colors = list(range(len(x)))

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
                text=emojis,
                textfont=dict(size=10),
                hovertext=texts,
                hoverinfo="text",
            ),
        ]
    )

    # Update layout
    fig.update_layout(
        title="Chunk Map Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        hovermode="closest",
        height=800,
    )
    return fig


# %%

for i, chunk in enumerate(chunk_collection.chunks):
    chunk.display_text = f"<b>Chunk #{i}</b><br>" + wrap_str(
        chunk.text, skip_line_char="<br>", max_line_len=100
    )
    if "emoji" not in chunk.attribs:
        chunk.attribs["emoji"] = ""

visualize_chunks(chunk_collection)


# %%
