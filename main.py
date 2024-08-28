# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pprint import pprint
from os import getenv
import numpy
import umap


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=800,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# This is a long document we can split up.
with open("data/plurality.txt") as f:
    document = f.read()

texts = text_splitter.create_documents([document])
print(texts[0])
print(texts[1])



# %%
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input=["Your text string goes here","second text"],
    model="text-embedding-3-small"
)
print(response.data[0].embedding)


# %%
from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df['embeddings'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('output/embedded_1k_reviews.csv', index=False)


# %%
# %% Claude response
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from pprint import pprint
from os import getenv
import numpy as np
import umap
import plotly.express as px
import plotly.io as pio

# Read the text file
with open("data/plurality.txt") as f:
    document = f.read()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([document])

# Limit the number of chunks to send to the API
max_chunks = 100  # Set the desired maximum number of chunks
texts = texts[:max_chunks]
# %%
# Initialize OpenAI client
client = OpenAI()

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Get embeddings for the chunks
embeddings = [get_embedding(text.page_content) for text in texts]
# %%
# Reduce dimensionality using UMAP
embeddings_np = np.array(embeddings)
umap_embeddings = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine').fit_transform(embeddings_np)

# Create a DataFrame with the text and UMAP embeddings
data = [{"text": text.page_content, "x": x, "y": y} for text, (x, y) in zip(texts, umap_embeddings)]
# %%
# Visualize the 2D points using Plotly
fig = px.scatter(data, x="x", y="y", hover_data=["text"])
#fig.show()

pio.show(fig)
# %%
# TODO: 
# 1. make the embedding generation async. 
# 2. Save the embeddings for a given document inside a temporary file
# 3. Incorporate into bash assistant in a new page. Upload the pdf => show the map with text when you over. Bonus: Maybe some emojis on the icon?
# 4. Clean the text extraction from the pdf. Need to remove the useless stuff, e.g. page number, words cut, references, etc. Easy (but expensive) way: use a small LLM to extract the meaningful information from the text. Add the sources in a way that makes sense, e.g. simply in parenthesis with the author + date.

# Later:
# Add images to the mix, use CLIP embeddings.
# Add LLM-factories to analyze diverse prompts. Easier text handling than with pdf.
# Add filters to create color maps on the fly.