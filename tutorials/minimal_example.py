# %%
from birds_eye_view.core import ChunkCollection
from birds_eye_view.plotting import visualize_chunks


chunks = [
    {
        "text": str(n),
        "number": n,
        "modulo": n % 10,
        "quotien": n // 10,
    }
    for n in range(3000)
]
collection = ChunkCollection.load_from_list(chunks)

# %%
collection.process_chunks()
# %%
visualize_chunks(collection, n_connections=0)

