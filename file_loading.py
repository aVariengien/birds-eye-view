# %%
from langchain_community.document_loaders import MathpixPDFLoader  # type: ignore
from core import wrap_str
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core import (
    Chunk,
    ChunkCollection,
    OpenAIEmbeddor,
    OpenAITextProcessor,
    UMAPReductor,
    Pipeline,
    wrap_str,
)

import concurrent.futures

from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional

import os
from typing import List

import os
from typing import List, Optional
import requests  # type: ignore
from readability import Document  # type: ignore
from urllib.parse import urlparse
from markdownify import markdownify as md


def load_txt(
    file_name: str, max_chunk: Optional[int] = 100, chunk_size: int = 800
) -> List[Chunk]:
    """
    Load a text file and split it into chunks.

    Args:
    file_name (str): Path to the text file.
    max_chunk (Optional[int]): Maximum number of chunks to return. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.

    Returns:
    List[Chunk]: List of Chunk objects.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(text)

    if max_chunk is not None:
        texts = texts[:max_chunk]

    chunks = [Chunk(og_text=text) for text in texts]
    return chunks


def find_page_number(idx: int, indices: List[int]) -> int:
    """idx is an int. indices is the list of the indices where the page starts.
    Return the page number of idx, i.e. n such that indices[n] <= idx < indices[n+1]"""
    assert (
        idx < indices[-1] and idx >= indices[0]
    ), f"Search for index {idx} is out of range!"
    left, right = 0, len(indices) - 1

    while left <= right:
        mid = (left + right) // 2

        if indices[mid] <= idx < indices[mid + 1]:
            return mid
        elif idx < indices[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return (
        len(indices) - 1 if idx >= indices[-1] else -1
    )  # Return -1 if idx is not within any page range


def import_pdf(
    file_name: str, max_chunk: Optional[int] = 100, chunk_size: int = 800
) -> List[Chunk]:
    """If chunk_size is None, then all the chuncks are included."""
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()

    document = ""
    indices = [0]
    idx = 0
    for page in pages:
        document += page.page_content
        idx += len(page.page_content)
        indices.append(idx)
    idx += 1
    indices.append(idx)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(document)

    page_nb = []
    for text in texts:
        idx = document.find(text)
        page_nb.append(find_page_number(idx, indices))

    chunks = [
        Chunk(
            og_text=text, attribs={"page": page_nb[i], "index": i, "title": file_name}
        )
        for i, text in enumerate(texts)
    ]
    return chunks


def download_pdf(url: str, cache_folder: str = "cache/pdf") -> str:
    """
    Download a PDF from a given URL and save it to the cache folder.

    Args:
    url (str): URL of the PDF to download.
    cache_folder (str): Folder to save the downloaded PDF.

    Returns:
    str: Path to the downloaded PDF file.
    """
    # Create cache folder if it doesn't exist
    os.makedirs(cache_folder, exist_ok=True)

    # Extract filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    # Full path for the downloaded file
    file_path = os.path.join(cache_folder, filename)

    # Check if file already exists in cache
    if os.path.exists(file_path):
        print(f"PDF already in cache: {file_path}")
        return file_path

    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Save the file
    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"PDF downloaded and saved to: {file_path}")
    return file_path


def load_url(
    url: str, max_chunk: Optional[int] = 100, chunk_size: int = 800, markdownify: bool = False
) -> List[Chunk]:
    """
    Load content from a URL and split it into chunks.

    Args:
    url (str): URL to load content from.
    max_chunk (Optional[int]): Maximum number of chunks to return. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.

    Returns:
    List[Chunk]: List of Chunk objects.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    doc = Document(response.content)

    title = doc.title()
    content = doc.summary()
    if markdownify:
        content = md(
            content,
            convert=[
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "a",
                "strong",
                "b",
                "em",
                "i",
                "ul",
                "ol",
                "li",
                "blockquote",
                "code",
                "pre",
                "img",
                "hr",
                "table",
                "tr",
                "th",
                "td",
                "br",
            ],
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(content)

    if max_chunk is not None:
        texts = texts[:max_chunk]

    chunks = [
        Chunk(og_text=text, attribs={"title": title, "url": url, "index": i})
        for i, text in enumerate(texts)
    ]
    return chunks


def load_file(
    file_name: str, max_chunk: Optional[int] = 100, chunk_size: int = 800
) -> List[Chunk]:
    """
    Load a file or URL and split it into chunks. Supports PDF, TXT files, and URLs.

    Args:
    file_name (str): Path to the file or URL.
    max_chunk (Optional[int]): Maximum number of chunks to return. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.

    Returns:
    List[Chunk]: List of Chunk objects.
    """
    if file_name.startswith("http://") or file_name.startswith("https://"):
        parsed_url = urlparse(file_name)
        if parsed_url.path.endswith(".pdf") or "arxiv.org/pdf/" in file_name:
            # Download PDF and use import_pdf
            pdf_path = download_pdf(file_name)
            return import_pdf(pdf_path, max_chunk, chunk_size)
        else:
            # Handle as regular URL
            return load_url(file_name, max_chunk, chunk_size)
    else:
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() == ".pdf":
            return import_pdf(file_name, max_chunk, chunk_size)
        elif file_extension.lower() == ".txt":
            return load_txt(file_name, max_chunk, chunk_size)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


def load_files(
    file_paths: List[str], max_chunk: Optional[int] = 100, chunk_size: int = 800
) -> List[Chunk]:
    """
    Load multiple files or URLs in parallel and return a list of chunks.

    Args:
    file_paths (List[str]): List of file paths or URLs to process.
    max_chunk (Optional[int]): Maximum number of chunks to return per file. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.

    Returns:
    List[Chunk]: Combined list of Chunk objects from all processed files.
    """
    all_chunks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(load_file, file_path, max_chunk, chunk_size): file_path
            for file_path in file_paths
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return all_chunks
