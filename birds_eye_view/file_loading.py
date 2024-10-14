# %%
from langchain_community.document_loaders import MathpixPDFLoader  # type: ignore
from birds_eye_view.core import wrap_str
from langchain_text_splitters import RecursiveCharacterTextSplitter
from birds_eye_view.core import (
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
from markdownify import markdownify as md  # type: ignore
from typing import List
import re
from bs4 import BeautifulSoup  # type: ignore
import html2text
import markdown  # type: ignore
import json

## from txt


def load_txt(
    file_name: str,
    max_chunk: Optional[int] = 100,
    chunk_size: int = 800,
    separator: Optional[str] = None,
) -> List[Chunk]:
    """
    Load a text file and split it into chunks.

    Args:
    file_name (str): Path to the text file.
    max_chunk (Optional[int]): Maximum number of chunks to return. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.
    separator (str or None): if not None, the file is split along in chunks separated by the string separator.

    Returns:
    List[Chunk]: List of Chunk objects.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(text)

    if max_chunk is not None:
        texts = texts[:max_chunk]

    chunks = [Chunk(og_text=text) for text in texts]
    return chunks


def import_json(file_name: str, max_chunk: Optional[int] = None) -> List[Chunk]:
    """
    Load chunks from a JSON file.

    Args:
    file_name (str): Path to the JSON file.
    max_chunk (Optional[int]): Maximum number of chunks to load. If None, load all chunks.

    Returns:
    List[Chunk]: A list of Chunk objects.
    """
    try:
        with open(file_name, "r") as file:
            data = json.load(file)

        chunks = []
        for i, item in enumerate(data):
            if max_chunk is not None and i >= max_chunk:
                break

            chunk = Chunk(og_text=item["text"], attribs=item["attribs"])
            chunks.append(chunk)

        return chunks

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: '{file_name}' is not a valid JSON file.")
        return []
    except KeyError as e:
        print(f"Error: Missing key in JSON structure: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


## from pdf


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
        chunk_overlap=0,
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
            og_text=text,
            attribs={
                "page": page_nb[i],
                "index": i,
                "doc_position": i / len(texts),
                "title": file_name,
            },
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


## from url


def get_heading_list(content: str, idx: int, markdown: bool = False) -> List[str]:
    """Given a document content (markdown or html), return the list of heading that covers the text at the string position idx"""
    if markdown:
        return get_markdown_headings(content, idx)
    else:
        return get_html_headings(content, idx)


def get_markdown_headings(content: str, idx: int) -> List[str]:
    headings = []  # type: List[str]
    current_level = 0
    lines = content.split("\n")
    current_position = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for the newline character
        if current_position + line_length > idx:
            break

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            while current_level >= level:
                if headings:
                    headings.pop()
                current_level -= 1

            headings.append(heading_text)
            current_level = level

        current_position += line_length

    return headings


def get_html_headings(content: str, idx: int) -> List[str]:
    headings = []  # type: List[str]
    current_position = 0
    heading_pattern = re.compile(r"<h([1-6]).*?>(.*?)</h\1>", re.DOTALL)

    for match in heading_pattern.finditer(content):
        start, end = match.span()
        if start > idx:
            break

        level = int(match.group(1))
        heading_text = re.sub(r"<.*?>", "", match.group(2)).strip()

        while len(headings) >= level:
            headings.pop()

        headings.append(heading_text)
        current_position = end
    return headings


def remove_script_tags(html_content):
    """Return the document without script tag."""
    soup = BeautifulSoup(html_content, "html.parser")
    script_tags = soup.find_all("script")
    for tag in script_tags:
        tag.decompose()
    return str(soup)


def download_html(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    return response.content


def read_file(file_name: str):
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def load_html(
    html_content: str,
    url: str,
    max_chunk: Optional[int] = 100,
    chunk_size: int = 800,
    markdownify: bool = False,
) -> List[Chunk]:
    """
    Load content from a URL and split it into chunks.

    Args:
    html_content: the html content in a string.
    max_chunk (Optional[int]): Maximum number of chunks to return. If None, return all chunks.
    chunk_size (int): Size of each chunk in characters.

    Returns:
    List[Chunk]: List of Chunk objects.
    """

    doc = Document(html_content)

    title = doc.title()
    content = markdown.markdown(html2text.html2text(remove_script_tags(html_content)))
    content = content.replace("\n", " ") #remove some unecessary line breaks
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
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(content)

    if max_chunk is not None:
        texts = texts[:max_chunk]

    chunks = []
    for i, text in enumerate(texts):
        idx = content.find(text)
        chunks.append(
            Chunk(
                og_text=text,
                attribs={
                    "title": title,
                    "url": url,
                    "index": i,
                    "doc_position": i / len(texts),
                    "headings": get_heading_list(content, idx, markdown=markdownify)[
                        ::
                    ],
                },
            )
        )

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
            return load_html(download_html(file_name), file_name, max_chunk, chunk_size)
    else:
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() == ".pdf":
            return import_pdf(file_name, max_chunk, chunk_size)
        elif file_extension.lower() == ".json":
            return import_json(file_name, max_chunk)
        elif file_extension.lower() == ".txt":
            return load_txt(file_name, max_chunk, chunk_size)
        elif file_extension.lower() == ".html":
            return load_html(read_file(file_name), file_name, max_chunk, chunk_size)
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
