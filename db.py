"""
This module prepares the database for the chatbot.
It loads the documents from the directory and splits them into chunks 
Chunks are  then stored in the chroma database for easier vector search functionality
"""

import os
import shutil

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import GooglePalmEmbeddings


DATA_PATH = "data/pdfs"
CHROMA_DB_PATH = "database/chroma"
load_dotenv()
api_key = os.getenv("GOOGLE_PALM_API")


def save_to_chroma_db(chunks: list[Document]):
    """
    Saves the chunks to the chroma database.

    Args:
        chunks: A list of chunks.
    """
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    db = Chroma.from_documents(
        chunks,
        GooglePalmEmbeddings(google_api_key=api_key),
        persist_directory=CHROMA_DB_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_DB_PATH}.")


def load_documents():
    """
    Loads all documents from the data directory.

    Returns:
        list[Document]: A list of documents.
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_texts(documents: list[Document]):
    """
    Splits the documents into chunks.

    Args:
        documents (list[Document]): A list of documents.

    Returns:
        chunks: A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[1]
    print(document.page_content)
    print(document.metadata)

    return chunks


def main():
    """
    Main function to prepare the database.
    """
    documents = load_documents()
    chunks = split_texts(documents)
    save_to_chroma_db(chunks)


if __name__ == "__main__":
    main()
