from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.llms.openai import OpenAI


import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CS_PAPERS_DIRECTORY_PATH = "./cs/pdf_small"
MATH_PAPERS_DIRECTORY_PATH = "./math/pdf_small"
INDEX_NAME_ON_PINECONE = "arxiv-subset-1"

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def process_pdfs(directory_path):
    reader = SimpleDirectoryReader(
        directory_path, recursive=True, required_exts=[".pdf"]
    )

    try:
        documents = reader.load_data(num_workers=10)

        # logging details of the first document
        # for i, doc in enumerate(documents[:1]):
        #     logger.info(f"Document {i}")
        #     logger.info(f"File path: {doc.metadata.get('file_path', 'N/A')}")
        #     logger.info(f"Text length:{len(doc.text)}")
        #     logger.info(f"First 100 characters: {doc.text[:100]}")
        #     logger.info(
        #         "============================================================================"
        #     )

        return documents
    except Exception as e:
        logger.error(f"An error occured while processing documents: ; {str(e)}")
        return None


if __name__ == "__main__":

    # initialize pinecone
    pc = Pinecone(
        api_key=config["PINECONE_API_KEY"],
    )
    index_name = INDEX_NAME_ON_PINECONE

    # Set up llama index
    Settings.embed_model = OpenAIEmbedding(api_key=config["OPENAI_API_KEY"])
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=config["OPENAI_API_KEY"])

    # Create pinecone index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} not found in pinecone")
        print(f"Creating index {index_name}")
        pc.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        documents = process_pdfs(CS_PAPERS_DIRECTORY_PATH)

        if documents:
            pinecone_index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )

            logger.info(
                f"Indexing documents completed. Vector embeddings stored in Pinecone"
            )

            query_engine = index.as_query_engine()
            response = query_engine.query(
                "What are the main topics covered in these research papers?"
            )
            print(response)
        else:
            logger.error("Indexing documents failed. No documents found.")

    else:
        print(f"Index {index_name} found in pinecone")
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query(
        "What are the main topics covered in these research papers?"
    )
    print(response)
