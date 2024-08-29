from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PropertyGraphIndex

def create_vector_index(documents, embed_model):
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

def create_property_graph_index(documents, llm, embed_model):
    return PropertyGraphIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model,
        show_progress=True
    )
