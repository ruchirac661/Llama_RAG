from dotenv import load_dotenv
load_dotenv()
from models.index import create_vector_index, create_property_graph_index
from tools.custom_tools import multiply_tool, add_tool, create_budget_tool
from agents.react_agent import create_react_agent
from utils.google_drive import mount_drive

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec

import nest_asyncio
import os

def main():
    # Access environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    llama_cloud_api_key = os.getenv('LLAMA_CLOUD_API_KEY')
    
    # Mount Google Drive
    mount_drive()

    # Set up LLM and embedding model
    Settings.llm = Gemini(model="models/gemini-1.5-flash-latest", temperature=0, request_timeout=360.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load documents
    documents = SimpleDirectoryReader("./drive/MyDrive/RAG/Agent").load_data()

    # Create index and query engine
    index = create_vector_index(documents, embed_model)
    query_engine = index.as_query_engine()

    # Create budget tool
    budget_tool = create_budget_tool(query_engine)

    # Create ReAct agent
    agent = create_react_agent([multiply_tool, add_tool, budget_tool], verbose=True)

    # Example queries
    queries = [
        "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
        "How much was allocated to implement a means-tested dental care program in the 2023 Canadian federal budget?",
        "How much was the total of those two allocations added together? Use a tool to answer any questions."
    ]

    for query in queries:
        response = agent.chat(query)
        print(f"Query: {query}\nResponse: {response}\n")

    # LlamaParse example
    nest_asyncio.apply()
    documents2 = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=True).load_data("./drive/MyDrive/RAG/Agent/2023_canadian_budget.pdf")
    index2 = create_vector_index(documents2, embed_model)
    query_engine2 = index2.as_query_engine()

    response2 = query_engine2.query(
        "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
    )
    print(f"LlamaParse Response: {response2}")

    # Yahoo Finance example
    finance_tools = YahooFinanceToolSpec().to_tool_list()
    finance_tools.extend([multiply_tool, add_tool])
    finance_agent = create_react_agent(finance_tools, verbose=True, truncate=True)

    finance_response = finance_agent.chat("What is the current price of NVDA?")
    print(f"Yahoo Finance Response: {finance_response}")

    # Property Graph Index example
    index3 = create_property_graph_index(documents, Settings.llm, embed_model)
    index3.property_graph_store.save_networkx_graph(name="./kg.html")

if __name__ == "__main__":
    main()
