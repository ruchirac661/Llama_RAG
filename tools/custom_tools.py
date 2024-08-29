from llama_index.core.tools import QueryEngineTool, FunctionTool

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

def create_budget_tool(query_engine):
    return QueryEngineTool.from_defaults(
        query_engine,
        name="canadian_budget_2023",
        description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
    )
