from llama_index.core.agent import ReActAgent

def create_react_agent(tools, verbose=True, truncate=False):
    return ReActAgent.from_tools(tools, verbose=verbose, truncate=truncate)
