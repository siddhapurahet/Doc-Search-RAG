"""Graph builder for LangGraph workflow"""

import importlib
import json
import time

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState

# #region agent log
def _dbg_log(hypothesis_id, location, message, data):
    try:
        with open("/Users/harshsiddhapura/Downloads/Het/Doc-Search-RAG/.cursor/debug-980812.log", "a") as f:
            f.write(json.dumps({"sessionId": "980812", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass

try:
    _nodes_mod = importlib.import_module("src.nodes.nodes")
    _dbg_log("A", "graph_builder.py:import", "nodes module loaded; checking exported names", {"dir_names": [n for n in dir(_nodes_mod) if not n.startswith("_")], "has_RAGNodes": hasattr(_nodes_mod, "RAGNodes"), "has_RAGNode": hasattr(_nodes_mod, "RAGNode")})
except Exception as e:
    _dbg_log("B", "graph_builder.py:import", "nodes module failed to load", {"error_type": type(e).__name__, "error": str(e)})
# #endregion

from src.nodes.nodes import RAGNodes
# from src.nodes.reactnodes import RAGNodes

class GraphBuilder:
    """Builds and manages the langgraph workflow"""

    def __init__(self, retriever, llm):
        """
        Initializes graph builder
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        # #region agent log
        _dbg_log("A", "graph_builder.py:__init__", "GraphBuilder init", {"has_retriever": retriever is not None, "has_llm": llm is not None})
        # #endregion
        self.nodes = RAGNodes(retriever=retriever, llm=llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph
        Returns:
            Compiled graph instance
        """

        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # Set the entry point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # Compile graph
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str)-> dict:
        """
        Run the RAG workflow
        Args:
            question: User question
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question = question)
        return self.graph.invoke(initial_state)