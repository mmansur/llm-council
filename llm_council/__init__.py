"""
LLM Council: Multi-Model AI Advisory Board

A LangGraph implementation of Andrej Karpathy's LLM Council concept.
"""

from llm_council.agent import graph, ask_council

__version__ = "0.1.0"

__all__ = ["graph", "ask_council"]
