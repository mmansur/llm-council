"""
LLM Council: Multi-Model AI Advisory Board

Implementation of Andrej Karpathy's LLM Council concept using LangGraph.

The council works through a 3-stage process:
1. Stage 1: First Opinions - All LLMs respond to the user query individually
2. Stage 2: Peer Review - Each LLM ranks the other LLMs' anonymized responses
3. Stage 3: Chairman Synthesis - A designated Chairman LLM synthesizes the final answer

This version uses MessagesState to enable the LangSmith Chat interface.
"""

import os
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage

from llm_council.utils.state import CouncilState
from llm_council.utils.nodes import (
    COUNCIL_MODELS,
    create_stage1_node,
    create_stage2_node,
    stage1_collector,
    stage2_collector,
    stage3_chairman_synthesis,
)


def extract_user_query(state: CouncilState) -> dict:
    """
    Extract the user query from the messages.
    This is the entry point that converts chat messages to our workflow.
    """
    messages = state.get("messages", [])
    
    # Get the last human message as the query
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    # Initialize the state fields
    return {
        "stage1_responses": [],
        "stage2_rankings": [],
        "label_to_model": {},
        "final_response": "",
        "chairman_model": "",
        "aggregate_rankings": [],
        # Store the query in a way nodes can access it
        "_user_query": user_query,
    }


def format_final_response(state: CouncilState) -> dict:
    """
    Format the final response as an AI message for the chat interface.
    """
    final_response = state.get("final_response", "")
    chairman_model = state.get("chairman_model", "")
    aggregate_rankings = state.get("aggregate_rankings", [])
    
    # Build a comprehensive response
    response_parts = []
    
    # Add the main answer
    response_parts.append(final_response)
    
    # Add rankings summary if available
    if aggregate_rankings:
        response_parts.append("\n\n---\n**Council Rankings:**")
        for i, agg in enumerate(aggregate_rankings, 1):
            response_parts.append(f"\n{i}. {agg['model']} (avg rank: {agg['average_rank']})")
    
    full_response = "".join(response_parts)
    
    # Return as an AI message for the chat interface
    return {
        "messages": [AIMessage(content=full_response)]
    }


def build_council_graph() -> StateGraph:
    """Build the complete LLM Council workflow graph."""
    
    # Create the graph with MessagesState-based CouncilState
    builder = StateGraph(CouncilState)
    
    # Entry node: Extract user query from messages
    builder.add_node("extract_query", extract_user_query)
    
    # STAGE 1: Add parallel nodes for initial responses
    stage1_node_names = []
    for i, model in enumerate(COUNCIL_MODELS):
        node_name = f"stage1_model_{i}"
        stage1_node_names.append(node_name)
        builder.add_node(node_name, create_stage1_node(model))
    
    # Add collector node for Stage 1
    builder.add_node("stage1_collector", stage1_collector)
    
    # STAGE 2: Add parallel nodes for rankings
    stage2_node_names = []
    for i, model in enumerate(COUNCIL_MODELS):
        node_name = f"stage2_model_{i}"
        stage2_node_names.append(node_name)
        builder.add_node(node_name, create_stage2_node(model))
    
    # Add collector node for Stage 2
    builder.add_node("stage2_collector", stage2_collector)
    
    # STAGE 3: Chairman synthesis
    builder.add_node("chairman_synthesis", stage3_chairman_synthesis)
    
    # Final node: Format response for chat interface
    builder.add_node("format_response", format_final_response)
    
    # EDGES
    
    # Start with query extraction
    builder.add_edge(START, "extract_query")
    
    # Stage 1: Parallel execution after query extraction
    for node_name in stage1_node_names:
        builder.add_edge("extract_query", node_name)
        builder.add_edge(node_name, "stage1_collector")
    
    # Stage 2: Parallel execution after Stage 1 collector
    for node_name in stage2_node_names:
        builder.add_edge("stage1_collector", node_name)
        builder.add_edge(node_name, "stage2_collector")
    
    # Stage 3: Chairman synthesis after Stage 2 collector
    builder.add_edge("stage2_collector", "chairman_synthesis")
    
    # Format response and end
    builder.add_edge("chairman_synthesis", "format_response")
    builder.add_edge("format_response", END)
    
    return builder.compile()


# Build and export the graph
# This is the compiled graph that LangSmith will use
graph = build_council_graph()


# Convenience function for direct invocation
def ask_council(question: str) -> dict:
    """
    Ask the LLM Council a question and get the synthesized response.
    
    Args:
        question: The question to ask the council
    
    Returns:
        The final state with all responses and the synthesized answer
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
    }
    
    return graph.invoke(initial_state)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the key differences between supervised and unsupervised machine learning?"
    
    print(f"Question: {question}\n")
    print("Running LLM Council...\n")
    
    result = ask_council(question)
    
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    
    # Get the last AI message
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            print(msg.content)
            break
