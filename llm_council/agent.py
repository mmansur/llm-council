"""
LLM Council: Multi-Model AI Advisory Board

Implementation of Andrej Karpathy's LLM Council concept using LangGraph.

The council works through a 3-stage process:
1. Stage 1: First Opinions - All LLMs respond to the user query individually
2. Stage 2: Peer Review - Each LLM ranks the other LLMs' anonymized responses
3. Stage 3: Chairman Synthesis - A designated Chairman LLM synthesizes the final answer
"""

import os
from langgraph.graph import StateGraph, START, END

from llm_council.utils.state import CouncilState
from llm_council.utils.nodes import (
    COUNCIL_MODELS,
    create_stage1_node,
    create_stage2_node,
    stage1_collector,
    stage2_collector,
    stage3_chairman_synthesis,
)


def build_council_graph() -> StateGraph:
    """Build the complete LLM Council workflow graph."""
    
    # Create the graph
    builder = StateGraph(CouncilState)
    
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
    
    # EDGES
    
    # Stage 1: Parallel execution from START
    for node_name in stage1_node_names:
        builder.add_edge(START, node_name)
        builder.add_edge(node_name, "stage1_collector")
    
    # Stage 2: Parallel execution after Stage 1 collector
    for node_name in stage2_node_names:
        builder.add_edge("stage1_collector", node_name)
        builder.add_edge(node_name, "stage2_collector")
    
    # Stage 3: Chairman synthesis after Stage 2 collector
    builder.add_edge("stage2_collector", "chairman_synthesis")
    builder.add_edge("chairman_synthesis", END)
    
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
        "user_query": question,
        "stage1_responses": [],
        "stage2_rankings": [],
        "label_to_model": {},
        "final_response": "",
        "chairman_model": "",
        "aggregate_rankings": []
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
    print(f"\nChairman: {result['chairman_model']}\n")
    print(result["final_response"])
    
    print("\n" + "=" * 80)
    print("AGGREGATE RANKINGS")
    print("=" * 80)
    for i, agg in enumerate(result["aggregate_rankings"], 1):
        print(f"  {i}. {agg['model']} - Average Rank: {agg['average_rank']}")
