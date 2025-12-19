"""Node functions for the LLM Council workflow."""

import os
import re
from typing import List, Dict, Any
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from llm_council.utils.state import CouncilState


# Configuration - can be overridden via environment variables
COUNCIL_MODELS = os.environ.get(
    "COUNCIL_MODELS", 
    "openai/gpt-4o,anthropic/claude-3.5-sonnet,google/gemini-2.0-flash-exp:free"
).split(",")

CHAIRMAN_MODEL = os.environ.get("CHAIRMAN_MODEL", "openai/gpt-4o")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm(model_name: str) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance configured for OpenRouter."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/karpathy/llm-council",
            "X-Title": "LLM Council",
        },
        temperature=0.7,
        max_tokens=2000,
    )


# Create LLM instances
council_llms: Dict[str, ChatOpenAI] = {}
chairman_llm: ChatOpenAI = None


def initialize_llms():
    """Initialize LLM instances. Called when module is loaded."""
    global council_llms, chairman_llm
    
    if os.environ.get("OPENROUTER_API_KEY"):
        council_llms = {model: get_llm(model) for model in COUNCIL_MODELS}
        chairman_llm = get_llm(CHAIRMAN_MODEL)


def get_user_query(state: CouncilState) -> str:
    """Extract the user query from state - checks both _user_query and messages."""
    # First check if we have the extracted query
    if state.get("_user_query"):
        return state["_user_query"]
    
    # Fallback: extract from messages
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    
    return ""


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """Parse the FINAL RANKING section from the model's response."""
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches
    
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def prepare_ranking_context(state: CouncilState) -> tuple:
    """Prepare anonymized responses and label mapping for ranking."""
    responses = state.get("stage1_responses", [])
    
    labels = [chr(65 + i) for i in range(len(responses))]
    
    label_to_model = {
        f"Response {label}": resp.get("model", "unknown")
        for label, resp in zip(labels, responses)
    }
    
    responses_text = "\n\n".join([
        f"Response {label}:\n{resp.get('response', '')}"
        for label, resp in zip(labels, responses)
    ])
    
    return responses_text, label_to_model


def calculate_aggregate_rankings(state: CouncilState) -> List[dict]:
    """Calculate aggregate rankings across all models."""
    label_to_model = state.get("label_to_model", {})
    rankings = state.get("stage2_rankings", [])
    
    model_positions = defaultdict(list)
    
    for ranking in rankings:
        parsed_ranking = ranking.get("parsed_ranking", [])
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
    
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })
    
    aggregate.sort(key=lambda x: x["average_rank"])
    
    return aggregate


def create_stage1_node(model_name: str):
    """Factory function to create a Stage 1 node for a specific model."""
    
    def stage1_node(state: CouncilState) -> dict:
        """Query a single council member for their initial response."""
        # Initialize LLMs if not already done
        if not council_llms:
            initialize_llms()
        
        llm = council_llms.get(model_name)
        if not llm:
            return {
                "stage1_responses": [{
                    "model": model_name,
                    "response": f"[Error: Model {model_name} not initialized]"
                }]
            }
        
        user_query = get_user_query(state)
        
        try:
            response = llm.invoke([HumanMessage(content=user_query)])
            return {
                "stage1_responses": [{
                    "model": model_name,
                    "response": response.content
                }]
            }
        except Exception as e:
            return {
                "stage1_responses": [{
                    "model": model_name,
                    "response": f"[Error: {str(e)}]"
                }]
            }
    
    return stage1_node


def create_stage2_node(model_name: str):
    """Factory function to create a Stage 2 ranking node for a specific model."""
    
    def stage2_node(state: CouncilState) -> dict:
        """Have a council member rank all anonymized responses."""
        # Initialize LLMs if not already done
        if not council_llms:
            initialize_llms()
        
        llm = council_llms.get(model_name)
        if not llm:
            return {
                "stage2_rankings": [{
                    "model": model_name,
                    "ranking_text": f"[Error: Model {model_name} not initialized]",
                    "parsed_ranking": []
                }],
                "label_to_model": {}
            }
        
        responses_text, label_to_model = prepare_ranking_context(state)
        user_query = get_user_query(state)
        
        ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")

Example format:

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""
        
        try:
            response = llm.invoke([HumanMessage(content=ranking_prompt)])
            parsed = parse_ranking_from_text(response.content)
            
            return {
                "stage2_rankings": [{
                    "model": model_name,
                    "ranking_text": response.content,
                    "parsed_ranking": parsed
                }],
                "label_to_model": label_to_model
            }
        except Exception as e:
            return {
                "stage2_rankings": [{
                    "model": model_name,
                    "ranking_text": f"[Error: {str(e)}]",
                    "parsed_ranking": []
                }],
                "label_to_model": label_to_model
            }
    
    return stage2_node


def stage1_collector(state: CouncilState) -> dict:
    """Collector node for Stage 1 - synchronization point."""
    return {}


def stage2_collector(state: CouncilState) -> dict:
    """Collector node for Stage 2 - synchronization point."""
    return {}


def stage3_chairman_synthesis(state: CouncilState) -> dict:
    """Chairman synthesizes all responses and rankings into final answer."""
    global chairman_llm
    
    # Initialize LLMs if not already done
    if not chairman_llm:
        initialize_llms()
    
    if not chairman_llm:
        return {
            "final_response": "Error: Chairman LLM not initialized. Check OPENROUTER_API_KEY.",
            "chairman_model": CHAIRMAN_MODEL,
            "aggregate_rankings": []
        }
    
    user_query = get_user_query(state)
    
    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {resp.get('model', 'unknown')}\nResponse: {resp.get('response', '')}"
        for resp in state.get("stage1_responses", [])
    ])
    
    stage2_text = "\n\n".join([
        f"Model: {ranking.get('model', 'unknown')}\nRanking: {ranking.get('ranking_text', '')}"
        for ranking in state.get("stage2_rankings", [])
    ])
    
    chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""
    
    try:
        response = chairman_llm.invoke([HumanMessage(content=chairman_prompt)])
        aggregate_rankings = calculate_aggregate_rankings(state)
        
        return {
            "final_response": response.content,
            "chairman_model": CHAIRMAN_MODEL,
            "aggregate_rankings": aggregate_rankings
        }
    except Exception as e:
        return {
            "final_response": f"Error: Unable to generate final synthesis. {str(e)}",
            "chairman_model": CHAIRMAN_MODEL,
            "aggregate_rankings": []
        }
