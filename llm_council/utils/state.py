"""State definitions for the LLM Council workflow."""

from typing import List, Annotated
from operator import add
from langgraph.graph import MessagesState


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class CouncilResponse(dict):
    """Individual response from a council member."""
    pass


class CouncilRanking(dict):
    """Ranking from a council member."""
    pass


class CouncilState(MessagesState):
    """
    State for the LLM Council workflow.
    
    Extends MessagesState to enable the LangSmith Chat interface.
    The 'messages' field is inherited from MessagesState.
    """
    # Stage 1: Individual responses (uses add reducer for parallel updates)
    stage1_responses: Annotated[List[CouncilResponse], add]
    
    # Stage 2: Rankings (uses add reducer for parallel updates)
    stage2_rankings: Annotated[List[CouncilRanking], add]
    
    # Label mapping (uses merge_dicts reducer for parallel updates)
    label_to_model: Annotated[dict, merge_dicts]
    
    # Stage 3: Final synthesis
    final_response: str
    chairman_model: str
    
    # Metadata
    aggregate_rankings: List[dict]
