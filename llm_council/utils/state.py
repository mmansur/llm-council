"""State definitions for the LLM Council workflow."""

from typing import TypedDict, List, Annotated
from operator import add


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


class CouncilResponse(TypedDict):
    """Individual response from a council member."""
    model: str
    response: str


class CouncilRanking(TypedDict):
    """Ranking from a council member."""
    model: str
    ranking_text: str
    parsed_ranking: List[str]


class CouncilState(TypedDict):
    """State for the LLM Council workflow."""
    # Input
    user_query: str
    
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
