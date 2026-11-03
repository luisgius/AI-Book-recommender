"""
Pydantic schemas for LLM-as-Judge evaluation (Block 3).

These schemas define the structured outputs for:
1. Explanation quality judgment (3 dimensions)
2. Grounding validation
"""

from pydantic import BaseModel, Field


class JudgmentDimension(BaseModel):
    """
    Single dimension of LLM judgment.
    
    Each dimension has a score (1-5) and brief reasoning.
    """
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score from 1 (poor) to 5 (excellent)"
    )
   
    reasoning: str = Field(max_length=200, description="Brief justification for the score")


class ExplanationJudgmentLLM(BaseModel):
    """
    LLM-as-judge output for explanation quality (MVP: 3 dimensions).
    
    The LLM evaluates an explanation on three criteria:
    - Groundedness: Are claims supported by citations?
    - Clarity: Is the explanation clear and well-structured?
    - Relevance: Does it address the user's query intent?
    """
    groundedness: JudgmentDimension = Field(description="Are all claims supported by the cited evidence?")
    clarity: JudgmentDimension = Field( description="Is the explanation clear and actionable?")
    relevance: JudgmentDimension = Field( description="Does it address what the user asked for?")


    @property
    def overall_score(self) -> float:
        """Compute average score across all dimensions."""
        return (self.groundedness.score + self.clarity.score + self.relevance.score) / 3
        