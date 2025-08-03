from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

class PolicyQuery(BaseModel):
    """Request model for policy query endpoint"""
    documents: HttpUrl = Field(
        ...,
        description="URL to the policy document (PDF)",
        example="https://example.com/policy.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        description="List of questions about the policy",
        example=["What is the grace period?", "What is covered?"],
    )

class PolicyResponse(BaseModel):
    """Response model for policy query endpoint"""
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the questions",
        example=[
            "The grace period is 30 days.",
            "The policy covers medical expenses up to the sum insured."
        ]
    )

class ParsedQuery(BaseModel):
    """Model for parsed query information"""
    policy_section: str = Field(..., description="Relevant section of the policy")
    query_type: str = Field(..., description="Type of the query")
    key_terms: List[str] = Field(
        default_factory=list,
        description="List of key terms extracted from the query"
    )
