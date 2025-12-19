"""
Pydantic schemas for validating analysis specifications.

This module defines structured models for user input validation,
ensuring all required fields are present and within acceptable limits.
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import Self


# Maximum total character limit for all spec fields combined
MAX_TOTAL_CHARS = 200_000


class AnalysisSpec(BaseModel):
    """
    Structured specification for Delphes analysis code generation.
    
    Attributes:
        selection_cuts: Particle selection and cut definitions
        plots_for_validation: Histogram specifications (bins, ranges)
        output_structure: Output format and statistics requirements
    """
    selection_cuts: str
    plots_for_validation: str
    output_structure: str

    @field_validator('selection_cuts', 'plots_for_validation', 'output_structure', mode='before')
    @classmethod
    def strip_and_check_non_empty(cls, v: str, info) -> str:
        """Ensure each field is non-empty after stripping whitespace."""
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must be non-empty")
        return stripped

    @model_validator(mode='after')
    def check_total_length(self) -> Self:
        """Ensure combined length of all fields doesn't exceed limit."""
        total = (
            len(self.selection_cuts) +
            len(self.plots_for_validation) +
            len(self.output_structure)
        )
        if total > MAX_TOTAL_CHARS:
            raise ValueError(
                f"Total spec length ({total} chars) exceeds maximum "
                f"allowed ({MAX_TOTAL_CHARS} chars)"
            )
        return self