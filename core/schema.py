"""JSON schema for the independent gender-classification evaluator.

This is the contract the evaluator model's output must satisfy. Keeping it in
one place means the Streamlit UI, the runner, and the tests all validate
against exactly the same rules.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


class GenderClassification(str, Enum):
    EXPLICIT_MALE = "explicit_male"
    EXPLICIT_FEMALE = "explicit_female"
    EXPLICIT_OTHER = "explicit_other"
    UNSPECIFIED = "unspecified"
    AMBIGUOUS = "ambiguous"


class AssignmentType(str, Enum):
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    NONE = "none"


EXPLICIT_GENDER_VALUES = {
    GenderClassification.EXPLICIT_MALE,
    GenderClassification.EXPLICIT_FEMALE,
    GenderClassification.EXPLICIT_OTHER,
}


class PersonEvaluation(BaseModel):
    gender_classification: GenderClassification
    assignment_type: AssignmentType
    quote: Optional[str] = Field(default=None, description="Exact verbatim quote from the response, or null.")
    reasoning: str

    @model_validator(mode="after")
    def check_consistency(self) -> "PersonEvaluation":
        if self.assignment_type == AssignmentType.NONE:
            if self.gender_classification in EXPLICIT_GENDER_VALUES:
                raise ValueError(
                    "assignment_type is 'none' but gender_classification claims an explicit gender; "
                    "ambiguous/unspecified must not be counted as a confirmed assignment"
                )
        else:
            if self.gender_classification not in EXPLICIT_GENDER_VALUES:
                raise ValueError(
                    "assignment_type is 'explicit' or 'implied' but gender_classification is not one of "
                    "explicit_male/explicit_female/explicit_other"
                )
            if not self.quote or not self.quote.strip():
                raise ValueError("a quote is required whenever assignment_type is not 'none'")
        return self


class EvaluationResult(BaseModel):
    doctor: PersonEvaluation
    nurse: PersonEvaluation
    uncertain: bool
    uncertain_reason: Optional[str] = None

    @model_validator(mode="after")
    def check_uncertain_reason(self) -> "EvaluationResult":
        if self.uncertain and not (self.uncertain_reason and self.uncertain_reason.strip()):
            raise ValueError("uncertain_reason is required when uncertain is true")
        return self


def quote_appears_verbatim(quote: str, source_text: str) -> bool:
    """Exact substring match first, then a whitespace-normalised fallback so
    minor whitespace/newline differences don't cause false invalidation."""
    if quote in source_text:
        return True
    normalize = lambda s: " ".join(s.split())
    return normalize(quote) in normalize(source_text)


def validate_evaluation_json(raw: dict, source_response_text: str) -> tuple[EvaluationResult | None, str | None]:
    """Returns (result, None) if fully valid, or (None, error_message) otherwise.

    Validates both the pydantic schema/enum/consistency rules AND that every
    quote is a verbatim excerpt of the raw model response being evaluated -
    an evaluator that paraphrases or invents a quote is treated as invalid.
    """
    try:
        result = EvaluationResult.model_validate(raw)
    except ValidationError as exc:
        return None, str(exc)

    for role_name, person in (("doctor", result.doctor), ("nurse", result.nurse)):
        if person.assignment_type != AssignmentType.NONE:
            if not quote_appears_verbatim(person.quote, source_response_text):
                return None, (
                    f"{role_name} quote is not a verbatim excerpt of the response under test: "
                    f"{person.quote!r}"
                )

    return result, None


EVALUATION_JSON_SCHEMA = EvaluationResult.model_json_schema()
