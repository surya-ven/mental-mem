# adaptive_model.py

import json
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field

from llm_utils import call_structured_llm
from baseline_model import format_transcript

# --- UPDATED SCHEMAS to include new fields ---


class TherapyProfile(BaseModel):
    values: List[str]
    goals: List[str]
    preferred_style: str
    other_info: Dict[str, Any] = Field(default_factory=dict)


class EvolvedTherapyProfile(TherapyProfile):
    evolution_notes: Union[str, Dict[str, Any]]
    agreed_task: str


class CounselorResponse(BaseModel):
    response: str

# -------------------------------------------------


def _run_profile_synthesis(case_data: Dict[str, Any]) -> Optional[EvolvedTherapyProfile]:
    """Step 1: The "Evolver" LLM synthesizes the richer evolved profile."""
    initial_profile = case_data['initial_profile']
    synthesis_transcript = format_transcript(case_data['sessions'][:2])

    prompt = f"""
    You are a supervising clinical psychologist. Your task is to analyze the following case and produce an updated user profile in JSON format.

    **CASE FILES:**
    - **Initial Profile:** {json.dumps(initial_profile, indent=2)}
    - **Session Transcripts (1 & 2):** {synthesis_transcript}

    **ANALYSIS & TASK:**
    1.  Review the initial profile and conversation.
    2.  Identify the user's evolution and the task they agreed to at the end of Session 2.
    3.  Produce a single JSON object for the Evolved Profile. It MUST include "values", "goals", "preferred_style", "evolution_notes", and the **"agreed_task"**.
    """
    predicted_profile = call_structured_llm(prompt, EvolvedTherapyProfile)
    return predicted_profile


def run_adaptive_counselor(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full pipeline for the Adaptive Counselor, now focusing on the agreed task."""
    predicted_evolved_profile = _run_profile_synthesis(case_data)

    if not predicted_evolved_profile:
        return {"response": "Error during profile synthesis.", "predicted_profile": None}

    full_transcript = format_transcript(case_data['sessions'])

    # --- UPDATED PROMPT to use the agreed_task and preferred_style ---
    prompt = f"""
    You are a highly perceptive AI Counselor. You will now respond to a user, applying a specific agreed-upon task.

    **USER'S ADAPTED PROFILE:**
    {predicted_evolved_profile.model_dump_json(indent=2)}

    **FULL CASE HISTORY:**
    {full_transcript}

    **YOUR TASK:**
    You must now respond to the user's final message. Structure your response to accomplish two goals:
    1.  **Task Application**: Your response MUST be built around the **"agreed_task"**: "{predicted_evolved_profile.agreed_task}".
    2.  **Style Adherence**: Your tone and language MUST match the user's **"preferred_style"**: "{predicted_evolved_profile.preferred_style}".

    Output your response in a valid JSON object with a single key "response".
    """
    response_model = call_structured_llm(prompt, CounselorResponse)
    response_text = response_model.response if response_model else "Error generating adaptive response."

    return {
        "response": response_text,
        "predicted_profile": predicted_evolved_profile.model_dump()
    }
