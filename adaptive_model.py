# adaptive_model.py

import json
# Import Union for the flexible schema
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field

from llm_utils import call_structured_llm
from baseline_model import format_transcript

# --- FIX: Make evolution_notes flexible (string OR dictionary) ---


class TherapyProfile(BaseModel):
    values: List[str]
    goals: List[str]
    other_info: Dict[str, Any] = Field(default_factory=dict)


class EvolvedTherapyProfile(TherapyProfile):
    evolution_notes: Union[str, Dict[str, Any]]


class CounselorResponse(BaseModel):
    response: str
# -------------------------------------------------


def _run_profile_synthesis(case_data: Dict[str, Any]) -> Optional[EvolvedTherapyProfile]:
    """Step 1: The "Evolver" LLM."""
    initial_profile = case_data['initial_profile']
    synthesis_transcript = format_transcript(case_data['sessions'][:2])

    prompt = f"""
    You are a supervising clinical psychologist reviewing a patient's case file over time.
    Your task is to analyze the initial profile and session transcripts and generate an updated 'Evolved Profile' in JSON format.

    **INITIAL PROFILE:**
    {json.dumps(initial_profile, indent=2)}

    **SESSION TRANSCRIPTS (SESSIONS 1 & 2):**
    {synthesis_transcript}

    **ANALYSIS & TASK:**
    Read the transcripts and identify the key insight in Session 2. Generate an updated profile object that includes "values" and "goals", plus a new key "evolution_notes" that concisely summarizes the user's transformation. This summary can be a simple string or a more detailed object.

    Output a single, valid JSON object.
    """
    predicted_profile = call_structured_llm(prompt, EvolvedTherapyProfile)
    return predicted_profile

# The run_adaptive_counselor function remains the same as before.


def run_adaptive_counselor(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full pipeline for the Adaptive Counselor, adapted for the new structure."""
    predicted_evolved_profile = _run_profile_synthesis(case_data)

    if not predicted_evolved_profile:
        return {"response": "Error during profile synthesis.", "predicted_profile": None}

    full_transcript = format_transcript(case_data['sessions'])

    prompt = f"""
    You are a highly perceptive AI Counselor. You adapt your approach based on the user's growth.

    **YOUR ADAPTED PROFILE (Use this new understanding):**
    {predicted_evolved_profile.model_dump_json(indent=2)}

    **FULL CASE HISTORY:**
    {full_transcript}

    **YOUR TASK:**
    Based on your NEWLY ADAPTED PROFILE and the final user message, generate a response. Your response should reflect your understanding of the user's recent evolution, as described in `evolution_notes`.

    Output your response in a valid JSON object with a single key "response".
    """
    response_model = call_structured_llm(prompt, CounselorResponse)
    response_text = response_model.response if response_model else "Error generating adaptive response."

    return {
        "response": response_text,
        "predicted_profile": predicted_evolved_profile.model_dump()
    }
