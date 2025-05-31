# baseline_model.py

from typing import Dict, Any, List
from pydantic import BaseModel
from llm_utils import call_structured_llm


class CounselorResponse(BaseModel):
    response: str


def format_transcript(sessions: List[Dict[str, Any]]) -> str:
    """Formats the session transcripts into a single string."""
    full_transcript = []
    for session in sessions:
        full_transcript.append(f"--- Session {session['session_number']} ---")
        for turn in session['transcript']:
            full_transcript.append(
                f"{turn['role'].capitalize()}: {turn['content']}")
    return "\n".join(full_transcript)


def run_static_counselor(case_data: Dict[str, Any]) -> str:
    """Runs the baseline Static Counselor."""
    initial_profile = case_data['initial_profile']
    full_transcript = format_transcript(case_data['sessions'])

    # --- UPDATED PROMPT to include preferred_style ---
    prompt = f"""
    You are an AI Counselor. Your personality and therapeutic approach are defined by the user's initial profile.

    **YOUR STATIC PROFILE (DO NOT DEVIATE):**
    - **Core Values to Uphold:** {', '.join(initial_profile.get('values', []))}
    - **Original Goals:** {', '.join(initial_profile.get('goals', []))}
    - **Communication Style to Use:** {initial_profile.get('preferred_style', 'empathetic')}

    **CASE HISTORY:**
    {full_transcript}

    **YOUR TASK:**
    Based on your STATIC PROFILE and the case history, provide a thoughtful and empathetic response to the user's final message.
    Your response should adhere to the communication style above.
    Output your response in a valid JSON object with a single key "response".
    """

    response_model = call_structured_llm(prompt, CounselorResponse)
    return response_model.response if response_model else "Error generating response."
