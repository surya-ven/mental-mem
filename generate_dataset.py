# generate_dataset.py

import json
# Import Union for the flexible schema
from typing import Any, List, Optional, Dict, Union
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

from llm_utils import call_structured_llm


class DialogueTurn(BaseModel):
    role: str = Field(...,
                      description="The speaker's role, either 'counselor' or 'user'.")
    content: str = Field(...,
                         description="The verbatim content of the dialogue turn.")


class TherapyProfile(BaseModel):
    values: List[str] = Field(...,
                              description="A list of the user's core values.")
    goals: List[str] = Field(...,
                             description="A list of the user's therapeutic goals.")
    other_info: Dict[str, Any] = Field(
        default_factory=dict, description="Other dynamic info from the profile.")

# --- FIX: Make evolution_notes flexible for consistency ---


class EvolvedTherapyProfile(TherapyProfile):
    evolution_notes: Union[str, Dict[str, Any]] = Field(
        ..., description="A detailed summary of how and why the user's values or goals have shifted.")


class Session1Response(BaseModel):
    transcript: List[DialogueTurn]
    profile: TherapyProfile


class Session2Response(BaseModel):
    transcript: List[DialogueTurn]
    profile: EvolvedTherapyProfile


class Session3Response(BaseModel):
    transcript: List[DialogueTurn]


# The prompts and main functions remain the same as the previous working version.
PROMPT_SESSION_1 = """
You are a creative writer and clinical psychologist. Based on the user's core problem below, generate an initial therapy session.
**Core Problem:**
---
{seed_text}
---
**Your Task:**
Generate a JSON object with two top-level keys: "transcript" and "profile".
1.  `transcript`: A list of 4-6 conversational turns. Each turn MUST have a "role" ('counselor' or 'user') and a "content" key.
2.  `profile`: An object representing the user's initial therapeutic profile. It MUST have a "values" key (a list of strings) and a "goals" key (a list of strings).
Your entire output must be a single, valid JSON object.
"""
PROMPT_SESSION_2 = """
You are a creative writer and clinical psychologist continuing a client's story.
**Context from Session 1:**
The user's initial profile is: {initial_profile_str}
The transcript was: {session_1_transcript_str}
**Your Task:**
Generate a JSON object for "Session 2" with two top-level keys: "transcript" and "profile".
1.  `transcript`: A 4-6 turn conversation where the user has a significant "aha!" moment, evolving their perspective. Each turn MUST have "role" and "content".
2.  `profile`: An 'Evolved Profile' object. It MUST contain the original "values" and "goals", plus a new key "evolution_notes" that clearly describes the user's profound shift. This can be a string or a detailed object.
Your entire output must be a single, valid JSON object.
"""
PROMPT_SESSION_3 = """
You are a creative writer and clinical psychologist concluding a client's story.
**Context - User's Evolved Profile:**
{evolved_profile_str}
**Your Task:**
Generate a JSON object with one top-level key: "transcript".
The transcript should be a short, 2-turn conversation. The final user turn is the **Test Probe**, an ambiguous problem that requires understanding the user's *evolved* profile to answer correctly. Each turn MUST have "role" and "content".
Your entire output must be a single, valid JSON object.
"""


def generate_case_study(case_id: str, seed_text: str) -> Optional[Dict[str, Any]]:
    # This function remains the same
    print(f"\n--- Generating Case Study {case_id} ---")
    print("Step 1: Generating Onboarding Session...")
    output1 = call_structured_llm(PROMPT_SESSION_1.format(
        seed_text=seed_text), Session1Response)
    if not output1:
        return None
    transcript1, initial_profile = output1.transcript, output1.profile
    print("Step 2: Generating Evolution Session...")
    prompt2 = PROMPT_SESSION_2.format(
        initial_profile_str=initial_profile.model_dump_json(indent=2),
        session_1_transcript_str=json.dumps(
            [t.model_dump() for t in transcript1])
    )
    output2 = call_structured_llm(prompt2, Session2Response)
    if not output2:
        return None
    transcript2, evolved_profile = output2.transcript, output2.profile
    print("Step 3: Generating Test Probe Session...")
    prompt3 = PROMPT_SESSION_3.format(
        evolved_profile_str=evolved_profile.model_dump_json(indent=2))
    output3 = call_structured_llm(prompt3, Session3Response)
    if not output3:
        return None
    transcript3 = output3.transcript
    case_study = {
        "case_id": case_id, "seed_text": seed_text,
        "initial_profile": initial_profile.model_dump(),
        "evolved_profile": evolved_profile.model_dump(),
        "sessions": [
            {"session_number": 1, "transcript": [
                t.model_dump() for t in transcript1]},
            {"session_number": 2, "transcript": [
                t.model_dump() for t in transcript2]},
            {"session_number": 3, "transcript": [
                t.model_dump() for t in transcript3]}
        ]
    }
    print(f"--- Successfully Generated Case Study {case_id} ---")
    return case_study


if __name__ == "__main__":
    # This main block remains the same
    SEED_DATA_FILENAME = os.path.join("seed_data", "counseling_data_seed.json")
    OUTPUT_DIR = "output"
    OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "counseling_dataset.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(SEED_DATA_FILENAME, 'r') as f:
            seed_dataset = json.load(f)
        print(f"Loaded {len(seed_dataset)} seeds from '{SEED_DATA_FILENAME}'.")
    except FileNotFoundError:
        print(
            f"ERROR: Seed data file not found. Please create '{SEED_DATA_FILENAME}'.")
        exit()
    final_dataset = []
    print(f"\nStarting generation for all {len(seed_dataset)} seeds...")
    for i, seed_data in enumerate(tqdm(seed_dataset, desc="Generating Case Studies")):
        case_id = f"case_{i + 1:03d}"
        seed_text = seed_data.get("questionText")
        if not seed_text:
            print(f"Warning: Seed #{i+1} is missing 'questionText'. Skipping.")
            continue
        case = generate_case_study(case_id, seed_text)
        if case:
            final_dataset.append(case)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(final_dataset, f, indent=4)
    print(
        f"\nGeneration complete. Saved {len(final_dataset)} case studies to '{OUTPUT_FILENAME}'.")
