# generate_dataset.py

import json
from typing import Any, List, Optional, Dict, Union
# Import model_validator for robust parsing
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
import os

from llm_utils import call_structured_llm

# --- UPDATED DialogueTurn Schema ---


class DialogueTurn(BaseModel):
    role: str = Field(...,
                      description="The speaker's role, either 'counselor' or 'user'.")
    content: str = Field(...,
                         description="The verbatim content of the dialogue turn.")

    # --- FIX: Add a validator to handle incorrect keying from the LLM ---
    @model_validator(mode='before')
    @classmethod
    def pre_process_turn_data(cls, data: Any) -> Any:
        # This function runs before standard validation to fix common LLM errors.
        if isinstance(data, dict):
            # If the keys are already correct, do nothing.
            if 'role' in data and 'content' in data:
                return data

            # If the LLM used the role as the key (e.g., {"therapist": "..."}),
            # transform it into the correct format.
            if len(data) == 1:
                role_key, content = list(data.items())[0]

                # Standardize role names
                standardized_role = "user"
                if role_key.lower() in ["therapist", "counselor", "assistant"]:
                    standardized_role = "counselor"
                elif role_key.lower() in ["client", "user", "human"]:
                    standardized_role = "user"

                return {"role": standardized_role, "content": content}

        # If the data is not in a format we can fix, let it pass through
        # to fail standard validation with a clear error.
        return data

# All other schemas remain the same as they were working correctly.


class TherapyProfile(BaseModel):
    values: List[str]
    goals: List[str]
    preferred_style: str = Field(
        description="The user's preferred communication style (e.g., 'direct', 'reflective').")
    other_info: Dict[str, Any] = Field(default_factory=dict)


class EvolvedTherapyProfile(TherapyProfile):
    evolution_notes: Union[str, Dict[str, Any]]
    agreed_task: str = Field(
        description="The concrete task or approach agreed upon by the counselor and user.")


class Session1Response(BaseModel):
    transcript: List[DialogueTurn]
    profile: TherapyProfile


class Session2Response(BaseModel):
    transcript: List[DialogueTurn]
    profile: EvolvedTherapyProfile


class Session3Response(BaseModel):
    transcript: List[DialogueTurn]


# All prompts and functions remain the same. The fix is entirely in the schema definition.
PROMPT_SESSION_1 = """
You are a creative writer and clinical psychologist. Based on the user's core problem below, generate an initial therapy session.
**Core Problem:**
---
{seed_text}
---
**Your Task:**
Generate a JSON object with "transcript" and "profile" keys.
1.  `transcript`: A 4-6 turn conversation. During the conversation, the counselor MUST ask the user about their **preferred communication style** (e.g., direct and advice-oriented vs. reflective and questioning). Each turn MUST have a "role" and "content" key.
2.  `profile`: An initial profile object. It MUST have "values", "goals", and the **"preferred_style"** key based on the user's answer.

Your entire output must be a single, valid JSON object.
"""
PROMPT_SESSION_2 = """
You are a creative writer and clinical psychologist continuing a client's story.
**Context from Session 1:**
The user's initial profile is: {initial_profile_str}
The transcript was: {session_1_transcript_str}
**Your Task:**
Generate a JSON object for "Session 2" with "transcript" and "profile" keys.
1.  `transcript`: A 4-6 turn conversation where the user has a significant "aha!" moment. The conversation MUST conclude with the counselor and user **explicitly agreeing on a concrete task** for the next step.
2.  `profile`: An 'Evolved Profile' object. It MUST contain "values", "goals", "preferred_style", "evolution_notes", and the new **"agreed_task"** key based on the conclusion of the transcript.

Your entire output must be a single, valid JSON object.
"""
PROMPT_SESSION_3 = """
You are a creative writer and clinical psychologist concluding a client's story.
**Context - User's Evolved Profile (including the agreed_task):**
{evolved_profile_str}
**Your Task:**
Generate a JSON object with one top-level key: "transcript".
The transcript should be a short, 2-turn conversation. The final user turn is the **Test Probe**, an ambiguous problem where the correct response would involve the 'agreed_task'.
"""


def generate_case_study(case_id: str, seed_text: str) -> Optional[Dict[str, Any]]:
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
            continue
        case = generate_case_study(case_id, seed_text)
        if case:
            final_dataset.append(case)
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(final_dataset, f, indent=4)
    print(
        f"\nGeneration complete. Saved {len(final_dataset)} case studies to '{OUTPUT_FILENAME}'.")
