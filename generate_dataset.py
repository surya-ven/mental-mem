# generate_dataset.py

import json
from typing import List, Optional, Dict
# Field is not used but good to keep for consistency if needed later
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

from llm_utils import call_api_client
import config

# --- Schemas for Dataset Generation ---


class DialogueTurn(BaseModel):
    role: str
    content: str


class Session(BaseModel):  # This class seems defined but not directly used by TranscriptResponse
    session_number: int
    transcript: List[DialogueTurn]


# --- Updated Prompts for a 6-Session Arc ---
SESSION_PROMPTS = [
    # Session 1: Onboarding
    "Generate the transcript for Session 1, an onboarding session for the following user problem: {seed_text}. The counselor should focus on building rapport, understanding the core issues, and asking about the user's preferred communication style.",
    # Session 2: Deeper Dive
    "Generate Session 2. Based on the previous session, dive deeper into the user's feelings and history related to the core problem.",
    # Session 3: First Insight
    "Generate Session 3. The user should have a minor insight or a new perspective on their problem, guided by the counselor.",
    # Session 4: Setback or Complication
    "Generate Session 4. The user reports a setback or a new complication related to their progress, expressing doubt or frustration.",
    # Session 5: Consolidation & Planning
    "Generate Session 5. The counselor helps the user process the setback from Session 4, consolidate their learning, and agree on a concrete plan or task for the user to try.",
    # Session 6: The Test Probe
    "Generate Session 6. This is a short, 2-turn session. The user reports on a new, ambiguous situation. This will be the 'Test Probe' to evaluate other models."
]


def generate_case_study(case_id: str, seed_text: str) -> Optional[Dict]:
    print(f"\n--- Generating Case Study {case_id} ---")
    full_case = {"case_id": case_id, "seed_text": seed_text, "sessions": []}
    history = ""

    for i, prompt_template in enumerate(SESSION_PROMPTS):
        session_number = i + 1
        print(f"Step {session_number}: Generating Session {session_number}...")

        prompt = f"You are a creative writer and clinical psychologist. Your task is to generate a realistic therapy session transcript.\n\n"
        prompt += f"PREVIOUS HISTORY (for context):\n{history if history else 'This is the first session.'}\n\n"
        prompt += f"CURRENT TASK: {prompt_template.format(seed_text=seed_text)}\n\n"
        prompt += "Output a JSON object with a 'transcript' key, containing a list of dialogue turns. Each turn must have a 'role' ('counselor' or 'user') and a 'content' key."

        class TranscriptResponse(BaseModel):
            transcript: List[DialogueTurn]

        response = call_api_client(
            prompt, TranscriptResponse, model=config.DATA_GEN_MODEL)
        if not response:
            print(
                f"Failed to generate session {session_number}. Aborting case.")
            return None

        session_data = {"session_number": session_number, "transcript": [
            # .model_dump() is correct here for JSON serialization
            t.model_dump() for t in response.transcript]}
        full_case["sessions"].append(session_data)

        # --- FIX: Use dot notation to access Pydantic model attributes ---
        session_text = "\n".join(
            [f"{t.role}: {t.content}" for t in response.transcript])
        history += f"\n--- Session {session_number} ---\n{session_text}"

    print(
        f"--- Successfully Generated Full 6-Session Case Study {case_id} ---")
    return full_case


if __name__ == "__main__":
    SEED_DATA_FILENAME = os.path.join("seed_data", "counseling_data_seed.json")
    OUTPUT_FILENAME = os.path.join(
        "output", "counseling_dataset_6sessions.json")
    os.makedirs("output", exist_ok=True)
    try:
        with open(SEED_DATA_FILENAME, 'r') as f:
            seed_dataset = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Seed data file '{SEED_DATA_FILENAME}' not found.")
        exit()

    final_dataset = []
    for i, seed_data in enumerate(tqdm(seed_dataset, desc="Generating Case Studies")):
        # Ensure seed_data is a dictionary and has "questionText"
        if isinstance(seed_data, dict) and "questionText" in seed_data:
            case = generate_case_study(
                f"case_{i+1:03d}", seed_data["questionText"])
            if case:
                final_dataset.append(case)
        else:
            print(
                f"Warning: Seed data at index {i} is not in the expected format or missing 'questionText'. Skipping.")

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print(
        f"\nGeneration complete. Saved {len(final_dataset)} cases to '{OUTPUT_FILENAME}'.")
