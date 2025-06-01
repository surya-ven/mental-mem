# generate_dataset.py

import json
from typing import List, Optional, Dict, Any, Union
# Ensure model_validator is imported
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
import os

from llm_utils import call_api_client
import config

# --- Schemas for Dataset Generation ---


class DialogueTurn(BaseModel):
    role: str
    content: str

    @model_validator(mode='before')
    @classmethod
    def pre_process_turn_data(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Ideal case: keys are already correct
            if 'role' in data and 'content' in data:
                return data

            # --- UPDATED VALIDATOR: Check for "speaker" and "text" keys ---
            elif 'speaker' in data and 'text' in data:
                role_from_speaker = data.get('speaker')
                content_from_text = data.get('text')

                standardized_role = "user"  # Default
                if isinstance(role_from_speaker, str):
                    if role_from_speaker.lower() in ["therapist", "counselor", "assistant", "ai"]:
                        standardized_role = "counselor"
                    elif role_from_speaker.lower() in ["client", "user", "human", "patient"]:
                        standardized_role = "user"
                return {"role": standardized_role, "content": content_from_text}

            # Existing check: role name used as key (e.g., {"therapist": "..."})
            elif len(data) == 1:
                role_key, content_value = list(data.items())[0]
                standardized_role = "user"  # Default
                if isinstance(role_key, str):
                    if role_key.lower() in ["therapist", "counselor", "assistant", "ai"]:
                        standardized_role = "counselor"
                    elif role_key.lower() in ["client", "user", "human", "patient"]:
                        standardized_role = "user"
                return {"role": standardized_role, "content": content_value}

        return data


class GeneratedInitialProfileElements(BaseModel):
    user_stated_values_session1: List[str] = Field(
        description="Values explicitly stated by the user in Session 1.")
    user_stated_goals_session1: List[str] = Field(
        description="Initial goals explicitly stated by the user in Session 1.")
    extracted_preferred_style: str = Field(
        description="The communication style explicitly chosen by the user in Session 1 (e.g., 'direct', 'reflective').")


class GeneratedSession5Outcome(BaseModel):
    key_insight_session5: str = Field(
        description="The main 'aha!' moment or realization from Session 5.")
    agreed_task_for_session6: str = Field(
        description="The specific, concrete task agreed upon at the end of Session 5 for the user to focus on next.")


class Session1DataResponse(BaseModel):
    transcript: List[DialogueTurn]
    generated_initial_profile_elements: GeneratedInitialProfileElements


class Session5DataResponse(BaseModel):
    transcript: List[DialogueTurn]
    generated_session5_outcome: GeneratedSession5Outcome


class GenericSessionTranscriptResponse(BaseModel):
    transcript: List[DialogueTurn]


SESSION_PROMPTS_CONFIG = [
    {
        "session_num": 1,
        "description": "Onboarding: Build rapport, understand core issues, AND EXPLICITLY ASK & CAPTURE user's preferred communication style.",
        "prompt_text_template": "Generate the transcript for Session 1, an onboarding session for the user problem: '{seed_text}'. The counselor MUST build rapport, understand core issues, and explicitly ask the user if they prefer a direct/advice-oriented style OR a reflective/questioning style. Based on the dialogue, also extract the user's stated values, initial goals, and their chosen preferred style.",
        "output_instruction": "Output a JSON object with two top-level keys: 'transcript' (list of dialogue turns with 'role' and 'content') AND 'generated_initial_profile_elements' (an object with 'user_stated_values_session1' (list of strings), 'user_stated_goals_session1' (list of strings), and 'extracted_preferred_style' (string - e.g., 'direct', 'reflective')).",
        "response_model": Session1DataResponse
    },
    {
        "session_num": 2,
        "description": "Deeper Dive: Explore feelings and history related to the core problem.",
        "prompt_text_template": "Generate Session 2. Based on the previous session (and user's preferred style: {preferred_style}), dive deeper into the user's feelings and history related to their core problem.",
        "output_instruction": "Output a JSON object with a 'transcript' key, containing a list of dialogue turns. Each turn must have 'role' and 'content'.",
        "response_model": GenericSessionTranscriptResponse
    },
    {
        "session_num": 3,
        "description": "First Insight: User has a minor insight or new perspective.",
        "prompt_text_template": "Generate Session 3. The user should have a minor insight or a new perspective on their problem, guided by the counselor (who is aware of preferred style: {preferred_style}).",
        "output_instruction": "Output a JSON object with a 'transcript' key, containing a list of dialogue turns. Each turn must have 'role' and 'content'.",
        "response_model": GenericSessionTranscriptResponse
    },
    {
        "session_num": 4,
        "description": "Setback or Complication: User reports a setback or new complication.",
        "prompt_text_template": "Generate Session 4. The user reports a setback or a new complication related to their progress, expressing doubt or frustration. Counselor adapts to preferred style: {preferred_style}.",
        "output_instruction": "Output a JSON object with a 'transcript' key, containing a list of dialogue turns. Each turn must have 'role' and 'content'.",
        "response_model": GenericSessionTranscriptResponse
    },
    {
        "session_num": 5,
        "description": "Consolidation & Planning: Process setback, consolidate learning, AND EXPLICITLY AGREE on a concrete task.",
        "prompt_text_template": "Generate Session 5. The counselor helps the user process the setback from Session 4 and consolidate learning (respecting preferred style: {preferred_style}). The session MUST conclude with the counselor and user explicitly agreeing on a very specific, actionable 'agreed_task_for_session6' (e.g., 'try one specific mindfulness exercise daily', 'write down three negative thoughts and challenge one'). Also extract the key insight from this session.",
        "output_instruction": "Output a JSON object with two top-level keys: 'transcript' (list of dialogue turns) AND 'generated_session5_outcome' (an object with 'key_insight_session5' (string) and 'agreed_task_for_session6' (string)).",
        "response_model": Session5DataResponse
    },
    {
        "session_num": 6,
        "description": "Test Probe: Short session where user presents an ambiguous situation related to the agreed task.",
        "prompt_text_template": "Generate Session 6. This is a short, 2-turn session. The user reports on a new, ambiguous situation that directly relates to or challenges the 'agreed_task_for_session6': '{agreed_task}'. This will be the 'Test Probe'. Counselor respects preferred style: {preferred_style}.",
        "output_instruction": "Output a JSON object with a 'transcript' key, containing a list of dialogue turns. Each turn MUST have 'role' and 'content' keys.",  # Reinforce key names
        "response_model": GenericSessionTranscriptResponse
    }
]


def generate_case_study(case_id: str, seed_text: str) -> Optional[Dict]:
    print(f"\n--- Generating Case Study {case_id} ---")
    full_case = {
        "case_id": case_id,
        "seed_text": seed_text,
        "sessions": [],
        "generated_initial_profile_elements": None,
        "generated_session5_outcome": None
    }
    history_for_llm = ""

    current_preferred_style = "reflective"
    current_agreed_task = "discuss progress"

    for session_config in SESSION_PROMPTS_CONFIG:
        session_number = session_config["session_num"]
        print(
            f"Step {session_number}: Generating Session {session_number} ({session_config['description']})...")

        prompt_text = session_config["prompt_text_template"].format(
            seed_text=seed_text,
            preferred_style=current_preferred_style,
            agreed_task=current_agreed_task
        )

        full_prompt_for_llm = (
            f"You are a creative writer and clinical psychologist. Your task is to generate realistic therapy session data.\n\n"
            f"CUMULATIVE HISTORY (for context):\n{history_for_llm if history_for_llm else 'This is the first session.'}\n\n"
            f"CURRENT TASK FOR SESSION {session_number}:\n{prompt_text}\n\n"
            f"OUTPUT INSTRUCTIONS:\n{session_config['output_instruction']}\n"
            f"Ensure your entire output is a single, valid JSON object. For transcripts, each turn MUST use 'role' and 'content' as keys."  # Extra reinforcement
        )

        response_data_model = session_config["response_model"]
        response = call_api_client(
            full_prompt_for_llm, response_data_model, model=config.DATA_GEN_MODEL
        )

        if not response:
            # Added case_id for clarity
            print(
                f"Failed to generate session {session_number}. Aborting case for {case_id}.")
            return None

        session_data_to_store = {"session_number": session_number, "transcript": [
            t.model_dump() for t in response.transcript]}
        full_case["sessions"].append(session_data_to_store)

        session_transcript_text = "\n".join(
            [f"{t.role}: {t.content}" for t in response.transcript])
        history_for_llm += f"\n--- Session {session_number} ---\n{session_transcript_text}"

        if isinstance(response, Session1DataResponse):
            full_case["generated_initial_profile_elements"] = response.generated_initial_profile_elements.model_dump()
            current_preferred_style = response.generated_initial_profile_elements.extracted_preferred_style
            history_for_llm += f"\n[System Note: User preferred style set to '{current_preferred_style}']"
        elif isinstance(response, Session5DataResponse):
            full_case["generated_session5_outcome"] = response.generated_session5_outcome.model_dump()
            current_agreed_task = response.generated_session5_outcome.agreed_task_for_session6
            history_for_llm += f"\n[System Note: Agreed task for next session is '{current_agreed_task}']"

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
    for i, seed_data in enumerate(tqdm(seed_dataset, desc="Generating Enriched Case Studies")):
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
        f"\nGeneration complete. Saved {len(final_dataset)} enriched cases to '{OUTPUT_FILENAME}'.")
