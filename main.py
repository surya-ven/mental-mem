import os
import json
from typing import Dict, Any, List, Optional

# The OpenAI SDK is used for its native support for structured output via Pydantic
from openai import OpenAI, APIError
from pydantic import BaseModel, Field, ValidationError

# --- Pydantic Schemas for Structured Output ---
# This defines the structure of each turn in the conversation.


class DialogueTurn(BaseModel):
    role: str = Field(...,
                      description="The speaker's role, either 'counselor' or 'user'.")
    content: str = Field(...,
                         description="The verbatim content of the dialogue turn.")

# This defines the schema for the user's initial profile.


class InitialProfile(BaseModel):
    core_values: List[str] = Field(...,
                                   description="A list of the user's core values.")
    preferred_modality: str = Field(
        ..., description="The user's preferred therapeutic modality (e.g., 'Cognitive-Behavioral').")
    communication_style: str = Field(
        ..., description="The user's preferred communication style (e.g., 'Casual', 'Motivational').")
    initial_goals: List[str] = Field(
        ..., description="A list of the user's initial therapeutic goals.")

# This defines the schema for the user's evolved profile, including notes on what changed.


class EvolvedProfile(InitialProfile):
    value_evolution_notes: str = Field(
        ..., description="A summary of how and why the user's values or goals have shifted since the initial session.")

# This is the main response model for the LLM call for Sessions 1 and 2.


class GenerationOutput(BaseModel):
    transcript: List[DialogueTurn] = Field(
        ..., description="The full, multi-turn transcript of the session.")
    profile: Dict[str, Any] = Field(
        ..., description="The user's therapeutic profile, matching either the InitialProfile or EvolvedProfile schema.")


# --- Configuration ---
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("ERROR: The OPENROUTER_API_KEY environment variable is not set.")
    exit()

# Note: Changed model name based on user's code, but this might need updating
# to a model that supports JSON mode well on OpenRouter.
# e.g., "openai/gpt-4o" or "google/gemini-1.5-pro-latest"
GENERATION_MODEL = "openai/gpt-4.1-nano"
SEED_DATA_FILENAME = "counseling_data_seed.json"
OUTPUT_FILENAME = "generated_counseling_dataset.json"


# --- Updated Prompts for Structured Output ---
PROMPT_SESSION_1 = """
You are a creative writer and clinical psychologist. Your task is to generate a realistic initial therapy session based on a user's core problem.

A new user presents the following core problem:
---
{seed_text}
---

Based on this, generate a **4-6 turn** conversation for "Session 1". In this session, the AI Counselor must guide the user through a "Values & Modality Discovery" to determine their initial profile.
Then, formulate the 'Initial Therapeutic Profile' based on the conversation you generated.

Your entire output must be a single, valid JSON object that strictly adheres to the structure provided below. Do not include any other text, markdown, or explanations.

**JSON OUTPUT STRUCTURE:**
```json
{{
  "transcript": [
    {{
      "role": "counselor",
      "content": "Verbatim content of the counselor's turn."
    }},
    {{
      "role": "user",
      "content": "Verbatim content of the user's turn."
    }}
  ],
  "profile": {{
    "core_values": ["A list of strings representing the user's core values."],
    "preferred_modality": "A string for the user's preferred therapeutic modality.",
    "communication_style": "A string for the user's preferred communication style.",
    "initial_goals": ["A list of strings for the user's initial therapeutic goals."]
  }}
}}
"""

PROMPT_SESSION_2 = """
You are a creative writer and clinical psychologist continuing a client's story.

Context from Session 1:
The user's initial profile is: {initial_profile_str}
The transcript was: {session_1_transcript_str}

Your Task:
Generate a 4-6 turn conversation for "Session 2". In this session, the user must have a significant insight or experience that evolves their initial profile. Capture this "aha!" moment.
Then, formulate the 'Evolved Therapeutic Profile', making sure to detail the changes in the 'value_evolution_notes' field.

Your entire output must be a single, valid JSON object that strictly adheres to the structure provided below. Do not include any other text, markdown, or explanations.

JSON OUTPUT STRUCTURE:

JSON
{{
  "transcript": [
    {{
      "role": "counselor",
      "content": "Verbatim content of the counselor's turn."
    }},
    {{
      "role": "user",
      "content": "Verbatim content of the user's turn."
    }}
  ],
  "profile": {{
    "core_values": ["An updated list of the user's core values."],
    "preferred_modality": "The user's therapeutic modality.",
    "communication_style": "The user's communication style.",
    "initial_goals": ["An updated list of the user's therapeutic goals."],
    "value_evolution_notes": "A detailed summary of how and why the user's values or goals have shifted since the initial session."
  }}
}}
"""

PROMPT_SESSION_3 = """
You are a creative writer and clinical psychologist concluding a client's story.

Context - User's Evolved Profile:
{evolved_profile_str}

Your Task:
Generate a short, 2-3 turn conversation for "Session 3". The user should present a new, ambiguous problem. This problem should be a test case where a correct response requires understanding their evolved profile, not their initial one. The final user turn is the "Test Probe".

Your entire output must be a single, valid JSON object that strictly adheres to the structure provided below. Do not include a profile. Do not include any other text, markdown, or explanations.

JSON OUTPUT STRUCTURE:

JSON
{{
  "transcript": [
    {{
      "role": "counselor",
      "content": "Verbatim content of the counselor's turn."
    }},
    {{
      "role": "user",
      "content": "The final user turn, which acts as the test probe."
    }}
  ]
}}
"""


def call_structured_generative_model(client: OpenAI, prompt: str, response_schema: BaseModel) -> Optional[BaseModel]:
    """Calls the generative model requesting JSON output, then parses and validates it."""
    messages = [{"role": "user", "content": prompt}]
    try:
        # Step 1: Call the API with JSON mode enabled
        completion = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            # Standard way to request JSON output
            response_format={"type": "json_object"}
        )

        json_output = completion.choices[0].message.content

        # Step 2: Parse the JSON string from the response
        parsed_data = json.loads(json_output)

        # Step 3: Validate the parsed data against the Pydantic schema
        validated_model = response_schema.model_validate(parsed_data)

        return validated_model

    except APIError as e:
        print(f"An API error occurred: {e}. Skipping this generation.")
        return None
    except json.JSONDecodeError as e:
        print(
            f"Failed to parse JSON from model output: {e}. Output was:\n{json_output}")
        return None
    except ValidationError as e:
        print(f"Pydantic validation failed: {e}. Output was:\n{json_output}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Skipping this generation.")
        return None


def generate_case_study(client: OpenAI, case_id: str, seed_text: str) -> Optional[Dict[str, Any]]:
    """Generates a full multi-session case study using structured output calls."""
    print(f"--- Generating Case Study {case_id} ---")

    # Define the response models for each step
    class Session1Response(BaseModel):
        transcript: List[DialogueTurn]
        profile: InitialProfile

    class Session2Response(BaseModel):
        transcript: List[DialogueTurn]
        profile: EvolvedProfile

    class Session3Response(BaseModel):
        transcript: List[DialogueTurn]

    # Step 1: Generating Session 1
    print("Step 1: Generating Onboarding Session...")
    prompt1 = PROMPT_SESSION_1.format(seed_text=seed_text)
    output1 = call_structured_generative_model(
        client, prompt1, Session1Response)
    if not output1:
        return None

    transcript1 = output1.transcript
    initial_profile = output1.profile

    # Step 2: Generating Session 2
    print("Step 2: Generating Evolution Session...")
    prompt2 = PROMPT_SESSION_2.format(
        initial_profile_str=initial_profile.model_dump_json(indent=2),
        session_1_transcript_str=json.dumps(
            [t.model_dump() for t in transcript1])
    )
    output2 = call_structured_generative_model(
        client, prompt2, Session2Response)
    if not output2:
        return None

    transcript2 = output2.transcript
    evolved_profile = output2.profile

    # Step 3: Generating Session 3
    print("Step 3: Generating Test Probe Session...")
    prompt3 = PROMPT_SESSION_3.format(
        evolved_profile_str=evolved_profile.model_dump_json(indent=2))
    output3 = call_structured_generative_model(
        client, prompt3, Session3Response)
    if not output3:
        return None

    transcript3 = output3.transcript

    # Assemble the final case study using .model_dump() to convert Pydantic objects to dicts
    case_study = {
        "case_id": case_id,
        "seed_text": seed_text,
        "initial_profile": initial_profile.model_dump(),
        "evolved_profile": evolved_profile.model_dump(),
        "sessions": [
            {"session_number": 1, "summary": "Onboarding session.",
                "transcript": [t.model_dump() for t in transcript1]},
            {"session_number": 2, "summary": "Evolution session.",
                "transcript": [t.model_dump() for t in transcript2]},
            {"session_number": 3, "summary": "Test probe session.",
                "transcript": [t.model_dump() for t in transcript3]}
        ]
    }
    print(f"--- Successfully Generated Case Study {case_id} ---")
    return case_study


if __name__ == "__main__":
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
        print("OpenAI client for OpenRouter initialized successfully.")
    except Exception as e:
        print(f"FATAL: Failed to initialize OpenAI Client: {e}")
        exit()

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

    for i, seed_data in enumerate(seed_dataset):
        case_id = f"case_{i + 1:03d}"
        seed_text = seed_data.get("questionText")
        if not seed_text:
            print(f"Warning: Seed #{i+1} is missing 'questionText'. Skipping.")
            continue

        case = generate_case_study(client, case_id, seed_text)

        if case:
            final_dataset.append(case)

    try:
        with open(OUTPUT_FILENAME, 'w') as f:
            json.dump(final_dataset, f, indent=4)
        print(
            f"\nGeneration complete. Saved {len(final_dataset)} case studies to '{OUTPUT_FILENAME}'.")
    except IOError as e:
        print(
            f"FATAL: Could not write to output file '{OUTPUT_FILENAME}': {e}")
