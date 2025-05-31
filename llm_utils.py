# llm_utils.py

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Optional

import config

# --- OpenRouter Client Initialization ---
# Check for the API key
if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError(
        "OpenRouter API key is not set. Please set it in config.py or as an environment variable.")

# Instantiate the client to point to the OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
)
print("OpenAI client configured for OpenRouter.")
# -----------------------------------------


# Pydantic model type variable for generic responses
T = TypeVar('T', bound=BaseModel)


def call_structured_llm(
    prompt: str,
    response_model: Type[T],
    model: str = config.MAIN_MODEL,
    temperature: float = 0.7
) -> Optional[T]:
    """
    Calls a large language model via OpenRouter and parses the response into a Pydantic model.

    Args:
        prompt (str): The prompt to send to the model.
        response_model (Type[T]): The Pydantic class to validate the response against.
        model (str): The model name to use for the call (must be an OpenRouter identifier).
        temperature (float): The creativity of the response.

    Returns:
        Optional[T]: An instance of the response_model or None if an error occurs.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        response_content = response.choices[0].message.content

        # Validate the JSON output against the Pydantic model
        validated_response = response_model.model_validate_json(
            response_content)
        return validated_response

    except ValidationError as e:
        print(f"--- Pydantic Validation Error ---")
        print(f"Error: {e}")
        print(f"Model Output: {response_content}")
        return None
    except json.JSONDecodeError as e:
        print(f"--- JSON Decode Error ---")
        print(f"Error: {e}")
        print(f"Model Output: {response_content}")
        return None
    except Exception as e:
        print(f"--- An unexpected error occurred via OpenRouter ---")
        print(f"Error: {e}")
        return None
