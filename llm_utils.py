# llm_utils.py

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Optional

import config

# --- Client for OpenRouter (All Models) ---
if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
    raise ValueError(
        "OpenRouter API key is not set. Please set it in config.py or as an environment variable."
    )

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
)
print("OpenAI client configured for OpenRouter for all LLM calls.")

# Pydantic model type variable for generic responses
T = TypeVar('T', bound=BaseModel)


def call_api_client(prompt: str, response_model: Type[T], model: str, temperature: float = 0.7, system_prompt: Optional[str] = None) -> Optional[T]:
    """
    Calls a model via the OpenRouter API and parses the response into a Pydantic model.
    Supports an optional system prompt.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = openrouter_client.chat.completions.create(
            model=model,
            messages=messages,
            # Assuming model supports JSON mode via OpenRouter
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        response_content = response.choices[0].message.content
        return response_model.model_validate_json(response_content)
    except ValidationError as e:
        print(f"--- Pydantic Validation Error ({model}) --- \n{e}")
        print(
            f"Model Output: {response_content if 'response_content' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"--- API Client Error ({model}) --- \n{e}")
        return None
