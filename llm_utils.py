# llm_utils.py

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, Optional, Union

import config

if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
    raise ValueError(
        "OpenRouter API key is not set. Please set it in config.py or as an environment variable."
    )

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY,
)
print("OpenAI client configured for OpenRouter for all LLM calls.")

T = TypeVar('T', bound=BaseModel)


def call_api_client(
    prompt: str,
    response_model: Type[T],
    model: str,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    expect_json_object: bool = True
) -> Optional[Union[T, str]]:
    """
    Calls a model via the OpenRouter API.
    If expect_json_object is True, parses the response into a Pydantic model.
    If expect_json_object is False, returns the raw string response.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if expect_json_object:
        request_params["response_format"] = {"type": "json_object"}

    try:
        response = openrouter_client.chat.completions.create(**request_params)
        response_content = response.choices[0].message.content

        if expect_json_object:
            return response_model.model_validate_json(response_content)
        else:
            # For raw text, we still want to return something that can be assigned
            # to where a Pydantic model was expected, if the caller handles it.
            # Or the caller can instantiate the Pydantic model itself.
            # For CounselorResponse, it just has one field "response".
            return response_content  # Return raw string

    except ValidationError as e:
        print(f"--- Pydantic Validation Error ({model}) --- \n{e}")
        print(
            f"Model Output: {response_content if 'response_content' in locals() else 'N/A'}")
        return None  # Keep returning None on Pydantic error
    except Exception as e:
        print(f"--- API Client Error ({model}) --- \n{e}")
        # For simplicity, if we expected JSON and failed, or any other API error, return None.
        # If we didn't expect JSON and an API error occurs, also None.
        return None
