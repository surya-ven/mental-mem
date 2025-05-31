# config.py

import os

# --- Essential Configuration ---
# Place your OpenRouter API key here.
# It is recommended to use an environment variable for security.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")

# --- Model Configuration for OpenRouter ---
# Models used for generating the dataset and running the counselors.
# You can find model identifiers on the OpenRouter documentation.
# We recommend a powerful model like gpt-4o for quality.
# Example: "openai/gpt-4o", "google/gemini-pro-1.5", "anthropic/claude-3-opus"
MAIN_MODEL = "openai/gpt-4o-mini"

# Model used for the evaluation rubric.
# A powerful model is crucial for nuanced, high-quality evaluation.
JUDGE_MODEL = "openai/gpt-4o-mini"
