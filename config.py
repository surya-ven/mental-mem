# config.py

import os

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")

# --- Model Configuration for OpenRouter ---
# Models used for generating the dataset and running the counselors.
MAIN_MODEL = "openai/gpt-4.1-mini"

# Model used for the evaluation rubric.
JUDGE_MODEL = "openai/gpt-4.1-mini"
