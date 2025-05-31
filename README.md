# Project: Longitudinal Profile Synthesis for Adaptive AI Counseling

This project implements and evaluates an AI counselor that adapts to a user's personal evolution over time. It compares a **Static Counselor** (Baseline) against an **Adaptive Counselor** (Proposed Method) that uses a "Profile Synthesis" step to update its understanding of the user.

## How It Works

The core idea is to test if an AI can detect shifts in a user's values and goals across multiple sessions and use that understanding to provide better-aligned responses.

1.  **Dataset Generation**: A synthetic dataset of multi-session therapy cases is created. Each case includes an `initial_profile`, an `evolved_profile` after a key insight, and a final `test_probe`.
2.  **Baseline Model (Static Counselor)**: This model uses _only_ the user's `initial_profile` to inform its personality and response.
3.  **Proposed Model (Adaptive Counselor)**: This model first performs **Profile Synthesis**. It analyzes the conversation history to create a _predicted_ `evolved_profile`. It then uses this new, updated profile to inform its response.
4.  **Evaluation**: We use a two-pronged approach:
    -   **Synthesis Accuracy**: We quantitatively measure how accurately the Adaptive Counselor's predicted profile matches the ground-truth evolved profile from the dataset.
    -   **G.R.O.W.T.H. Rubric**: An LLM-as-Judge scores the final responses from both models on therapeutic quality, measuring alignment with the user's _actual_ evolved state.

## Setup

1.  **Prerequisites**: Make sure you have Python 3.9+ installed.
2.  **Create Project Structure**: Create the folders and files as described in the project documentation.
3.  **API Key**: Open `config.py` and enter your OpenAI API key.
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Seed Data**: Place your seed data file in the `seed_data/` directory and name it `counseling_data_seed.json`. It should be a JSON array of objects, where each object has a `"questionText"` field containing the initial user problem.

## How to Run

Execute the scripts in order.

**Step 1: Generate the Dataset**
This script uses the seed data to generate the multi-session case studies.

```bash
python generate_dataset.py
```

This will create `counseling_dataset.json` in the `output/` folder.

**Step 2: Run the Full Evaluation**
This script will run both the baseline and adaptive models on the generated dataset and then evaluate their performance.

```bash
python run_evaluation.py
```

The script will print a detailed report to the console and save the full results to `evaluation_results.json` in the `output/` folder.

## Note on `mem0`

The attached documentation for `mem0` describes a valuable memory layer for AI. In a production version of this system, a tool like `mem0` would be ideal for storing and retrieving the user profiles and conversation transcripts discussed here. For this specific research experiment, we focus on the higher-level **synthesis logic** itself, using local JSON files as our data store to clearly isolate and evaluate the core contribution.
