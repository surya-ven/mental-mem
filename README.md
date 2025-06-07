# Project: Longitudinal Profile Synthesis for Adaptive AI Counseling

This project implements and evaluates AI counselors that adapt to a user's personal evolution over multiple simulated therapy sessions. It compares various counselor models, including those that use a "Profile Synthesis" step (via a Reflector module) to update their understanding of the user against baseline and no-memory versions.

## How It Works

The core process involves generating a synthetic dataset, running different counselor models, and evaluating their responses.

1.  **Dataset Generation**:

    -   The `generate_dataset.py` script creates multi-session therapy case studies.
    -   It uses seed problem statements from `seed_data/counseling_data_seed.json`.
    -   Each generated case includes session transcripts and details about the user's evolving state.
    -   The output dataset (e.g., `counseling_dataset_*.json`) is saved in the `output/` directory.

2.  **Counselor Models & Configuration**:

    -   `counselor_models.py` defines the various AI counselors, such as:
        -   **Adaptive Counselors**: Utilize a "Reflector" (`run_srs_reflector`) to perform profile synthesis, creating a `ClinicalSummary` to adapt to the user over sessions.
        -   **Baseline Counselors**: May use RAG from initial information but do not dynamically update the user profile in the same way.
        -   **No-Memory Counselors**: Serve as a control.
    -   Models can be run using "local" open-source LLMs or "closed" proprietary LLMs.
    -   Key model choices are configured in `config.py`, specifically:
        -   `LOCAL_MODEL_NAME`: For open-source counselor models.
        -   `CLOSED_MODEL_NAME`: For proprietary counselor models.
        -   `REFLECTOR_MODEL`: Used for the profile synthesis step.
        -   `JUDGE_MODEL`: Used in the evaluation phase.

3.  **Evaluation**:
    -   The `evaluation.py` script runs the full evaluation pipeline.
    -   It tests the different counselor models against the generated dataset.
    -   An LLM-as-Judge (using `JUDGE_MODEL`) scores counselor responses based on:
        -   **T.A.S.C.S. Rubric**: Assessing Task Alignment, Alliance Bond, Stylistic Congruence, and Congruence With Goals.
        -   **Safety Scores**: Evaluating the safety of the model's output and its handling of sensitive situations.
    -   Detailed evaluation results (e.g., `evaluation_results_*.json`) are saved in the `output/` directory.
    -   Historical evaluation runs and their detailed logs can be found in the `results_archive/` directory for inspection.

## Setup

1.  **Prerequisites**:

    -   Python 3.9+
    -   Access to the LLM APIs used in `config.py`.

2.  **API Keys**:

    -   Open `config.py` and set your `OPENROUTER_API_KEY` and `OPENAI_API_KEY` (and any other relevant API keys for the models you intend to use).

3.  **Install Dependencies**:

    ```bash
    uv sync
    ```

4.  **Seed Data**:
    -   Ensure your seed data file, `counseling_data_seed.json`, is located in the `seed_data/` directory. This file should contain initial user problem statements used by `generate_dataset.py`.
    -   The `source_data/` folder contains other raw data that may have been used to prepare the seed data.

## How to Run

Execute the scripts in the following order:

**Step 1: Generate the Dataset**
This script uses the seed data to generate multi-session case studies.

```bash
python generate_dataset.py
```

This will create `counseling_dataset_*.json` (or similar) in the `output/` folder.

**Step 2: Run the Full Evaluation**
This script runs the different counselor models on the generated dataset and evaluates their performance.

```bash
python evaluation.py
```

The script will output a summary to the console and save detailed results (e.g., `evaluation_results_*.json`) in the `output/` folder. Check `output/run_logs/` for detailed logs of each case and model.
