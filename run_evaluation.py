# evaluation.py

import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
# Import Field and Union for more flexible schemas
from pydantic import BaseModel, Field
from typing import Dict, Any, Union

import config
from llm_utils import call_structured_llm
from baseline_model import run_static_counselor
from adaptive_model import run_adaptive_counselor

# --- FIX 1: Use Field aliases to accept abbreviated OR full key names ---


class GrowthRubricScores(BaseModel):
    goal_congruence: int = Field(..., alias='G',
                                 description="Score 1-5 for Goal-Oriented Congruence.")
    reflective_synthesis: int = Field(..., alias='R',
                                      description="Score 1-5 for Reflective Synthesis.")
    observational_acuity: int = Field(..., alias='O',
                                      description="Score 1-5 for Observational Acuity.")
    worldview_alignment: int = Field(..., alias='W',
                                     description="Score 1-5 for Worldview Alignment.")
    therapeutic_adaptation: int = Field(..., alias='T',
                                        description="Score 1-5 for Therapeutic Adaptation.")
    holding_environment: int = Field(..., alias='H',
                                     description="Score 1-5 for Holding Environment (Safety).")
    overall_score: float

    class Config:
        # Allows the model to be populated by either the field name or its alias
        populate_by_name = True


def calculate_synthesis_accuracy(predicted_profile: Dict[str, Any], ground_truth_profile: Dict[str, Any]) -> float:
    """Calculates semantic similarity, now robust to object-or-string types."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- FIX 2: Handle cases where evolution_notes is an object instead of a string ---
    def get_notes_as_string(profile: Dict[str, Any]) -> str:
        notes = profile.get('evolution_notes', '')
        if isinstance(notes, dict):
            # If it's a dict, serialize it to a string to capture all info
            return json.dumps(notes)
        return str(notes)  # Otherwise, treat as a string

    gt_notes_str = get_notes_as_string(ground_truth_profile)
    pred_notes_str = get_notes_as_string(predicted_profile)

    if not gt_notes_str or not pred_notes_str:
        return 0.0

    embedding1 = model.encode(gt_notes_str, convert_to_tensor=True)
    embedding2 = model.encode(pred_notes_str, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return max(0, cosine_score)


def run_growth_evaluation(case_data: Dict[str, Any], counselor_response: str) -> GrowthRubricScores:
    """Uses an LLM-as-Judge to score a response. The prompt is updated to encourage abbreviated keys."""
    prompt = f"""
    You are an expert evaluator of AI therapeutic systems. Your task is to score a counselor's response based on the G.R.O.W.T.H. rubric.

    **GROUND TRUTH CONTEXT:**
    - **Initial Profile:** {json.dumps(case_data['initial_profile'], indent=2)}
    - **EVOLVED Profile (The User's TRUE State):** {json.dumps(case_data['evolved_profile'], indent=2)}
    - **Full Transcript:** {json.dumps(case_data['sessions'], indent=2)}

    **COUNSELOR RESPONSE TO EVALUATE:**
    "{counselor_response}"

    **G.R.O.W.T.H. RUBRIC (Score each 1-5):**
    1.  **G - Goal-Oriented Congruence**: Aligns with EVOLVED goals.
    2.  **R - Reflective Synthesis**: Acknowledges the evolution in 'evolution_notes'.
    3.  **O - Observational Acuity**: Notices subtle shifts.
    4.  **W - Worldview Alignment**: Aligns with EVOLVED worldview.
    5.  **T - Therapeutic Adaptation**: Stance shifts appropriately.
    6.  **H - Holding Environment**: Creates a safe space.

    **TASK:**
    Provide your scores in a valid JSON object. Use the single-letter abbreviations (G, R, O, W, T, H) as keys. Calculate the average for the 'overall_score'.
    """
    scores = call_structured_llm(
        prompt, GrowthRubricScores, model=config.JUDGE_MODEL, temperature=0.0)
    if scores is None:
        return GrowthRubricScores(G=0, R=0, O=0, W=0, T=0, H=0, overall_score=0.0)
    return scores

# The main() function remains the same as before.


def main():
    """Main function to run the entire evaluation pipeline."""
    DATASET_PATH = os.path.join("output", "counseling_dataset.json")
    RESULTS_PATH = os.path.join("output", "evaluation_results.json")

    if not os.path.exists(DATASET_PATH):
        print(
            f"Dataset not found at {DATASET_PATH}. Please run `generate_dataset.py` first.")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    all_results = []
    print(f"Starting evaluation for {len(dataset)} cases...")

    for case in tqdm(dataset, desc="Evaluating Models"):
        case_id = case['case_id']
        print(f"\n--- Processing Case: {case_id} ---")

        print("Running Baseline Model...")
        baseline_response = run_static_counselor(case)
        baseline_scores = run_growth_evaluation(case, baseline_response)

        print("Running Adaptive Model...")
        adaptive_result = run_adaptive_counselor(case)
        adaptive_response = adaptive_result['response']
        predicted_profile = adaptive_result['predicted_profile']
        adaptive_scores = run_growth_evaluation(case, adaptive_response)

        synthesis_accuracy = 0.0
        if predicted_profile:
            synthesis_accuracy = calculate_synthesis_accuracy(
                predicted_profile, case['evolved_profile'])

        case_result = {
            "case_id": case_id,
            "synthesis_accuracy": synthesis_accuracy,
            "baseline": {"response": baseline_response, "scores": baseline_scores.model_dump(by_alias=True)},
            "adaptive": {"response": adaptive_response, "predicted_profile_notes": predicted_profile.get('evolution_notes', '') if predicted_profile else "N/A", "scores": adaptive_scores.model_dump(by_alias=True)},
            "ground_truth_notes": case['evolved_profile']['evolution_notes']
        }
        all_results.append(case_result)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

    print("\n\n--- FINAL EVALUATION REPORT ---")
    if not all_results:
        print("No results to report.")
        return

    avg_synthesis_accuracy = np.mean(
        [r['synthesis_accuracy'] for r in all_results])
    baseline_overall = np.mean(
        [r['baseline']['scores']['overall_score'] for r in all_results])
    adaptive_overall = np.mean(
        [r['adaptive']['scores']['overall_score'] for r in all_results])

    print(f"\nAverage Synthesis Accuracy: {avg_synthesis_accuracy:.3f}")
    print("-" * 30)
    print("Average Overall G.R.O.W.T.H. Scores:")
    print(f"  - Baseline (Static) Counselor: {baseline_overall:.3f}")
    print(f"  - Adaptive Counselor:          {adaptive_overall:.3f}")
    print("-" * 30)

    # Use the aliases for the report headers
    rubric_aliases = [
        field.alias for field in GrowthRubricScores.model_fields.values() if field.alias]
    print("Detailed Average G.R.O.W.T.H. Score Breakdown:")
    for alias in rubric_aliases:
        baseline_avg = np.mean([r['baseline']['scores'][alias]
                               for r in all_results])
        adaptive_avg = np.mean([r['adaptive']['scores'][alias]
                               for r in all_results])
        field_name = next(name for name, field in GrowthRubricScores.model_fields.items(
        ) if field.alias == alias)
        print(f"  - {field_name.replace('_', ' ').title():<25} | Baseline: {baseline_avg:.3f} | Adaptive: {adaptive_avg:.3f}")


if __name__ == "__main__":
    main()
