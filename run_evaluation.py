# evaluation.py

import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from typing import Dict, Any

import config
from llm_utils import call_structured_llm
from baseline_model import run_static_counselor
from adaptive_model import run_adaptive_counselor


class GrowthRubricScores(BaseModel):
    goal_congruence: int = Field(...,
                                 description="Score 1-5 for Goal-Oriented Congruence.")
    reflective_synthesis: int = Field(...,
                                      description="Score 1-5 for Reflective Synthesis.")
    observational_acuity: int = Field(...,
                                      description="Score 1-5 for Observational Acuity.")
    worldview_alignment: int = Field(...,
                                     description="Score 1-5 for Worldview Alignment.")
    therapeutic_adaptation: int = Field(...,
                                        description="Score 1-5 for Therapeutic Adaptation.")
    holding_environment: int = Field(
        ..., description="Score 1-5 for Holding Environment (Safety).")
    overall_score: float = Field(..., description="The average of all scores.")


def calculate_synthesis_accuracy(predicted_profile: Dict[str, Any], ground_truth_profile: Dict[str, Any]) -> float:
    """Calculates semantic similarity of the evolution notes from the new schema."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Adapt to new schema key: 'evolution_notes'
    gt_notes = ground_truth_profile.get('evolution_notes', '')
    pred_notes = predicted_profile.get('evolution_notes', '')

    if not gt_notes or not pred_notes:
        return 0.0

    embedding1 = model.encode(gt_notes, convert_to_tensor=True)
    embedding2 = model.encode(pred_notes, convert_to_tensor=True)

    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return max(0, cosine_score)


def run_growth_evaluation(case_data: Dict[str, Any], counselor_response: str) -> GrowthRubricScores:
    """Uses an LLM-as-Judge to score a response, using the new data structure."""
    prompt = f"""
    You are an expert evaluator of AI therapeutic systems. Your task is to score a counselor's response based on the G.R.O.W.T.H. rubric.

    **GROUND TRUTH CONTEXT:**
    - **Initial Profile:** {json.dumps(case_data['initial_profile'], indent=2)}
    - **EVOLVED Profile (The User's TRUE State):** {json.dumps(case_data['evolved_profile'], indent=2)}
    - **Full Transcript:** {json.dumps(case_data['sessions'], indent=2)}

    **COUNSELOR RESPONSE TO EVALUATE:**
    "{counselor_response}"

    **G.R.O.W.T.H. RUBRIC (Score each 1-5):**
    1.  **G - Goal-Oriented Congruence**: Does the response align with the user's EVOLVED goals?
    2.  **R - Reflective Synthesis**: Does it connect the user's past feelings to their present state, acknowledging the evolution described in 'evolution_notes'?
    3.  **O - Observational Acuity**: Does it notice subtle shifts in the user's self-perception?
    4.  **W - Worldview Alignment**: Does its approach align with the user's EVOLVED values and worldview? A response aligned only with the Initial Profile should score low.
    5.  **T - Therapeutic Adaptation**: Does the stance shift appropriately based on the user's evolution?
    6.  **H - Holding Environment**: Does it create a stable, non-judgmental space?

    **TASK:**
    Provide your scores in a valid JSON object. Calculate the average for the 'overall_score'.
    """
    scores = call_structured_llm(
        prompt, GrowthRubricScores, model=config.JUDGE_MODEL, temperature=0.1)
    if scores is None:
        return GrowthRubricScores(goal_congruence=0, reflective_synthesis=0, observational_acuity=0, worldview_alignment=0, therapeutic_adaptation=0, holding_environment=0, overall_score=0.0)
    return scores


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

        # 1. Run Baseline (Static Counselor)
        print("Running Baseline Model...")
        baseline_response = run_static_counselor(case)
        baseline_scores = run_growth_evaluation(case, baseline_response)

        # 2. Run Proposed (Adaptive Counselor)
        print("Running Adaptive Model...")
        adaptive_result = run_adaptive_counselor(case)
        adaptive_response = adaptive_result['response']
        predicted_profile = adaptive_result['predicted_profile']
        adaptive_scores = run_growth_evaluation(case, adaptive_response)

        # 3. Calculate Synthesis Accuracy
        synthesis_accuracy = 0.0
        if predicted_profile:
            synthesis_accuracy = calculate_synthesis_accuracy(
                predicted_profile, case['evolved_profile'])

        case_result = {
            "case_id": case_id,
            "synthesis_accuracy": synthesis_accuracy,
            "baseline": {"response": baseline_response, "scores": baseline_scores.model_dump()},
            "adaptive": {"response": adaptive_response, "predicted_profile_notes": predicted_profile.get('evolution_notes', '') if predicted_profile else "N/A", "scores": adaptive_scores.model_dump()},
            "ground_truth_notes": case['evolved_profile']['evolution_notes']
        }
        all_results.append(case_result)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

    # Final Report Generation
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

    rubric_keys = list(GrowthRubricScores.model_fields.keys())
    print("Detailed Average G.R.O.W.T.H. Score Breakdown:")
    for key in rubric_keys[:-1]:
        baseline_avg = np.mean([r['baseline']['scores'][key]
                               for r in all_results])
        adaptive_avg = np.mean([r['adaptive']['scores'][key]
                               for r in all_results])
        print(
            f"  - {key.replace('_', ' ').title():<25} | Baseline: {baseline_avg:.3f} | Adaptive: {adaptive_avg:.3f}")


if __name__ == "__main__":
    main()
