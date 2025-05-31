# run_evaluation.py

import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from typing import Dict, Any, Union

import config
from llm_utils import call_structured_llm
from baseline_model import run_static_counselor
from adaptive_model import run_adaptive_counselor


class Scores(BaseModel):
    task_alignment: int = Field(..., alias='T')
    alliance_bond: int = Field(..., alias='A')
    stylistic_congruence: int = Field(..., alias='S')
    congruence_with_goals: int = Field(..., alias='C')
    overall_score: float

    class Config:
        populate_by_name = True


class TascRubricScores(BaseModel):
    justification: Union[str, Dict[str, Any]]
    scores: Scores


def calculate_synthesis_accuracy(predicted_profile: Dict[str, Any], ground_truth_profile: Dict[str, Any]) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_notes_as_string(profile: Dict[str, Any]) -> str:
        notes = profile.get('evolution_notes', '')
        if isinstance(notes, dict):
            return json.dumps(notes)
        return str(notes)
    gt_notes_str = get_notes_as_string(ground_truth_profile)
    pred_notes_str = get_notes_as_string(predicted_profile)
    if not gt_notes_str or not pred_notes_str:
        return 0.0
    embedding1 = model.encode(gt_notes_str, convert_to_tensor=True)
    embedding2 = model.encode(pred_notes_str, convert_to_tensor=True)
    return max(0, util.pytorch_cos_sim(embedding1, embedding2).item())


def run_tasc_evaluation(case_data: Dict[str, Any], counselor_response: str) -> TascRubricScores:
    prompt = f"""
    You are a meticulous research assistant and clinical psychology expert. Your objective is to provide an unbiased, rigorous evaluation of an AI Counselor's response by strictly adhering to the detailed scoring rubric and output format provided below.

    **PART 1: GROUND TRUTH CONTEXT**
    - **Initial Profile:** {json.dumps(case_data['initial_profile'], indent=2)}
    - **EVOLVED Profile:** {json.dumps(case_data['evolved_profile'], indent=2)}
    - **Final User Turn:** "{case_data['sessions'][-1]['transcript'][-1]['content']}"

    **PART 2: AI COUNSELOR'S RESPONSE TO ANALYZE**
    "{counselor_response}"

    **PART 3: DETAILED SCORING RUBRIC**
    1.  **T - Task Alignment**: The user's `agreed_task` was "{case_data['evolved_profile']['agreed_task']}".
        - 5 (Excellent): Masterfully implements the task, weaving it seamlessly into the conversation.
        - 4 (Good): Correctly and clearly implements the task as requested.
        - 3 (Moderate): Attempts to implement the task, but the execution is flawed, incomplete, or slightly misaligned.
        - 2 (Weak): Barely mentions the task or alludes to it vaguely without any real implementation.
        - 1 (Poor): Completely ignores or contradicts the `agreed_task`.
    2.  **A - Alliance Bond**:
        - 5 (Excellent): Deeply empathetic; not only validates the user's feelings but also reflects their underlying meaning, significantly strengthening the alliance.
        - 4 (Good): Clearly validates the user's specific emotions and perspective, making them feel heard and understood.
        - 3 (Moderate): Shows basic empathy (e.g., "I understand," "That sounds hard") but lacks specificity.
        - 2 (Weak): The response is polite but emotionally detached or uses generic, unhelpful platitudes.
        - 1 (Poor): The response is robotic, dismissive, or invalidating.
    3.  **S - Stylistic Congruence**: The user's `preferred_style` was "{case_data['initial_profile']['preferred_style']}".
        - 5 (Excellent): Perfectly embodies the requested style, making the interaction feel naturally and precisely tailored.
        - 4 (Good): Consistently and clearly matches the requested style.
        - 3 (Moderate): Attempts to match the style but does so inconsistently or awkwardly.
        - 2 (Weak): The style is generic and does not align with the user's preference.
        - 1 (Poor): The style is the opposite of what the user requested (e.g., highly directive when reflective was asked for).
    4.  **C - Congruence with Goals**: The user's evolved `goals` are {case_data['evolved_profile']['goals']}.
        - 5 (Excellent): Masterfully connects the immediate task and conversation to the user's broader, evolved goals, providing insight.
        - 4 (Good): The response clearly helps the user move toward one or more of their stated evolved goals.
        - 3 (Moderate): The response is generally helpful but does not explicitly connect to the evolved goals.
        - 2 (Weak): The connection to the user's goals is tangential or very weak.
        - 1 (Poor): The response is irrelevant or counter-productive to the user's evolved goals.

    **PART 4: YOUR TASK & OUTPUT FORMAT**
    You must provide your analysis in a single JSON object with two top-level keys: "justification" and "scores".
    1.  `justification`: A string or a dictionary explaining your reasoning for the scores.
    2.  `scores`: A NESTED JSON object containing your scores for "T", "A", "S", "C", and the calculated "overall_score" (the average of the four scores).
    """
    scores = call_structured_llm(
        prompt, TascRubricScores, model=config.JUDGE_MODEL, temperature=0.0)
    if scores is None:
        default_scores = Scores(T=1, A=1, S=1, C=1, overall_score=1.0)
        return TascRubricScores(justification="Evaluation failed due to a model or parsing error.", scores=default_scores)
    return scores


def main():
    DATASET_PATH = os.path.join("output", "counseling_dataset.json")
    RESULTS_PATH = os.path.join("output", "evaluation_results.json")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found. Please run `generate_dataset.py` first.")
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
        baseline_scores = run_tasc_evaluation(case, baseline_response)

        print("Running Adaptive Model...")
        adaptive_result = run_adaptive_counselor(case)
        adaptive_response = adaptive_result['response']
        predicted_profile = adaptive_result['predicted_profile']
        adaptive_scores = run_tasc_evaluation(case, adaptive_response)

        synthesis_accuracy = 0.0
        if predicted_profile:
            synthesis_accuracy = calculate_synthesis_accuracy(
                predicted_profile, case['evolved_profile'])

        # --- FIX: Use by_alias=True when dumping the model to a dictionary ---
        # This ensures the dictionary keys are 'T', 'A', 'S', 'C', matching what the reporting loop expects.
        case_result = {
            "case_id": case_id, "synthesis_accuracy": synthesis_accuracy,
            "baseline": {"response": baseline_response, "scores": baseline_scores.model_dump(by_alias=True)},
            "adaptive": {"response": adaptive_response, "predicted_profile": predicted_profile, "scores": adaptive_scores.model_dump(by_alias=True)},
        }
        all_results.append(case_result)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed results saved to {RESULTS_PATH}")
    print("\n\n--- FINAL EVALUATION REPORT ---")
    if not all_results:
        return
    avg_synthesis_accuracy = np.mean(
        [r['synthesis_accuracy'] for r in all_results])

    baseline_overall = np.mean(
        [r['baseline']['scores']['scores']['overall_score'] for r in all_results])
    adaptive_overall = np.mean(
        [r['adaptive']['scores']['scores']['overall_score'] for r in all_results])

    print(f"\nAverage Synthesis Accuracy: {avg_synthesis_accuracy:.3f}")
    print("-" * 30)
    print("Average Overall T.A.S.C. Scores:")
    print(f"  - Baseline (Static) Counselor: {baseline_overall:.3f}")
    print(f"  - Adaptive Counselor:          {adaptive_overall:.3f}")
    print("-" * 30)

    rubric_aliases = [field.alias for field in Scores.model_fields.values(
    ) if field.alias and field.alias in ["T", "A", "S", "C"]]
    print("Detailed Average T.A.S.C. Score Breakdown:")
    for alias in rubric_aliases:
        # This loop will now work correctly because the dictionary keys match the aliases.
        baseline_avg = np.mean(
            [r['baseline']['scores']['scores'][alias] for r in all_results])
        adaptive_avg = np.mean(
            [r['adaptive']['scores']['scores'][alias] for r in all_results])
        field_name = next(
            name for name, field in Scores.model_fields.items() if field.alias == alias)
        print(f"  - {field_name.replace('_', ' ').title():<30} | Baseline: {baseline_avg:.3f} | Adaptive: {adaptive_avg:.3f}")


if __name__ == "__main__":
    main()
