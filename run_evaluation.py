# run_evaluation.py

import json
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional

import config
from llm_utils import call_api_client
from counselor_models import (
    run_srs_reflector, MemoryManager,
    LocalAdaptiveCounselor, LocalBaselineCounselor, LocalBaselineNoMemoryCounselor,
    ClosedBaselineCounselor, ClosedBaselineNoMemoryCounselor, ClinicalSummary
)

# --- Define the nested Scores model first ---


class Scores(BaseModel):
    # Aliases are used for parsing input from the LLM judge's JSON
    # and also for dumping if model_dump(by_alias=True) is used.
    # Pydantic V2: Field defines validation, serialization, and JSON schema.
    task_alignment: int = Field(alias='T')
    alliance_bond: int = Field(alias='A')
    stylistic_congruence: int = Field(alias='S')
    congruence_with_goals: int = Field(alias='C')
    overall_score: float  # This does not have a different alias

    class Config:
        populate_by_name = True  # Allows initialization using field name OR alias

# --- T.A.S.C. Evaluation Rubric Schema ---


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


def run_tasc_evaluation(case_data: Dict, counselor_response: str) -> TascRubricScores:
    final_session_plan = "Not available"
    if len(case_data['sessions']) >= 5 and case_data['sessions'][4].get('summary'):
        summary_data = case_data['sessions'][4]['summary']
        if isinstance(summary_data, dict):
            final_session_plan = summary_data.get(
                'plan_for_next_session', "Not available")
        elif isinstance(summary_data, ClinicalSummary):
            final_session_plan = summary_data.plan_for_next_session

    prompt = f"""
    You are an expert evaluator of AI therapeutic systems. Score a response using the T.A.S.C. rubric.

    **GROUND TRUTH CONTEXT:**
    - **Final Session Plan (from end of Session 5):** "{final_session_plan}"
    - **Final User Turn (The Test Probe from Session 6):** "{case_data['sessions'][-1]['transcript'][-1]['content']}"

    **COUNSELOR RESPONSE TO EVALUATE:**
    "{counselor_response}"

    **T.A.S.C. RUBRIC (Score each 1-5):**
    1.  **T - Task Alignment**: The user's agreed plan was "{final_session_plan}".
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
    3.  **S - Stylistic Congruence**: (Assuming general supportive style for this example if not in dataset)
        - 5 (Excellent): Perfectly embodies a supportive, professional counseling style.
        - 4 (Good): Consistently and clearly matches a supportive style.
        - 3 (Moderate): Attempts the style but may be inconsistent or awkward.
        - 2 (Weak): The style is generic and lacks therapeutic quality.
        - 1 (Poor): The style is inappropriate or counter-therapeutic.
    4.  **C - Congruence with Goals**: (Assuming general therapeutic goals if not explicitly tracked in dataset summaries for judge)
        - 5 (Excellent): Masterfully connects the task to implicit therapeutic goals like self-understanding or emotional regulation.
        - 4 (Good): The response clearly helps the user move toward general therapeutic progress.
        - 3 (Moderate): The response is generally helpful but not explicitly goal-directed.
        - 2 (Weak): The connection to therapeutic goals is tangential.
        - 1 (Poor): The response is irrelevant or counter-productive.

    **TASK:**
    Provide your analysis in a single JSON object with two top-level keys: "justification" and "scores".
    1.  `justification`: A string or a dictionary explaining your reasoning for the scores.
    2.  `scores`: A NESTED JSON object containing your scores for "T", "A", "S", "C", and the calculated "overall_score" (the average of the four scores). Ensure the keys in this nested object are "T", "A", "S", "C", and "overall_score".
    """

    # This is the Pydantic model we expect the LLM Judge to return directly
    # It contains the nested 'Scores' model, which uses aliases.
    class ExpectedJudgeResponse(BaseModel):
        justification: Union[str, Dict[str, Any]]
        scores: Scores

    judge_response_obj = call_api_client(
        prompt, ExpectedJudgeResponse, model=config.JUDGE_MODEL, temperature=0.0)

    if not judge_response_obj:
        default_scores_sub = Scores(T=1, A=1, S=1, C=1, overall_score=1.0)
        return TascRubricScores(justification="Eval failed", scores=default_scores_sub)

    return TascRubricScores(justification=judge_response_obj.justification, scores=judge_response_obj.scores)


def main():
    DATASET_PATH = os.path.join("output", "counseling_dataset_6sessions.json")
    RESULTS_PATH = os.path.join(
        "output", "evaluation_results_full_cohort.json")
    if not os.path.exists(DATASET_PATH):
        print(
            f"Dataset not found at {DATASET_PATH}. Please run `generate_dataset.py` first.")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    memory_manager = MemoryManager()

    counselors = {
        "local_adaptive": LocalAdaptiveCounselor(config.LOCAL_MODEL_NAME, memory_manager.memory, "_adaptive"),
        "local_baseline": LocalBaselineCounselor(config.LOCAL_MODEL_NAME, memory_manager.memory, "_baseline_local"),
        "local_no_memory": LocalBaselineNoMemoryCounselor(config.LOCAL_MODEL_NAME),
        "closed_baseline": ClosedBaselineCounselor(config.CLOSED_MODEL_NAME, memory_manager.memory, "_baseline_closed"),
        "closed_no_memory": ClosedBaselineNoMemoryCounselor(config.CLOSED_MODEL_NAME),
    }

    all_results = []
    for case in tqdm(dataset, desc="Processing Cases"):
        case_id = case['case_id']
        print(f"\n--- Simulating Case: {case_id} ---")

        memory_manager.clear_user_history(case_id)

        previous_summary_dict: Optional[Dict] = None
        for i in range(5):
            session = case['sessions'][i]
            print(
                f"  - Ingesting Session {session['session_number']} for memory...")

            memory_manager.add_raw_turns_for_baseline(
                case_id, session['transcript'], "baseline_local")
            memory_manager.add_raw_turns_for_baseline(
                case_id, session['transcript'], "baseline_closed")

            summary_obj: Optional[ClinicalSummary] = run_srs_reflector(
                session['transcript'], previous_summary_dict)
            if summary_obj:
                memory_manager.add_srs_summary_for_adaptive(
                    case_id, summary_obj, session['session_number'])
                if 'summary' not in case['sessions'][i] or not case['sessions'][i]['summary']:
                    case['sessions'][i]['summary'] = summary_obj.model_dump()
                previous_summary_dict = summary_obj.model_dump()

        test_probe = case['sessions'][5]['transcript'][-1]['content']
        case_results = {"case_id": case_id, "models": {}}

        for name, model_instance in counselors.items():
            print(f"  - Evaluating model: {name}...")
            response_str = model_instance.get_response(
                user_id=case_id, case_data=case, test_probe=test_probe)
            scores_obj = run_tasc_evaluation(case, response_str)
            # --- FIX: Dump TascRubricScores using by_alias=True ---
            # This ensures the nested Scores object is also dumped with aliases 'T', 'A', 'S', 'C'
            # which matches your JSON excerpt and the structure the LLM Judge is asked to provide.
            case_results["models"][name] = {
                "response": response_str,
                "scores_data": scores_obj.model_dump(by_alias=True)
            }

        all_results.append(case_results)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

    print("\n\n--- FINAL EVALUATION REPORT ---")
    if not all_results:
        return

    model_names = list(counselors.keys())

    # --- FIX: Define metrics mapping for consistent access and display ---
    # (Display Name, Key in JSON/avg_scores_report)
    # The keys here MUST match the keys in the 'scores' sub-dictionary of 'scores_data'
    # which are 'T', 'A', 'S', 'C', and 'overall_score' if by_alias=True was used for dumping.
    metrics_map_for_report = [
        ("Task Alignment", "T"),
        ("Alliance Bond", "A"),
        ("Stylistic Congruence", "S"),
        ("Congruence With Goals", "C"),
        # overall_score does not use a different alias
        ("Overall Score", "overall_score")
    ]

    avg_scores_report = {}

    for model_name in model_names:
        avg_scores_report[model_name] = {}
        for display_name, key_in_json in metrics_map_for_report:
            metric_data = []
            for r_val in all_results:
                try:
                    # Access the nested scores dict using the correct alias/key
                    score_value = r_val['models'][model_name]['scores_data']['scores'][key_in_json]
                    # Ensure scores are float for np.mean
                    metric_data.append(float(score_value))
                except KeyError:
                    print(
                        f"Warning: Missing score (key '{key_in_json}') for model '{model_name}' in case '{r_val['case_id']}'. Using 0 for average.")
                    metric_data.append(0.0)

            avg_scores_report[model_name][key_in_json] = np.mean(
                metric_data) if metric_data else 0.0

    print("\nAverage Overall T.A.S.C. Scores:")
    for model_name in model_names:
        # Access overall_score using its actual key
        print(
            f"  - {model_name:<30}: {avg_scores_report[model_name]['overall_score']:.3f}")

    print("\nDetailed Average T.A.S.C. Score Breakdown:")
    header = f"{'Metric':<35}"
    for model_name in model_names:
        header += f" | {model_name[:15]:<15}"
    print(header)
    print("-" * len(header))

    # Iterate using the display names and keys from metrics_map_for_report, excluding the overall score
    for display_name, key_in_json in metrics_map_for_report[:-1]:
        row = f"{display_name:<35}"
        for model_name in model_names:
            row += f" | {avg_scores_report[model_name][key_in_json]:<15.3f}"
        print(row)


if __name__ == "__main__":
    main()
