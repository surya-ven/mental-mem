# evaluation.py

import json
import os
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer, util

import config
from llm_utils import call_api_client
from counselor_models import (
    run_srs_reflector, MemoryManager,
    LocalAdaptiveCounselor,
    ClosedAdaptiveCounselor,  # --- NEW: Import the new model ---
    LocalBaselineCounselor, LocalBaselineNoMemoryCounselor,
    ClosedBaselineCounselor, ClosedBaselineNoMemoryCounselor, ClinicalSummary
)

# --- Define the nested Scores model first ---


class Scores(BaseModel):
    task_alignment: int = Field(alias='T')
    alliance_bond: int = Field(alias='A')
    stylistic_congruence: int = Field(alias='S')
    congruence_with_goals: int = Field(alias='C')
    overall_score: float

    class Config:
        populate_by_name = True

# --- T.A.S.C. Evaluation Rubric Schema ---


class TascRubricScores(BaseModel):
    justification: Union[str, Dict[str, Any]]
    scores: Scores


def calculate_synthesis_accuracy(predicted_profile: Optional[Dict[str, Any]], ground_truth_profile: Dict[str, Any]) -> float:
    # Added Optional to predicted_profile and a check
    if not predicted_profile:
        return 0.0  # Or handle as an error/specific value
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_notes_as_string(profile: Dict[str, Any]) -> str:
        # Assuming ClinicalSummary might not have this, but it should
        notes = profile.get('evolution_notes', '')
        if isinstance(notes, dict):
            return json.dumps(notes)
        return str(notes)
    # Ground truth for evolution notes comes from the dataset's evolved_profile if available
    # For this study, the ClinicalSummary's 'emerging_themes' or 'therapeutic_milestones' might be more relevant
    # than a direct 'evolution_notes' field if ClinicalSummary doesn't have it.
    # Let's assume ground_truth_profile refers to case_data['sessions'][4]['summary'] (last SRS summary)
    # and it contains something comparable to 'evolution_notes' or we compare specific fields.
    # For simplicity, if 'evolution_notes' is the target, ensure it's in the ClinicalSummary or adapt.
    # The current ClinicalSummary does not have 'evolution_notes'.
    # Synthesis accuracy might need to compare 'emerging_themes' or 'therapeutic_milestones' from predicted vs. ground_truth.
    # Let's assume we are comparing a field that exists, e.g. concatenated string of milestones or themes.
    # This function might need more refinement based on what 'predicted_profile' (ClinicalSummary) contains.
    # For now, we'll attempt to use 'emerging_themes' as a proxy.
    gt_notes_proxy = " ".join(ground_truth_profile.get('emerging_themes', []))
    pred_notes_proxy = " ".join(predicted_profile.get('emerging_themes', []))

    if not gt_notes_proxy or not pred_notes_proxy:
        return 0.0
    embedding1 = model.encode(gt_notes_proxy, convert_to_tensor=True)
    embedding2 = model.encode(pred_notes_proxy, convert_to_tensor=True)
    return max(0, util.pytorch_cos_sim(embedding1, embedding2).item())


def run_tasc_evaluation(case_data: Dict, counselor_response: str) -> TascRubricScores:
    final_session_plan = "Not available"
    # Ensure we get summary from the correct session (session 5 is at index 4)
    if len(case_data['sessions']) > 4 and 'summary' in case_data['sessions'][4] and case_data['sessions'][4]['summary'] is not None:
        summary_data = case_data['sessions'][4]['summary']
        if isinstance(summary_data, dict):
            final_session_plan = summary_data.get(
                'plan_for_next_session', "Not available")
        # ClinicalSummary object is not directly stored in case_data, only its .model_dump()
        # So, the isinstance(summary_data, ClinicalSummary) check might not be relevant here
        # as 'summary' would be a dict.

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

    # --- UPDATED: Add ClosedAdaptiveCounselor to the cohort ---
    counselors = {
        "local_adaptive": LocalAdaptiveCounselor(config.LOCAL_MODEL_NAME, memory_manager.memory, "_adaptive"),
        # Uses same "_adaptive" suffix
        "closed_adaptive": ClosedAdaptiveCounselor(config.CLOSED_MODEL_NAME, memory_manager.memory, "_adaptive"),
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
            memory_manager.add_raw_turns_for_baseline(  # Also for the closed baseline track
                case_id, session['transcript'], "baseline_closed")

            summary_obj: Optional[ClinicalSummary] = run_srs_reflector(
                session['transcript'], previous_summary_dict)
            if summary_obj:
                memory_manager.add_srs_summary_for_adaptive(
                    case_id, summary_obj, session['session_number'])
                # Store the summary in case_data for the judge's context (for all models)
                case['sessions'][i]['summary'] = summary_obj.model_dump()
                previous_summary_dict = summary_obj.model_dump()
            # Ensure summary field exists if reflector fails
            elif i > 0 and 'summary' not in case['sessions'][i]:
                case['sessions'][i]['summary'] = previous_summary_dict if previous_summary_dict else {}

        test_probe = case['sessions'][5]['transcript'][-1]['content']
        case_results = {"case_id": case_id, "models": {}}

        for name, model_instance in counselors.items():
            print(f"  - Evaluating model: {name}...")
            response_str = model_instance.get_response(
                user_id=case_id, case_data=case, test_probe=test_probe)
            scores_obj = run_tasc_evaluation(case, response_str)
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

    # This will now include "closed_adaptive"
    model_names = list(counselors.keys())

    metrics_map_for_report = [
        ("Task Alignment", "T"),
        ("Alliance Bond", "A"),
        ("Stylistic Congruence", "S"),
        ("Congruence With Goals", "C"),
        ("Overall Score", "overall_score")
    ]

    avg_scores_report = {}

    for model_name in model_names:
        avg_scores_report[model_name] = {}
        for display_name, key_in_json in metrics_map_for_report:
            metric_data = []
            for r_val in all_results:
                try:
                    # Ensure scores_data and its nested scores exist
                    model_result = r_val.get('models', {}).get(model_name, {})
                    scores_data = model_result.get('scores_data', {})
                    nested_scores = scores_data.get('scores', {})

                    score_value = nested_scores.get(key_in_json)
                    if score_value is not None:
                        metric_data.append(float(score_value))
                    else:
                        print(
                            f"Warning: Score key '{key_in_json}' not found for model '{model_name}' in case '{r_val['case_id']}' under 'scores'. Using 0.")
                        metric_data.append(0.0)
                except Exception as e:  # Catch any other potential access errors
                    print(
                        f"Error accessing score (key '{key_in_json}') for model '{model_name}' in case '{r_val.get('case_id', 'Unknown')}': {e}. Using 0.")
                    metric_data.append(0.0)

            avg_scores_report[model_name][key_in_json] = np.mean(
                metric_data) if metric_data else 0.0

    print("\nAverage Overall T.A.S.C. Scores:")
    for model_name in model_names:
        print(
            f"  - {model_name:<30}: {avg_scores_report[model_name]['overall_score']:.3f}")

    print("\nDetailed Average T.A.S.C. Score Breakdown:")
    header = f"{'Metric':<35}"
    for model_name in model_names:
        header += f" | {model_name[:15]:<15}"  # Truncate model_name for header
    print(header)
    print("-" * len(header))

    for display_name, key_in_json in metrics_map_for_report[:-1]:
        row = f"{display_name:<35}"
        for model_name in model_names:
            row += f" | {avg_scores_report[model_name][key_in_json]:<15.3f}"
        print(row)


if __name__ == "__main__":
    main()
