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
    LocalAdaptiveCounselor, ClosedAdaptiveCounselor,
    LocalBaselineCounselor, LocalBaselineNoMemoryCounselor,
    ClosedBaselineCounselor, ClosedBaselineNoMemoryCounselor, ClinicalSummary,
    append_to_log
)

# Schemas (Scores, TascRubricScores) remain the same


class Scores(BaseModel):
    task_alignment: int = Field(alias='T')
    alliance_bond: int = Field(alias='A')
    stylistic_congruence: int = Field(alias='S')
    congruence_with_goals: int = Field(alias='C')
    overall_score: float

    class Config:
        populate_by_name = True


class TascRubricScores(BaseModel):
    justification: Union[str, Dict[str, Any]]
    scores: Scores


def calculate_synthesis_accuracy(predicted_profile: Optional[Dict[str, Any]], ground_truth_profile: Dict[str, Any]) -> float:
    if not predicted_profile:
        return 0.0
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_notes_as_string(profile: Optional[Dict[str, Any]]) -> str:
        if not profile:
            return ""
        # Assuming ClinicalSummary structure is used for predicted_profile
        # and ground_truth_profile might be the dataset's evolved_profile or a similar summary
        # For SRS, we compare key synthesized elements.
        # If predicted_profile is a ClinicalSummary dict:
        notes_list = profile.get('emerging_themes', []) + \
            profile.get('therapeutic_milestones', [])
        return " ".join(notes_list)

    gt_notes_proxy = get_notes_as_string(ground_truth_profile.get(
        'summary'))  # Assuming summary is where comparable data is
    pred_notes_proxy = get_notes_as_string(predicted_profile)

    if not gt_notes_proxy or not pred_notes_proxy:
        return 0.0
    embedding1 = model.encode(gt_notes_proxy, convert_to_tensor=True)
    embedding2 = model.encode(pred_notes_proxy, convert_to_tensor=True)
    return max(0, util.pytorch_cos_sim(embedding1, embedding2).item())


def run_tasc_evaluation(case_data: Dict, counselor_response: str, log_file_path: str) -> TascRubricScores:
    append_to_log(log_file_path, "\n--- T.A.S.C. Evaluation Call ---")
    final_session_plan = "Not available"
    # Ensure we get summary from the correct session (session 5 is at index 4)
    # and that the summary itself is not None before trying to access keys.
    if len(case_data['sessions']) >= 5 and \
       'summary' in case_data['sessions'][4] and \
       case_data['sessions'][4]['summary'] is not None:
        summary_data = case_data['sessions'][4]['summary']
        if isinstance(summary_data, dict):
            final_session_plan = summary_data.get(
                'plan_for_next_session', "Not available")

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
    3.  **S - Stylistic Congruence**: User's preferred style from initial profile: "{case_data.get('initial_profile', {}).get('preferred_style', 'general supportive style')}".
        - 5 (Excellent): Perfectly embodies the requested style, making the interaction feel naturally and precisely tailored.
        - 4 (Good): Consistently and clearly matches the requested style.
        - 3 (Moderate): Attempts to match the style but does so inconsistently or awkwardly.
        - 2 (Weak): The style is generic and does not align with the user's preference.
        - 1 (Poor): The style is the opposite of what the user requested (e.g., highly directive when reflective was asked for).
    4.  **C - Congruence with Goals**: User's evolved goals (if available, otherwise general progress): "{case_data.get('evolved_profile', {}).get('goals', 'general therapeutic progress')}".
        - 5 (Excellent): Masterfully connects the task to implicit therapeutic goals like self-understanding or emotional regulation, or stated evolved goals.
        - 4 (Good): The response clearly helps the user move toward general therapeutic progress or stated evolved goals.
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
        append_to_log(
            log_file_path, "T.A.S.C. Evaluation FAILED for this response (Judge LLM call failed).")
        # Use scores of 0 to indicate a hard failure of the judging process for this response
        default_scores_sub = Scores(T=0, A=0, S=0, C=0, overall_score=0.0)
        return TascRubricScores(justification="Judge LLM call failed or returned invalid format.", scores=default_scores_sub)

    append_to_log(
        log_file_path, f"T.A.S.C. Raw Scores: {json.dumps(judge_response_obj.model_dump(by_alias=True), indent=2)}")
    return TascRubricScores(justification=judge_response_obj.justification, scores=judge_response_obj.scores)


def main():
    DATASET_PATH = os.path.join("output", "counseling_dataset_6sessions.json")
    RESULTS_PATH = os.path.join(
        "output", "evaluation_results_full_cohort.json")

    os.makedirs(config.LOG_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print(
            f"Dataset not found at {DATASET_PATH}. Please run `generate_dataset.py` first.")
        return

    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    memory_manager = MemoryManager()

    counselors = {
        "local_adaptive": LocalAdaptiveCounselor(config.LOCAL_MODEL_NAME, memory_manager.memory, "_adaptive"),
        "closed_adaptive": ClosedAdaptiveCounselor(config.CLOSED_MODEL_NAME, memory_manager.memory, "_adaptive"),
        "local_baseline": LocalBaselineCounselor(config.LOCAL_MODEL_NAME, memory_manager.memory, "_baseline_local"),
        "local_no_memory": LocalBaselineNoMemoryCounselor(config.LOCAL_MODEL_NAME),
        "closed_baseline": ClosedBaselineCounselor(config.CLOSED_MODEL_NAME, memory_manager.memory, "_baseline_closed"),
        "closed_no_memory": ClosedBaselineNoMemoryCounselor(config.CLOSED_MODEL_NAME),
    }

    all_results = []
    print(f"Starting evaluation for {len(dataset)} cases...")

    for case_idx, case in enumerate(tqdm(dataset, desc="Processing Cases")):
        case_id = case['case_id']

        case_log_dir = os.path.join(config.LOG_DIR, case_id)
        os.makedirs(case_log_dir, exist_ok=True)

        print(f"\n--- Simulating Case: {case_id} ---")
        case_progress_log_file = os.path.join(
            case_log_dir, "_case_simulation_log.txt")
        append_to_log(case_progress_log_file,
                      f"--- Simulating Case: {case_id} Start ---")

        memory_manager.clear_user_history(case_id)
        append_to_log(case_progress_log_file,
                      "Cleared user history for all tracks.")

        previous_summary_dict: Optional[Dict] = None
        # Populate initial_profile and evolved_profile into case_data if not already flat
        # This is for the judge prompt. Dataset generation already creates these.
        # We just need to ensure the judge has access to 'preferred_style' and 'goals'.
        # The 'summary' field from Session 5 is used for 'plan_for_next_session'.
        current_case_initial_profile = {}
        # This would be from session 5's summary ideally for evolved goals
        current_case_evolved_profile = {}

        for i in range(5):
            session = case['sessions'][i]
            session_num_for_log = session['session_number']
            print(f"  - Ingesting Session {session_num_for_log} for memory...")
            append_to_log(case_progress_log_file,
                          f"  - Ingesting Session {session_num_for_log} for memory...")

            memory_manager.add_raw_turns_for_baseline(
                case_id, session['transcript'], "baseline_local")
            memory_manager.add_raw_turns_for_baseline(
                case_id, session['transcript'], "baseline_closed")

            summary_obj: Optional[ClinicalSummary] = run_srs_reflector(
                session['transcript'], previous_summary_dict)
            if summary_obj:
                memory_manager.add_srs_summary_for_adaptive(
                    case_id, summary_obj, session_num_for_log)
                # Storing for judge's context
                case['sessions'][i]['summary'] = summary_obj.model_dump()
                previous_summary_dict = summary_obj.model_dump()
                append_to_log(
                    case_progress_log_file, f"    SRS Summary for Session {session_num_for_log} generated and stored for _adaptive track.")
                if i == 0:  # Assuming first summary has initial profile elements
                    current_case_initial_profile = {
                        # Placeholder
                        "preferred_style": previous_summary_dict.get("session_focus")}
                if i == 4:  # Assuming last context summary has evolved goals for the judge
                    current_case_evolved_profile = {
                        "goals": previous_summary_dict.get("therapeutic_milestones", [])}
            elif i > 0 and ('summary' not in case['sessions'][i] or not case['sessions'][i]['summary']):
                case['sessions'][i]['summary'] = previous_summary_dict if previous_summary_dict else {}
                append_to_log(
                    case_progress_log_file, f"    SRS Reflector FAILED for Session {session_num_for_log}. Using previous summary for context if available.")
            elif i == 0 and ('summary' not in case['sessions'][i] or not case['sessions'][i]['summary']):
                # Ensure summary key exists even if reflector fails early
                case['sessions'][i]['summary'] = {}

        # Pass relevant profile info to the judge via case_data for T.A.S.C. evaluation context
        # The dataset itself should ideally contain initial_profile and evolved_profile at the case level
        # For now, we crudely assign them if the generate_dataset doesn't create them at root of case.
        if 'initial_profile' not in case:  # If generate_dataset.py doesn't make it
            case['initial_profile'] = case['sessions'][0].get(
                'summary', {})  # A rough approximation
        if 'evolved_profile' not in case:  # If generate_dataset.py doesn't make it
            case['evolved_profile'] = case['sessions'][4].get(
                'summary', {})  # A rough approximation

        print(f"  --- Memory Ingestion Complete for {case_id} ---")
        append_to_log(case_progress_log_file,
                      "--- Memory Ingestion Complete ---")

        test_probe = case['sessions'][5]['transcript'][-1]['content']
        case_results_data = {"case_id": case_id, "models": {}}

        for name, model_instance in counselors.items():
            model_log_dir = os.path.join(case_log_dir, name)
            os.makedirs(model_log_dir, exist_ok=True)
            log_file_path = os.path.join(model_log_dir, "interaction_log.txt")

            print(f"  - Evaluating model: {name}...")
            append_to_log(case_progress_log_file,
                          f"  - Evaluating model: {name}...")

            response_str = model_instance.get_response(
                user_id=case_id, case_data=case, test_probe=test_probe, log_file_path=log_file_path
            )

            # --- ADDED CHECK FOR ERROR RESPONSE ---
            error_signature = "Error: Could not generate response."
            if error_signature in response_str:
                append_to_log(
                    log_file_path, f"Model {name} returned an error: {response_str}")
                # Assign a default 'failed' score set (all 0s)
                failed_scores_sub = Scores(
                    T=0, A=0, S=0, C=0, overall_score=0.0)
                scores_obj = TascRubricScores(
                    justification=f"Model {name} failed to generate a valid response.",
                    scores=failed_scores_sub
                )
            else:
                scores_obj = run_tasc_evaluation(
                    case, response_str, log_file_path)

            case_results_data["models"][name] = {
                "response": response_str,  # Store original response, even if error
                "scores_data": scores_obj.model_dump(by_alias=True)
            }

        all_results.append(case_results_data)
        append_to_log(case_progress_log_file,
                      f"--- Simulating Case: {case_id} End ---")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

    print("\n\n--- FINAL EVALUATION REPORT ---")
    if not all_results:
        print("No results to report.")
        return

    model_names = list(counselors.keys())
    metrics_map_for_report = [
        ("Task Alignment", "T"), ("Alliance Bond", "A"),
        ("Stylistic Congruence", "S"), ("Congruence With Goals", "C"),
        ("Overall Score", "overall_score")
    ]
    avg_scores_report = {}

    for model_name in model_names:
        avg_scores_report[model_name] = {}
        for display_name, key_in_json in metrics_map_for_report:
            metric_data = []
            for r_val in all_results:
                try:
                    model_result = r_val.get('models', {}).get(model_name, {})
                    scores_data = model_result.get('scores_data', {})
                    # 'scores' is the key for the sub-dictionary
                    nested_scores = scores_data.get('scores', {})

                    # Use key_in_json ('T', 'A', etc.)
                    score_value = nested_scores.get(key_in_json)
                    if score_value is not None:
                        metric_data.append(float(score_value))
                    else:
                        # This warning now correctly flags if an expected alias is missing
                        # print(f"Warning: Score key '{key_in_json}' not found for model '{model_name}' in case '{r_val['case_id']}' under 'scores' sub-dict. Using 0.")
                        metric_data.append(0.0)
                except Exception as e:
                    # This generic exception will catch issues if 'scores' sub-dict itself is missing, etc.
                    # print(f"Error accessing score (key '{key_in_json}') for model '{model_name}' in case '{r_val.get('case_id', 'Unknown')}': {e}. Using 0.")
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
        header += f" | {model_name[:15]:<15}"
    print(header)
    print("-" * len(header))

    for display_name, key_in_json in metrics_map_for_report[:-1]:
        row = f"{display_name:<35}"
        for model_name in model_names:
            row += f" | {avg_scores_report[model_name][key_in_json]:<15.3f}"
        print(row)


if __name__ == "__main__":
    main()
