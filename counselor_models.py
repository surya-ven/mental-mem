# counselor_models.py

import json
from mem0 import Memory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

import config
from llm_utils import call_api_client
from shield_gemma_utils import ShieldGemmaModerator, ShieldGemmaResponse, UseCase as ShieldUseCase

# --- Helper function for logging ---


def append_to_log(log_file_path: str, message: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# --- Schemas ---


class ClinicalSummary(BaseModel):
    session_focus: str = Field(
        description="The main topic or focus of this session.")
    user_emotional_state: str = Field(
        description="The user's primary emotional state during the session.")
    therapeutic_milestones: List[str] = Field(
        default_factory=list, description="Key insights, breakthroughs, or progress made by the user.")
    emerging_themes: List[str] = Field(
        default_factory=list, description="Recurring topics or underlying themes.")
    plan_for_next_session: str = Field(
        description="The concrete plan or task for the user.")
    key_user_quotes_from_session: List[str] = Field(
        default_factory=list, description="Impactful user quotes.")
    counselor_reflections_on_session: str = Field(
        description="Psychologist-level reflections.")
    suggested_focus_points_for_probe: List[str] = Field(
        default_factory=list, description="Points for the next counselor to consider.")
    session_safety_alerts: List[str] = Field(
        default_factory=list, description="Safety flags from user input in session.")


class CounselorResponse(BaseModel):
    response: str

# --- Reflector ---


def run_srs_reflector(
    session_transcript: List[Dict],
    previous_summary_obj: Optional[ClinicalSummary] = None,
    session_safety_alerts: Optional[List[str]] = None
) -> Optional[ClinicalSummary]:
    transcript_text = "\n".join(
        [f"{t['role']}: {t['content']}" for t in session_transcript])
    previous_summary_text = previous_summary_obj.model_dump_json(
        indent=2) if previous_summary_obj else "N/A"
    safety_alerts_for_prompt = json.dumps(
        session_safety_alerts) if session_safety_alerts else "None noted"
    prompt = f"""
    You are a supervising clinical psychologist. Distill a therapy session into an ENHANCED Clinical Summary.
    PREVIOUS SUMMARY: {previous_summary_text}
    CURRENT SESSION TRANSCRIPT: {transcript_text}
    AUTOMATED SAFETY ALERTS FOR USER INPUTS IN CURRENT SESSION: {safety_alerts_for_prompt}
    TASK: Generate a new Clinical Summary JSON. Include EXACT keys: "session_focus", "user_emotional_state", 
    "therapeutic_milestones" (LIST OF STRINGS), "emerging_themes" (LIST OF STRINGS), 
    "plan_for_next_session", "key_user_quotes_from_session" (LIST OF STRINGS), 
    "counselor_reflections_on_session", "suggested_focus_points_for_probe" (LIST OF STRINGS),
    "session_safety_alerts" (LIST OF STRINGS, from "AUTOMATED SAFETY ALERTS" or empty []).
    Ensure LIST OF STRINGS are JSON arrays (e.g., [] or ["item"]).
    """
    return call_api_client(prompt, ClinicalSummary, model=config.REFLECTOR_MODEL, expect_json_object=True)

# --- MemoryManager ---


class MemoryManager:
    def __init__(self):
        self.memory = Memory.from_config(config.MEM0_CONFIG)
        print("MemoryManager initialized with mem0 configured.")

    def add_raw_turns_for_baseline(self, user_id: str, transcript: List[Dict], track_suffix: str):
        payloads = [{"role": t['role'], "content": t['content']}
                    for t in transcript]
        if payloads:
            self.memory.add(
                payloads, user_id=f"{user_id}_{track_suffix}", infer=False)

    def add_srs_summary_for_adaptive(self, user_id: str, summary: ClinicalSummary, session_num: int):
        meta = {"type": "srs_summary", "session_number": session_num}
        self.memory.add(summary.model_dump_json(
        ), user_id=f"{user_id}_adaptive", metadata=meta, infer=False)

    def clear_user_history(self, user_id: str):
        [self.memory.delete_all(user_id=f"{user_id}{s}") for s in [
            "_adaptive", "_baseline_local", "_baseline_closed"]]

# --- Base Counselor ---


class BaseCounselor:
    def __init__(self, model_name: str, memory_instance: Optional[Memory], user_id_track_suffix: str):
        self.model_name = model_name
        self.memory = memory_instance
        self.user_id_track_suffix = user_id_track_suffix

    def _prepare_context(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        raise NotImplementedError("Subclasses must implement _prepare_context")

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        raise NotImplementedError(
            "Subclasses must implement _construct_llm_prompt")

    def get_response(self, user_id: str, case_data: Dict, test_probe_content: str, log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator]) -> str:
        context_for_llm = self._prepare_context(
            user_id, case_data, test_probe_content, log_file_path)
        return _generate_and_moderate_response(
            self, user_id, case_data, test_probe_content, log_file_path,
            shield_moderator, context_for_llm,
            expect_json_counselor_output=False
        )


# --- Safety constants & Helper ---
DEFAULT_ESCALATION_MESSAGE = "I'm hearing a lot of pain and distress in what you're sharing, and I'm concerned for your safety. It's really important to talk to someone who can offer immediate support. Please reach out to a crisis hotline or emergency services. There are people available right now who want to help you through this."
DEFAULT_SAFE_FALLBACK_MODEL_OUTPUT = "I understand you're looking for guidance on this topic. It's a complex area, and for sensitive matters, it's always best to consult with a qualified professional or refer to trusted, expert resources. I'm here to support your emotional processing as we've discussed."


def _generate_and_moderate_response(
    counselor_obj: BaseCounselor, user_id: str, case_data: Dict, test_probe_content: str,
    log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator],
    context_for_llm: str, expect_json_counselor_output: bool
) -> str:
    append_to_log(
        log_file_path, f"--- {counselor_obj.__class__.__name__} Start ({counselor_obj.model_name}) ---")
    if shield_moderator:
        user_probe_assessment = shield_moderator.moderate_text(
            test_probe_content, ShieldUseCase.PROMPT_ONLY)
        append_to_log(
            log_file_path, f"User Probe ShieldGemma Assessment: {user_probe_assessment.model_dump_json(indent=2) if user_probe_assessment else 'Skipped'}")
        if user_probe_assessment and user_probe_assessment.is_self_harm_detected:
            append_to_log(log_file_path, "SELF-HARM IN PROBE. Escalating.")
            return DEFAULT_ESCALATION_MESSAGE
    else:
        append_to_log(
            log_file_path, "ShieldGemma N/A. Skipping user probe moderation.")

    system_prompt, prompt_content_main = counselor_obj._construct_llm_prompt(
        context_for_llm, test_probe_content)
    append_to_log(
        log_file_path, f"\nLLM Input ({counselor_obj.model_name}):\nSysPrompt: {system_prompt}\nPrompt (excerpt): {prompt_content_main[:500]}...")

    api_response = call_api_client(
        prompt_content_main, CounselorResponse, model=counselor_obj.model_name,
        system_prompt=system_prompt, expect_json_object=expect_json_counselor_output
    )
    generated_text = api_response.strip() if isinstance(api_response, str) else \
        (api_response.response if isinstance(api_response, CounselorResponse)
         else "Error: Could not generate response.")
    append_to_log(log_file_path, f"Initial Model Output: {generated_text}")

    final_response = generated_text
    if shield_moderator:
        output_assessment = shield_moderator.moderate_text(
            generated_text, ShieldUseCase.PROMPT_RESPONSE, original_user_prompt=test_probe_content)
        append_to_log(
            log_file_path, f"Model Output ShieldGemma: {output_assessment.model_dump_json(indent=2) if output_assessment else 'Skipped'}")
        if output_assessment and output_assessment.is_generally_unsafe_detected:
            append_to_log(log_file_path, "HARMFUL MODEL OUTPUT. Overriding.")
            final_response = DEFAULT_SAFE_FALLBACK_MODEL_OUTPUT
    else:
        append_to_log(
            log_file_path, "ShieldGemma N/A. Skipping output moderation.")

    append_to_log(
        log_file_path, f"Final Response: {final_response}\n--- {counselor_obj.__class__.__name__} End ---\n")
    return final_response

# --- Counselor Implementations ---


class LocalAdaptiveCounselor(BaseCounselor):
    def _prepare_context(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        context = "No clinical summary available for context."
        if self.memory:
            uid_for_search = f"{user_id}{self.user_id_track_suffix}"
            filters = {"type": "srs_summary"}
            append_to_log(
                log_file_path, f"Mem0 Search for SRS: UserID='{uid_for_search}', Query='{test_probe}', Filters={json.dumps(filters)}")
            resp = self.memory.search(
                query=test_probe, user_id=uid_for_search, filters=filters, limit=1)
            append_to_log(
                log_file_path, f"Mem0 SRS Resp: {json.dumps(resp, indent=2)}")
            res = resp.get('results', [])
            if res:
                try:
                    s_dict = json.loads(res[0]['memory'])
                    s_obj = ClinicalSummary.model_validate(s_dict)
                    context = (f"Overall Focus: {s_obj.session_focus}\nState: {s_obj.user_emotional_state}\nMilestones: {json.dumps(s_obj.therapeutic_milestones)}\nThemes: {json.dumps(s_obj.emerging_themes)}\nQuotes: {json.dumps(s_obj.key_user_quotes_from_session)}\nReflections: {s_obj.counselor_reflections_on_session}\nFocus Points: {json.dumps(s_obj.suggested_focus_points_for_probe)}\nPlan: {s_obj.plan_for_next_session}\nSafety Alerts: {json.dumps(s_obj.session_safety_alerts)}")
                except Exception as e:
                    context = f"Error parsing summary: {res[0]['memory']}. Err: {e}"
        append_to_log(
            log_file_path, f"Formatted Clinical Summary for Prompt:\n{context}")
        return context

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor. Use the detailed clinical summary (focus, state, milestones, themes, plan, quotes, reflections, focus points, safety alerts) to inform your response to the user's current message, applying the 'plan_for_next_session'."
        prompt_content = f"CLINICAL SUMMARY:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content


class ClosedAdaptiveCounselor(LocalAdaptiveCounselor):
    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor (large model). Use the detailed clinical summary (focus, state, milestones, themes, plan, quotes, reflections, focus points, safety alerts) to respond, applying the 'plan_for_next_session'."
        prompt_content = f"CLINICAL SUMMARY:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content


class BaselineRAGCounselor(BaseCounselor):
    def _prepare_context(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        context = "No relevant past conversation snippets found."
        if self.memory:
            uid_for_search = f"{user_id}{self.user_id_track_suffix}"
            append_to_log(
                log_file_path, f"Mem0 RAG Search: UserID='{uid_for_search}', Query='{test_probe}'")
            resp = self.memory.search(
                query=test_probe, user_id=uid_for_search, limit=5)
            append_to_log(
                log_file_path, f"Mem0 RAG Resp: {json.dumps(resp, indent=2)}")
            res = resp.get('results', [])
            if res:
                context = "\n".join([m['memory'] for m in res])
        append_to_log(
            log_file_path, f"Extracted RAG Context for Prompt:\n{context}")
        return context

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor. Use provided past conversation snippets to respond."
        prompt_content = f"PAST SNIPPETS:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content


class LocalBaselineCounselor(BaselineRAGCounselor):
    pass


class ClosedBaselineCounselor(BaselineRAGCounselor):
    pass


class NoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str): super().__init__(model_name, None, "")

    def _prepare_context(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        # --- MODIFIED: No Memory models get NO prior session context ---
        context = "No prior session context is available for this interaction."
        append_to_log(
            log_file_path, f"Context for {self.__class__.__name__}: {context}")
        return context

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        # Context will be "No prior session context..."
        system_prompt = "You are an empathetic AI counselor. Respond to the user's message based only on the current interaction, without any prior session history."
        prompt_content = f"USER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        # We don't include the "LIMITED CONTEXT:" prefix if it's truly no context
        return system_prompt, prompt_content


class LocalBaselineNoMemoryCounselor(NoMemoryCounselor):
    # Inherits _prepare_context and _construct_llm_prompt from NoMemoryCounselor.
    # No further changes needed here as they are now truly no-memory.
    pass


class ClosedBaselineNoMemoryCounselor(NoMemoryCounselor):
    # Inherits _prepare_context and _construct_llm_prompt from NoMemoryCounselor.
    # The original distinction for ClosedBaselineNoMemory was its ability to see S1-S5.
    # To make it a true "no memory" like LocalBaselineNoMemory for this test,
    # it will also use the NoMemoryCounselor's context preparation.
    # If you want it to see full history *without mem0*, that's a different setup.
    # For this request (no ingestion beforehand), it behaves like LocalBaselineNoMemory.
    pass
