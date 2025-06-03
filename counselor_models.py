# counselor_models.py

import json
from mem0 import Memory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

import config
from llm_utils import call_api_client
# Assuming shield_gemma_utils.py and its contents are correctly defined
from shield_gemma_utils import ShieldGemmaModerator, ShieldGemmaResponse, UseCase as ShieldUseCase

# --- Helper function for logging ---


def append_to_log(log_file_path: str, message: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# --- Schemas for our Enhanced SRS Method ---


class ClinicalSummary(BaseModel):
    session_focus: str = Field(
        description="The main topic or focus of this session.")
    user_emotional_state: str = Field(
        description="The user's primary emotional state during the session.")
    therapeutic_milestones: List[str] = Field(
        default_factory=list, description="Key insights, breakthroughs, or progress made by the user. Must be a list of strings.")
    emerging_themes: List[str] = Field(
        default_factory=list, description="Recurring topics or underlying themes noticed across sessions. Must be a list of strings.")
    plan_for_next_session: str = Field(
        description="The concrete plan or task for the user to focus on before the next session.")
    key_user_quotes_from_session: List[str] = Field(
        default_factory=list, description="Short, impactful verbatim quotes from the user during this session that highlight key moments or feelings.")
    counselor_reflections_on_session: str = Field(
        description="Brief (1-2 sentences) psychologist-level reflections on the session's dynamics, user's progress, potential stuck points, or areas for future exploration.")
    suggested_focus_points_for_probe: List[str] = Field(
        default_factory=list, description="Based on the entire summary, list 1-3 specific themes, past events, or user statements the responding counselor should consider or link to when addressing the next user probe.")
    session_safety_alerts: List[str] = Field(
        default_factory=list, description="Alerts if user input in session triggered safety flags (e.g., 'self_harm_detected_in_turn_X').")


class CounselorResponse(BaseModel):
    response: str

# --- "Reflector" ---


def run_srs_reflector(
    session_transcript: List[Dict],
    previous_summary_obj: Optional[ClinicalSummary] = None,
    # --- FIX: Changed parameter name ---
    session_safety_alerts: Optional[List[str]] = None
) -> Optional[ClinicalSummary]:
    transcript_text = "\n".join(
        [f"{t['role']}: {t['content']}" for t in session_transcript])
    previous_summary_text = previous_summary_obj.model_dump_json(
        indent=2) if previous_summary_obj else "N/A"

    # Use the passed 'session_safety_alerts' directly for the prompt
    safety_alerts_for_prompt = json.dumps(
        session_safety_alerts) if session_safety_alerts else "None noted"

    prompt = f"""
    You are a supervising clinical psychologist. Your task is to distill a therapy session into an enhanced, structured clinical summary.

    PREVIOUS CLINICAL SUMMARY (if any):
    {previous_summary_text}

    CURRENT SESSION TRANSCRIPT:
    {transcript_text}
    
    AUTOMATED SAFETY ALERTS IDENTIFIED FOR USER INPUTS IN CURRENT SESSION: {safety_alerts_for_prompt}

    TASK:
    Review all information and generate a new, ENHANCED Clinical Summary.
    Output a single, valid JSON object with the following EXACT top-level keys and structure:
    "session_focus", "user_emotional_state", 
    "therapeutic_milestones" (LIST OF STRINGS, e.g., ["User realized X"]),
    "emerging_themes" (LIST OF STRINGS, e.g., ["Fear of failure"]),
    "plan_for_next_session",
    "key_user_quotes_from_session" (LIST OF STRINGS),
    "counselor_reflections_on_session",
    "suggested_focus_points_for_probe" (LIST OF STRINGS),
    "session_safety_alerts" (LIST OF STRINGS, this should be the list of strings directly from "AUTOMATED SAFETY ALERTS" above. If "None noted" above, output an empty list []).
    Ensure all LIST OF STRINGS fields are proper JSON arrays (e.g., [] or ["item"]).
    """
    summary = call_api_client(prompt, ClinicalSummary,
                              model=config.REFLECTOR_MODEL, expect_json_object=True)
    return summary

# --- MemoryManager ---


class MemoryManager:
    def __init__(self):
        self.memory = Memory.from_config(config.MEM0_CONFIG)
        print("MemoryManager initialized with mem0 configured (Vector Store: Qdrant, LLM/Embedder via OpenRouter).")

    def add_raw_turns_for_baseline(self, user_id: str, transcript: List[Dict], track_suffix: str):
        memory_payloads = []
        for turn in transcript:
            memory_payloads.append(
                {"role": turn['role'], "content": turn['content']})
        if memory_payloads:
            self.memory.add(memory_payloads,
                            user_id=f"{user_id}_{track_suffix}", infer=False)

    def add_srs_summary_for_adaptive(self, user_id: str, summary: ClinicalSummary, session_num: int):
        summary_json_string = summary.model_dump_json()
        metadata_payload = {"type": "srs_summary",
                            "session_number": session_num}
        self.memory.add(
            summary_json_string,
            user_id=f"{user_id}_adaptive",
            metadata=metadata_payload,
            infer=False
        )

    def clear_user_history(self, user_id: str):
        self.memory.delete_all(user_id=f"{user_id}_adaptive")
        self.memory.delete_all(user_id=f"{user_id}_baseline_local")
        self.memory.delete_all(user_id=f"{user_id}_baseline_closed")

# --- Base Class for All Counselors ---


class BaseCounselor:
    def __init__(self, model_name: str, memory_instance: Optional[Memory], user_id_track_suffix: str):
        self.model_name = model_name
        self.memory = memory_instance
        self.user_id_track_suffix = user_id_track_suffix

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        """This method should be implemented by subclasses to define their specific system and user prompts."""
        raise NotImplementedError(
            "Subclasses must implement _construct_llm_prompt")

    def get_response(self, user_id: str, case_data: Dict, test_probe_content: str, log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator]) -> str:
        # This method will call _generate_and_moderate_response
        # Subclasses will define _prepare_context and _construct_llm_prompt
        # The expect_json_counselor_output will be False for all counselors as per last update.
        context_for_llm = self._prepare_context(
            user_id, test_probe_content, log_file_path)  # type: ignore
        return _generate_and_moderate_response(
            self, user_id, case_data, test_probe_content, log_file_path,
            shield_moderator,
            lambda s, u, t, l: context_for_llm,  # Pass already prepared context
            expect_json_counselor_output=False
        )


# --- Safety constants ---
DEFAULT_ESCALATION_MESSAGE = "I'm hearing a lot of pain and distress in what you're sharing, and I'm concerned for your safety. It's really important to talk to someone who can offer immediate support. Please reach out to a crisis hotline or emergency services. There are people available right now who want to help you through this."
DEFAULT_SAFE_FALLBACK_MODEL_OUTPUT = "I understand you're looking for guidance on this topic. It's a complex area, and for sensitive matters, it's always best to consult with a qualified professional or refer to trusted, expert resources. I'm here to support your emotional processing as we've discussed."

# --- Centralized response generation with safety checks ---


def _generate_and_moderate_response(
    counselor_obj: BaseCounselor,  # Takes BaseCounselor for self access
    user_id: str,
    case_data: Dict,
    test_probe_content: str,
    log_file_path: str,
    shield_moderator: Optional[ShieldGemmaModerator],
    # context_for_llm is now directly passed after being prepared by the subclass
    context_for_llm: str,
    expect_json_counselor_output: bool
) -> str:
    append_to_log(
        log_file_path, f"--- {counselor_obj.__class__.__name__} Start ({counselor_obj.model_name}) ---")

    if shield_moderator:
        user_probe_assessment: Optional[ShieldGemmaResponse] = shield_moderator.moderate_text(
            test_probe_content, ShieldUseCase.PROMPT_ONLY)
        append_to_log(
            log_file_path, f"User Probe ShieldGemma Assessment: {user_probe_assessment.model_dump_json(indent=2) if user_probe_assessment else 'ShieldGemma call failed / skipped'}")
        if user_probe_assessment and user_probe_assessment.is_self_harm_detected:
            append_to_log(
                log_file_path, "SELF-HARM DETECTED IN USER PROBE. Overriding response with escalation message.")
            append_to_log(
                log_file_path, f"--- {counselor_obj.__class__.__name__} End (Escalation) ---\n")
            return DEFAULT_ESCALATION_MESSAGE
    else:
        append_to_log(
            log_file_path, "ShieldGemma moderator not available. Skipping user probe moderation.")

    system_prompt, prompt_content_main = counselor_obj._construct_llm_prompt(  # Call instance method
        context_for_llm, test_probe_content)

    append_to_log(
        log_file_path, f"\n--- LLM Call Input ({counselor_obj.model_name}) ---")
    append_to_log(log_file_path, f"System Prompt: {system_prompt}")
    append_to_log(
        log_file_path, f"User/Main Prompt Content (excerpt):\n{prompt_content_main[:500]}...")

    raw_or_json_response = call_api_client(
        prompt_content_main, CounselorResponse, model=counselor_obj.model_name,
        system_prompt=system_prompt, expect_json_object=expect_json_counselor_output
    )

    counselor_generated_text = "Error: Could not generate response."
    if expect_json_counselor_output:  # Should always be False for counselors now
        if raw_or_json_response and isinstance(raw_or_json_response, CounselorResponse):
            counselor_generated_text = raw_or_json_response.response
    elif raw_or_json_response and isinstance(raw_or_json_response, str):
        counselor_generated_text = raw_or_json_response.strip()

    append_to_log(
        log_file_path, f"Initial Model Output: {counselor_generated_text}")

    final_response = counselor_generated_text
    if shield_moderator:
        model_output_assessment: Optional[ShieldGemmaResponse] = shield_moderator.moderate_text(
            counselor_generated_text, ShieldUseCase.PROMPT_RESPONSE, original_user_prompt=test_probe_content
        )
        append_to_log(
            log_file_path, f"Model Output ShieldGemma Assessment: {model_output_assessment.model_dump_json(indent=2) if model_output_assessment else 'ShieldGemma call failed / skipped'}")
        if model_output_assessment and model_output_assessment.is_generally_unsafe_detected:
            append_to_log(
                log_file_path, "HARMFUL/UNSAFE ADVICE DETECTED IN MODEL OUTPUT. Overriding with safe fallback.")
            final_response = DEFAULT_SAFE_FALLBACK_MODEL_OUTPUT
    else:
        append_to_log(
            log_file_path, "ShieldGemma moderator not available. Skipping model output moderation.")

    append_to_log(
        log_file_path, f"\n--- Final Response to User (after safety checks) ---")
    append_to_log(log_file_path, f"Final Response: {final_response}")
    append_to_log(
        log_file_path, f"--- {counselor_obj.__class__.__name__} End ---\n")
    return final_response

# --- Counselor Implementations ---


class LocalAdaptiveCounselor(BaseCounselor):
    def _prepare_context(self, user_id: str, test_probe: str, log_file_path: str) -> str:
        context = "No clinical summary available for context."
        if self.memory:
            uid_for_search = f"{user_id}{self.user_id_track_suffix}"
            filters = {"type": "srs_summary"}
            append_to_log(
                log_file_path, f"Mem0 Search for SRS: UserID='{uid_for_search}', Query='Retrieve latest summary', Filters={filters}")
            resp = self.memory.search(
                query="Retrieve latest clinical summary", user_id=uid_for_search, filters=filters, limit=1)
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

    def get_response(self, user_id: str, case_data: Dict, test_probe_content: str, log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator]) -> str:
        context = self._prepare_context(
            user_id, test_probe_content, log_file_path)
        return _generate_and_moderate_response(self, user_id, case_data, test_probe_content, log_file_path, shield_moderator, context, expect_json_counselor_output=False)


class ClosedAdaptiveCounselor(LocalAdaptiveCounselor):
    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor (large model). Use the detailed clinical summary (focus, state, milestones, themes, plan, quotes, reflections, focus points, safety alerts) to respond, applying the 'plan_for_next_session'."
        prompt_content = f"CLINICAL SUMMARY:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content
    # get_response is inherited and will correctly call _generate_and_moderate_response


class BaselineRAGCounselor(BaseCounselor):
    def _prepare_context(self, user_id: str, test_probe: str, log_file_path: str) -> str:
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

    def get_response(self, user_id: str, case_data: Dict, test_probe_content: str, log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator]) -> str:
        context = self._prepare_context(
            user_id, test_probe_content, log_file_path)
        return _generate_and_moderate_response(self, user_id, case_data, test_probe_content, log_file_path, shield_moderator, context, expect_json_counselor_output=False)


class LocalBaselineCounselor(BaselineRAGCounselor):
    pass


class ClosedBaselineCounselor(BaselineRAGCounselor):
    pass


class NoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str): super().__init__(model_name, None, "")

    def _prepare_context(self, case_data: Dict, log_file_path: str) -> str:
        # This is defined by subclasses LocalBaselineNoMemoryCounselor and ClosedBaselineNoMemoryCounselor
        raise NotImplementedError(
            "Subclasses must implement _prepare_context for NoMemoryCounselor")

    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor."
        prompt_content = f"CONTEXT:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content

    def get_response(self, user_id: str, case_data: Dict, test_probe_content: str, log_file_path: str, shield_moderator: Optional[ShieldGemmaModerator]) -> str:
        context = self._prepare_context(
            case_data, log_file_path)  # type: ignore
        return _generate_and_moderate_response(self, user_id, case_data, test_probe_content, log_file_path, shield_moderator, context, expect_json_counselor_output=False)


class LocalBaselineNoMemoryCounselor(NoMemoryCounselor):
    def _prepare_context(self, case_data: Dict, log_file_path: str) -> str:
        context = "\n".join(
            [f"{t['role']}: {t['content']}" for t in case_data['sessions'][4]['transcript']])
        append_to_log(
            log_file_path, f"Context (Session 5 Transcript):\n{context[:500]}...")
        return context

    # Specific override for prompt name
    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor."
        prompt_content = f"PREVIOUS SESSION CONTEXT:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content


class ClosedBaselineNoMemoryCounselor(NoMemoryCounselor):
    def _prepare_context(self, case_data: Dict, log_file_path: str) -> str:
        context = "\n\n".join([f"--- Session {s['session_number']} ---\n" + "\n".join(
            [f"{t['role']}: {t['content']}" for t in s['transcript']]) for s in case_data['sessions'][:5]])
        append_to_log(
            log_file_path, f"Context (Full Transcript S1-5):\n{context[:1000]}...")
        return context

    # Specific override for prompt name
    def _construct_llm_prompt(self, context: str, test_probe: str) -> tuple[str, str]:
        system_prompt = "You are an empathetic AI counselor. Based on the entire conversation history, respond to the user's message."
        prompt_content = f"HISTORY:\n{context}\n\nUSER MESSAGE:\n{test_probe}\n\nYOUR RESPONSE (direct text):"
        return system_prompt, prompt_content
