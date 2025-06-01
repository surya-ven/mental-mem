# counselor_models.py

import json
from mem0 import Memory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

import config
# call_api_client is used for OpenRouter models
from llm_utils import call_api_client

# Schemas (ClinicalSummary, CounselorResponse) remain the same


class ClinicalSummary(BaseModel):
    session_focus: str = Field(
        description="The main topic or focus of this session.")
    user_emotional_state: str = Field(
        description="The user's primary emotional state during the session.")
    therapeutic_milestones: List[str] = Field(
        description="Key insights, breakthroughs, or progress made by the user.")
    emerging_themes: List[str] = Field(
        description="Recurring topics or underlying themes noticed across sessions.")
    plan_for_next_session: str = Field(
        description="The concrete plan or task for the user to focus on before the next session.")


class CounselorResponse(BaseModel):
    response: str

# run_srs_reflector function remains the same


def run_srs_reflector(session_transcript: List[Dict], previous_summary: Optional[Dict] = None) -> Optional[ClinicalSummary]:
    transcript_text = "\n".join(
        [f"{t['role']}: {t['content']}" for t in session_transcript])
    prompt = f"""
    You are a supervising clinical psychologist. Your task is to distill a therapy session into a structured clinical summary.
    PREVIOUS SUMMARY (if any):
    {json.dumps(previous_summary, indent=2) if previous_summary else "N/A"}
    CURRENT SESSION TRANSCRIPT:
    {transcript_text}
    TASK:
    Review all information and generate a new Clinical Summary.
    Output a single, valid JSON object with the following EXACT top-level keys and structure:
    - "session_focus": (string) The main topic or focus of THIS session.
    - "user_emotional_state": (string) The user's primary emotional state observed during THIS session.
    - "therapeutic_milestones": (list of strings) Key insights, breakthroughs, or progress made by the user in THIS session.
    - "emerging_themes": (list of strings) Recurring topics or underlying themes noticed across sessions, considering the previous summary and the current session.
    - "plan_for_next_session": (string) The concrete plan or task agreed upon or implied at the end of THIS current session for the user to focus on next.
    Ensure your output strictly adheres to this flat JSON structure with these exact keys. Do NOT wrap it in any other parent key.
    """
    summary = call_api_client(prompt, ClinicalSummary,
                              model=config.REFLECTOR_MODEL)
    return summary

# MemoryManager remains the same


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
        metadata = {"type": "srs_summary", "session_number": session_num}
        # Both local_adaptive and closed_adaptive will read from user_id_adaptive memories
        self.memory.add(
            summary_json_string, user_id=f"{user_id}_adaptive", metadata=metadata, infer=False)

    def clear_user_history(self, user_id: str):
        # Ensure all relevant user_id tracks are cleared
        # For both adaptive models
        self.memory.delete_all(user_id=f"{user_id}_adaptive")
        self.memory.delete_all(user_id=f"{user_id}_baseline_local")
        self.memory.delete_all(user_id=f"{user_id}_baseline_closed")

# BaseCounselor remains the same


class BaseCounselor:
    def __init__(self, model_name: str, memory_instance: Optional[Memory], user_id_track_suffix: str):
        self.model_name = model_name
        self.memory = memory_instance
        self.user_id_track_suffix = user_id_track_suffix

    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        raise NotImplementedError

# --- Counselor model implementations ---


class LocalAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        latest_summary_text = "No clinical summary available for context."
        if self.memory:
            # user_id_track_suffix for adaptive models is "_adaptive"
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            srs_metadata_filter = {"type": "srs_summary"}

            memories_response = self.memory.search(
                query=test_probe,
                user_id=user_id_for_search,
                filters=srs_metadata_filter,
                limit=1
            )
            memories_result = memories_response.get('results', [])
            if memories_result:
                try:
                    summary_data = json.loads(memories_result[0]['memory'])
                    if isinstance(summary_data, dict):
                        plan = summary_data.get(
                            "plan_for_next_session", "Continue exploring feelings.")
                        latest_summary_text = f"Focus: {summary_data.get('session_focus', 'N/A')}. Emotional State: {summary_data.get('user_emotional_state', 'N/A')}. Plan: {plan}"
                    else:
                        latest_summary_text = memories_result[0]['memory']
                except json.JSONDecodeError:
                    latest_summary_text = memories_result[0]['memory']

        system_prompt = "You are an empathetic AI counselor. Use the provided clinical summary to inform your response. Your main goal is to address the user's situation by applying the 'plan_for_next_session' from the summary."
        prompt_content = f"CLINICAL SUMMARY CONTEXT:\n{latest_summary_text}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        response_obj = call_api_client(  # Uses call_api_client for OpenRouter
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        return response_obj.response if response_obj else "Error: Could not generate response."

# --- NEW: ClosedAdaptiveCounselor ---


class ClosedAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        latest_summary_text = "No clinical summary available for context."
        if self.memory:
            # It uses the same "_adaptive" track for summaries as LocalAdaptiveCounselor
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            srs_metadata_filter = {"type": "srs_summary"}

            memories_response = self.memory.search(
                query=test_probe,
                user_id=user_id_for_search,
                filters=srs_metadata_filter,
                limit=1
            )
            memories_result = memories_response.get('results', [])
            if memories_result:
                try:
                    summary_data = json.loads(memories_result[0]['memory'])
                    if isinstance(summary_data, dict):
                        plan = summary_data.get(
                            "plan_for_next_session", "Continue exploring feelings.")
                        latest_summary_text = f"Focus: {summary_data.get('session_focus', 'N/A')}. Emotional State: {summary_data.get('user_emotional_state', 'N/A')}. Plan: {plan}"
                    else:
                        latest_summary_text = memories_result[0]['memory']
                except json.JSONDecodeError:
                    latest_summary_text = memories_result[0]['memory']

        system_prompt = "You are an empathetic AI counselor (using a large language model). Use the provided clinical summary to inform your response. Your main goal is to address the user's situation by applying the 'plan_for_next_session' from the summary."
        prompt_content = f"CLINICAL SUMMARY CONTEXT:\n{latest_summary_text}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        response_obj = call_api_client(  # Uses call_api_client as it's a closed model via OpenRouter
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        return response_obj.response if response_obj else "Error: Could not generate response."


class LocalBaselineCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        memory_context = "No relevant past conversation snippets found."
        if self.memory:
            memories_response = self.memory.search(
                query=test_probe,
                user_id=f"{user_id}{self.user_id_track_suffix}",
                limit=5
            )
            memories_result = memories_response.get('results', [])
            if memories_result:
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets from past sessions to inform your response."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."
        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        return response_obj.response if response_obj else "Error: Could not generate response."


class LocalBaselineNoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str):  # Does not need memory_instance or suffix
        super().__init__(model_name, None, "")

    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        last_session_transcript = "\n".join(
            [f"{t['role']}: {t['content']}" for t in case_data['sessions'][4]['transcript']])
        system_prompt = "You are an empathetic AI counselor."
        prompt_content = f"PREVIOUS SESSION CONTEXT:\n{last_session_transcript}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."
        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        return response_obj.response if response_obj else "Error: Could not generate response."


class ClosedBaselineNoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str):  # Does not need memory_instance or suffix
        super().__init__(model_name, None, "")

    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        full_history = "\n\n".join([f"--- Session {s['session_number']} ---\n" + "\n".join(
            [f"{t['role']}: {t['content']}" for t in s['transcript']]) for s in case_data['sessions'][:5]])
        prompt_content = f"You are an empathetic AI counselor. Based on the entire conversation history below, respond to the user's final message.\n\nHISTORY:\n{full_history}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."
        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name)
        return response_obj.response if response_obj else "Error: Could not generate response."


class ClosedBaselineCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str) -> str:
        memory_context = "No relevant past conversation snippets found."
        if self.memory:
            memories_response = self.memory.search(
                query=test_probe,
                user_id=f"{user_id}{self.user_id_track_suffix}",
                limit=5
            )
            memories_result = memories_response.get('results', [])
            if memories_result:
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets to respond to the user's latest message."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."
        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        return response_obj.response if response_obj else "Error: Could not generate response."
