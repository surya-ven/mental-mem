# counselor_models.py

import json
from mem0 import Memory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

import config
from llm_utils import call_api_client

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
        default_factory=list, description="Short, impactful verbatim quotes from the user during this session. Must be a list of strings.")
    counselor_reflections_on_session: str = Field(
        description="Brief (1-2 sentences) psychologist-level reflections on the session's dynamics, user's progress, potential stuck points, or areas for future exploration.")
    suggested_focus_points_for_probe: List[str] = Field(
        default_factory=list, description="Based on the entire summary, list 1-3 specific themes, past events, or user statements the responding counselor should consider. Must be a list of strings.")


class CounselorResponse(BaseModel):
    response: str

# --- UPDATED "Reflector" prompt for more explicit list formatting instructions ---


def run_srs_reflector(session_transcript: List[Dict], previous_summary_obj: Optional[ClinicalSummary] = None) -> Optional[ClinicalSummary]:
    transcript_text = "\n".join(
        [f"{t['role']}: {t['content']}" for t in session_transcript])
    previous_summary_text = previous_summary_obj.model_dump_json(
        indent=2) if previous_summary_obj else "N/A"

    prompt = f"""
    You are a supervising clinical psychologist. Your task is to distill a therapy session into an enhanced, structured clinical summary.

    PREVIOUS CLINICAL SUMMARY (if any):
    {previous_summary_text}

    CURRENT SESSION TRANSCRIPT:
    {transcript_text}

    TASK:
    Review all information and generate a new, ENHANCED Clinical Summary.
    Output a single, valid JSON object with the following EXACT top-level keys and structure.
    Pay close attention to fields requiring a LIST OF STRINGS.

    - "session_focus": (string) The main topic or focus of THIS current session.
    - "user_emotional_state": (string) The user's primary emotional state observed during THIS current session.
    - "therapeutic_milestones": (LIST OF STRINGS) Key insights, breakthroughs, or progress made by the user in THIS current session. 
        This MUST be a JSON array of strings. 
        Example for one milestone: ["User identified a core belief."]. 
        Example for multiple: ["User practiced a new coping skill.", "User expressed a difficult emotion openly."].
        If no specific milestones, provide an empty list: [].
    - "emerging_themes": (LIST OF STRINGS) Recurring topics or underlying themes. Consider both the current session and previous summary. 
        This MUST be a JSON array of strings.
        Example: ["Difficulty with trust in relationships.", "Pattern of avoidance."].
        If no specific themes, provide an empty list: [].
    - "plan_for_next_session": (string) The concrete plan or task agreed upon or implied at the end of THIS current session.
    - "key_user_quotes_from_session": (LIST OF STRINGS) Identify 1-3 short, verbatim quotes from the USER in the CURRENT session that are particularly impactful or revealing. 
        This MUST be a JSON array of strings.
        Example: ["I finally feel understood.", "That's a perspective I hadn't considered."].
        If no specific quotes, provide an empty list: [].
    - "counselor_reflections_on_session": (string) Provide 1-2 sentences of your own clinical reflections on this current session's dynamics, progress, or stuck points.
    - "suggested_focus_points_for_probe": (LIST OF STRINGS) Based on everything, list 1-3 specific concepts, themes, or past statements the next counselor should ideally link to or consider when responding to a new problem from the user.
        This MUST be a JSON array of strings.
        Example: ["Explore the user's hesitation around vulnerability.", "Link current anxiety to previously discussed family dynamics."].
        If no specific focus points, provide an empty list: [].

    Ensure your output strictly adheres to this flat JSON structure with these exact keys and specified types (string or LIST OF STRINGS).
    For fields specified as LIST OF STRINGS, always provide a JSON array, even if it's empty [] or contains a single string ["single item"]. Do not provide a single string value for these fields.
    """
    summary = call_api_client(prompt, ClinicalSummary,
                              model=config.REFLECTOR_MODEL, expect_json_object=True)
    return summary

# MemoryManager (no changes needed from the version that uses MEM0_CONFIG)


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
        self.memory.add(
            summary_json_string, user_id=f"{user_id}_adaptive", metadata=metadata, infer=False)

    def clear_user_history(self, user_id: str):
        self.memory.delete_all(user_id=f"{user_id}_adaptive")
        self.memory.delete_all(user_id=f"{user_id}_baseline_local")
        self.memory.delete_all(user_id=f"{user_id}_baseline_closed")

# BaseCounselor (no changes needed)


class BaseCounselor:
    def __init__(self, model_name: str, memory_instance: Optional[Memory], user_id_track_suffix: str):
        self.model_name = model_name
        self.memory = memory_instance
        self.user_id_track_suffix = user_id_track_suffix

    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        raise NotImplementedError

# All Counselor subclasses (LocalAdaptiveCounselor, ClosedAdaptiveCounselor, etc.) remain unchanged
# as the issue is with the data generated by run_srs_reflector, which is then fed into them.
# The logic within these classes for handling the ClinicalSummary (once correctly parsed) is fine.
# (Ensure you have the latest versions of these classes that ask for raw text from local models
# and also from closed models as per your last request for fairness)


class LocalAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")
        latest_summary_text_for_prompt = "No clinical summary available for context."
        if self.memory:
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            srs_metadata_filter = {"type": "srs_summary"}
            append_to_log(
                log_file_path, f"\n--- Mem0 Search Call for SRS Summary ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(
                log_file_path, f"Query: 'Retrieve latest clinical summary' (for adaptive retrieval)")
            append_to_log(
                log_file_path, f"Filters: {json.dumps(srs_metadata_filter)}")
            memories_response = self.memory.search(
                query="Retrieve latest clinical summary",
                user_id=user_id_for_search,
                filters=srs_metadata_filter, limit=1
            )
            append_to_log(
                log_file_path, f"Mem0 Raw SRS Response for Adaptive: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} summary memory.")
                try:
                    summary_data_dict = json.loads(
                        memories_result[0]['memory'])
                    latest_summary_obj = ClinicalSummary.model_validate(
                        summary_data_dict)
                    latest_summary_text_for_prompt = (
                        f"Overall Focus: {latest_summary_obj.session_focus}\n"
                        f"User's Emotional State: {latest_summary_obj.user_emotional_state}\n"
                        f"Therapeutic Milestones: {json.dumps(latest_summary_obj.therapeutic_milestones)}\n"
                        f"Emerging Themes: {json.dumps(latest_summary_obj.emerging_themes)}\n"
                        f"Key User Quotes: {json.dumps(latest_summary_obj.key_user_quotes_from_session)}\n"
                        f"Counselor Reflections: {latest_summary_obj.counselor_reflections_on_session}\n"
                        f"Suggested Focus Points for this Probe: {json.dumps(latest_summary_obj.suggested_focus_points_for_probe)}\n"
                        f"Agreed Plan for Next Session: {latest_summary_obj.plan_for_next_session}"
                    )
                except Exception as e:
                    latest_summary_text_for_prompt = f"Error parsing summary: {memories_result[0]['memory']}. Error: {e}"
                append_to_log(
                    log_file_path, f"Formatted Clinical Summary for Prompt:\n{latest_summary_text_for_prompt}")
            else:
                append_to_log(log_file_path, "No SRS summary found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic and insightful AI counselor. You MUST use the detailed clinical summary provided to inform your response. Your goal is to address the user's current message by applying the 'plan_for_next_session' and considering the 'suggested_focus_points_for_probe' from the summary."
        prompt_content = f"**DETAILED CLINICAL SUMMARY CONTEXT (FROM PREVIOUS SESSIONS):**\n{latest_summary_text_for_prompt}\n\n**USER'S CURRENT MESSAGE (TEST PROBE):**\n{test_probe}\n\n**YOUR TASK:**\nBased on the clinical summary (especially the 'plan_for_next_session' and 'suggested_focus_points_for_probe'), and the user's current message, provide a therapeutic response. Make sure your response is congruent with the insights from the summary. Generate ONLY the response text directly."

        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content (excerpt):\n{prompt_content[:500]}...")

        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            system_prompt=system_prompt, expect_json_object=False
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."

        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response


class ClosedAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")
        latest_summary_text_for_prompt = "No clinical summary available for context."
        if self.memory:
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            srs_metadata_filter = {"type": "srs_summary"}
            append_to_log(
                log_file_path, f"\n--- Mem0 Search Call for SRS Summary ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(
                log_file_path, f"Query: 'Retrieve latest clinical summary'")
            append_to_log(
                log_file_path, f"Filters: {json.dumps(srs_metadata_filter)}")
            memories_response = self.memory.search(
                query="Retrieve latest clinical summary", user_id=user_id_for_search,
                filters=srs_metadata_filter, limit=1)
            append_to_log(
                log_file_path, f"Mem0 Raw SRS Response for Adaptive: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} summary memory.")
                try:
                    summary_data_dict = json.loads(
                        memories_result[0]['memory'])
                    latest_summary_obj = ClinicalSummary.model_validate(
                        summary_data_dict)
                    latest_summary_text_for_prompt = (
                        f"Overall Focus: {latest_summary_obj.session_focus}\n"
                        f"User's Emotional State: {latest_summary_obj.user_emotional_state}\n"
                        f"Therapeutic Milestones: {json.dumps(latest_summary_obj.therapeutic_milestones)}\n"
                        f"Emerging Themes: {json.dumps(latest_summary_obj.emerging_themes)}\n"
                        f"Key User Quotes: {json.dumps(latest_summary_obj.key_user_quotes_from_session)}\n"
                        f"Counselor Reflections: {latest_summary_obj.counselor_reflections_on_session}\n"
                        f"Suggested Focus Points for this Probe: {json.dumps(latest_summary_obj.suggested_focus_points_for_probe)}\n"
                        f"Agreed Plan for Next Session: {latest_summary_obj.plan_for_next_session}"
                    )
                except Exception as e:
                    latest_summary_text_for_prompt = f"Error parsing summary: {memories_result[0]['memory']}. Error: {e}"
                append_to_log(
                    log_file_path, f"Formatted Clinical Summary for Prompt:\n{latest_summary_text_for_prompt}")
            else:
                append_to_log(log_file_path, "No SRS summary found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic and insightful AI counselor (using a large language model). You MUST use the detailed clinical summary provided to inform your response. Your goal is to address the user's current message by applying the 'plan_for_next_session' and considering the 'suggested_focus_points_for_probe' from the summary."
        prompt_content = f"**DETAILED CLINICAL SUMMARY CONTEXT (FROM PREVIOUS SESSIONS):**\n{latest_summary_text_for_prompt}\n\n**USER'S CURRENT MESSAGE (TEST PROBE):**\n{test_probe}\n\n**YOUR TASK:**\nBased on the clinical summary (especially the 'plan_for_next_session' and 'suggested_focus_points_for_probe'), and the user's current message, provide a therapeutic response. Make sure your response is congruent with the insights from the summary. Generate ONLY the response text directly."

        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content (excerpt):\n{prompt_content[:500]}...")

        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            system_prompt=system_prompt, expect_json_object=False  # Changed for fairness
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."

        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response


class LocalBaselineCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")
        memory_context = "No relevant past conversation snippets found."
        if self.memory:
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            append_to_log(log_file_path, f"\n--- Mem0 Search Call ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(log_file_path, f"Query: {test_probe}")
            memories_response = self.memory.search(
                query=test_probe, user_id=user_id_for_search, limit=5)
            append_to_log(
                log_file_path, f"Mem0 Raw RAG Response: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} raw memories.")
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
            append_to_log(
                log_file_path, f"Extracted RAG Context for Prompt:\n{memory_context}")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets from past sessions to inform your response."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nGenerate ONLY your therapeutic response text directly."
        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")
        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            system_prompt=system_prompt, expect_json_object=False
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."
        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response


class LocalBaselineNoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str):
        super().__init__(model_name, None, "")

    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(log_file_path, f"Model: {self.model_name}")
        last_session_transcript = "\n".join(
            [f"{t['role']}: {t['content']}" for t in case_data['sessions'][4]['transcript']])
        append_to_log(
            log_file_path, f"Context Used (Session 5 Transcript):\n{last_session_transcript[:500]}...")
        system_prompt = "You are an empathetic AI counselor."
        prompt_content = f"PREVIOUS SESSION CONTEXT:\n{last_session_transcript}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nGenerate ONLY your therapeutic response text directly."
        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")
        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            system_prompt=system_prompt, expect_json_object=False
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."
        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response


class ClosedBaselineNoMemoryCounselor(BaseCounselor):
    def __init__(self, model_name: str):
        super().__init__(model_name, None, "")

    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(log_file_path, f"Model: {self.model_name}")
        full_history = "\n\n".join([f"--- Session {s['session_number']} ---\n" + "\n".join(
            [f"{t['role']}: {t['content']}" for t in s['transcript']]) for s in case_data['sessions'][:5]])
        append_to_log(
            log_file_path, f"Context Used (Full Transcript Sessions 1-5):\n{full_history[:1000]}...")
        prompt_content = f"You are an empathetic AI counselor. Based on the entire conversation history below, respond to the user's final message.\n\nHISTORY:\n{full_history}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nGenerate ONLY your therapeutic response text directly."
        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        # Fixed prompt logging excerpt
        append_to_log(
            log_file_path, f"User/Main Prompt Content (excerpt):\n{prompt_content.split('USER')[1][:500]}...")
        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            expect_json_object=False
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."
        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response


class ClosedBaselineCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")
        memory_context = "No relevant past conversation snippets found."
        if self.memory:
            user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
            append_to_log(log_file_path, f"\n--- Mem0 Search Call ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(log_file_path, f"Query: {test_probe}")
            memories_response = self.memory.search(
                query=test_probe, user_id=user_id_for_search, limit=5)
            append_to_log(
                log_file_path, f"Mem0 Raw RAG Response: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} raw memories.")
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
            append_to_log(
                log_file_path, f"Extracted RAG Context for Prompt:\n{memory_context}")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets to respond to the user's latest message."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nGenerate ONLY your therapeutic response text directly."
        append_to_log(
            log_file_path, f"\n--- LLM Call Input ({self.model_name}) ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")
        raw_response_string = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name,
            system_prompt=system_prompt, expect_json_object=False
        )
        final_response = raw_response_string.strip() if raw_response_string and isinstance(
            raw_response_string, str) else "Error: Could not generate response."
        append_to_log(
            log_file_path, f"\n--- LLM Call Output ({self.model_name}) ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response
