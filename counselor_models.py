# counselor_models.py

import json
from mem0 import Memory
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

import config
from llm_utils import call_api_client

# --- Helper function for logging (ensure it's defined or imported if you use it elsewhere) ---


def append_to_log(log_file_path: str, message: str):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# --- UPDATED Schemas for our Enhanced SRS Method ---


class ClinicalSummary(BaseModel):
    session_focus: str = Field(
        description="The main topic or focus of this session.")
    user_emotional_state: str = Field(
        description="The user's primary emotional state during the session.")
    therapeutic_milestones: List[str] = Field(
        default_factory=list, description="Key insights, breakthroughs, or progress made by the user.")
    emerging_themes: List[str] = Field(
        default_factory=list, description="Recurring topics or underlying themes noticed across sessions.")
    plan_for_next_session: str = Field(
        description="The concrete plan or task for the user to focus on before the next session.")
    # --- NEW ENHANCED FIELDS ---
    key_user_quotes_from_session: List[str] = Field(
        default_factory=list, description="Short, impactful verbatim quotes from the user during this session that highlight key moments or feelings.")
    counselor_reflections_on_session: str = Field(
        description="Brief (1-2 sentences) psychologist-level reflections on the session's dynamics, user's progress, potential stuck points, or areas for future exploration.")
    suggested_focus_points_for_probe: List[str] = Field(
        default_factory=list, description="Based on the entire summary, list 1-3 specific themes, past events, or user statements the responding counselor should consider or link to when addressing the next user probe.")


class CounselorResponse(BaseModel):
    response: str

# --- UPDATED "Reflector" to generate enhanced summary ---


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
    Output a single, valid JSON object with the following EXACT top-level keys and structure:
    - "session_focus": (string) The main topic or focus of THIS current session.
    - "user_emotional_state": (string) The user's primary emotional state observed during THIS current session.
    - "therapeutic_milestones": (list of strings) Key insights, breakthroughs, or progress made by the user in THIS current session.
    - "emerging_themes": (list of strings) Recurring topics or underlying themes. Consider both the current session and previous summary.
    - "plan_for_next_session": (string) The concrete plan or task agreed upon or implied at the end of THIS current session.
    - "key_user_quotes_from_session": (list of strings) Identify 1-3 short, verbatim quotes from the USER in the CURRENT session that are particularly impactful or revealing.
    - "counselor_reflections_on_session": (string) Provide 1-2 sentences of your own clinical reflections on this current session's dynamics, progress, or stuck points.
    - "suggested_focus_points_for_probe": (list of strings) Based on everything, list 1-3 specific concepts, themes, or past statements the next counselor should ideally link to or consider when responding to a new problem from the user.

    Ensure your output strictly adheres to this flat JSON structure with these exact keys.
    """
    summary = call_api_client(prompt, ClinicalSummary,
                              model=config.REFLECTOR_MODEL)
    return summary

# --- MemoryManager: Updated to pass ClinicalSummary object to add_srs_summary_for_adaptive ---


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

    # Takes ClinicalSummary object
    def add_srs_summary_for_adaptive(self, user_id: str, summary: ClinicalSummary, session_num: int):
        summary_json_string = summary.model_dump_json()  # Dumps the enhanced summary
        metadata = {"type": "srs_summary", "session_number": session_num}
        self.memory.add(
            summary_json_string, user_id=f"{user_id}_adaptive", metadata=metadata, infer=False)

    def clear_user_history(self, user_id: str):
        self.memory.delete_all(user_id=f"{user_id}_adaptive")
        self.memory.delete_all(user_id=f"{user_id}_baseline_local")
        self.memory.delete_all(user_id=f"{user_id}_baseline_closed")

# BaseCounselor remains the same


class BaseCounselor:
    def __init__(self, model_name: str, memory_instance: Optional[Memory], user_id_track_suffix: str):
        self.model_name = model_name
        self.memory = memory_instance
        self.user_id_track_suffix = user_id_track_suffix

    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        raise NotImplementedError

# --- UPDATED LocalAdaptiveCounselor to use enhanced summary ---


class LocalAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")

        latest_summary_obj: Optional[ClinicalSummary] = None
        latest_summary_text_for_prompt = "No clinical summary available for context."
        srs_metadata_filter = {"type": "srs_summary"}
        user_id_for_search = f"{user_id}{self.user_id_track_suffix}"

        if self.memory:
            append_to_log(
                log_file_path, f"\n--- Mem0 Search Call for SRS Summary ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            # Query might be less important if we fetch latest by metadata
            append_to_log(log_file_path, f"Query: {test_probe}")
            append_to_log(
                log_file_path, f"Filters: {json.dumps(srs_metadata_filter)}")

            memories_response = self.memory.search(
                # Generic query, relying on limit and potential recency
                query="Retrieve latest clinical summary",
                user_id=user_id_for_search,
                filters=srs_metadata_filter,
                limit=1  # Assuming the latest added is the most relevant or we sort if mem0 supports it
            )
            append_to_log(
                log_file_path, f"Mem0 Raw SRS Response: {json.dumps(memories_response, indent=2)}")

            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} summary memory.")
                try:
                    # The memory is stored as a JSON string of ClinicalSummary
                    summary_data_dict = json.loads(
                        memories_result[0]['memory'])
                    latest_summary_obj = ClinicalSummary.model_validate(
                        summary_data_dict)  # Validate and parse

                    # Construct a detailed text representation for the prompt
                    latest_summary_text_for_prompt = f"Overall Focus: {latest_summary_obj.session_focus}\n"
                    latest_summary_text_for_prompt += f"User's Emotional State: {latest_summary_obj.user_emotional_state}\n"
                    latest_summary_text_for_prompt += f"Therapeutic Milestones: {', '.join(latest_summary_obj.therapeutic_milestones)}\n"
                    latest_summary_text_for_prompt += f"Emerging Themes: {', '.join(latest_summary_obj.emerging_themes)}\n"
                    latest_summary_text_for_prompt += f"Key User Quotes: {json.dumps(latest_summary_obj.key_user_quotes_from_session)}\n"
                    latest_summary_text_for_prompt += f"Counselor Reflections: {latest_summary_obj.counselor_reflections_on_session}\n"
                    latest_summary_text_for_prompt += f"Suggested Focus Points for this Probe: {json.dumps(latest_summary_obj.suggested_focus_points_for_probe)}\n"
                    latest_summary_text_for_prompt += f"Agreed Plan for Next Session: {latest_summary_obj.plan_for_next_session}"
                    append_to_log(
                        log_file_path, f"Parsed & Formatted Clinical Summary for Prompt:\n{latest_summary_text_for_prompt}")
                except Exception as e:  # Catch JSONDecodeError or PydanticValidationError
                    latest_summary_text_for_prompt = f"Error parsing summary: {memories_result[0]['memory']}. Error: {e}"
                    append_to_log(
                        log_file_path, f"Failed to parse summary string: {e}")
            else:
                append_to_log(log_file_path, "No SRS summary found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        # --- UPDATED System Prompt & Prompt Content ---
        system_prompt = "You are an empathetic and insightful AI counselor. You MUST use the detailed clinical summary provided to inform your response. Your goal is to address the user's current message by applying the 'plan_for_next_session' and considering the 'suggested_focus_points_for_probe' from the summary."
        prompt_content = f"**DETAILED CLINICAL SUMMARY CONTEXT (FROM PREVIOUS SESSIONS):**\n{latest_summary_text_for_prompt}\n\n**USER'S CURRENT MESSAGE (TEST PROBE):**\n{test_probe}\n\n**YOUR TASK:**\nBased on the clinical summary (especially the 'plan_for_next_session' and 'suggested_focus_points_for_probe'), and the user's current message, provide a therapeutic response. Make sure your response is congruent with the insights from the summary. Output your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response

# --- ClosedAdaptiveCounselor: Updated to use ENHANCED summary ---


class ClosedAdaptiveCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")

        latest_summary_obj: Optional[ClinicalSummary] = None
        latest_summary_text_for_prompt = "No clinical summary available for context."
        srs_metadata_filter = {"type": "srs_summary"}
        # Should be "_adaptive"
        user_id_for_search = f"{user_id}{self.user_id_track_suffix}"

        if self.memory:
            append_to_log(
                log_file_path, f"\n--- Mem0 Search Call for SRS Summary ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(log_file_path, f"Query: {test_probe}")
            append_to_log(
                log_file_path, f"Filters: {json.dumps(srs_metadata_filter)}")

            memories_response = self.memory.search(
                query="Retrieve latest clinical summary",
                user_id=user_id_for_search,
                filters=srs_metadata_filter,
                limit=1
            )
            append_to_log(
                log_file_path, f"Mem0 Raw SRS Response: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} summary memory.")
                try:
                    summary_data_dict = json.loads(
                        memories_result[0]['memory'])
                    latest_summary_obj = ClinicalSummary.model_validate(
                        summary_data_dict)

                    latest_summary_text_for_prompt = f"Overall Focus: {latest_summary_obj.session_focus}\n"
                    latest_summary_text_for_prompt += f"User's Emotional State: {latest_summary_obj.user_emotional_state}\n"
                    latest_summary_text_for_prompt += f"Therapeutic Milestones: {', '.join(latest_summary_obj.therapeutic_milestones)}\n"
                    latest_summary_text_for_prompt += f"Emerging Themes: {', '.join(latest_summary_obj.emerging_themes)}\n"
                    latest_summary_text_for_prompt += f"Key User Quotes: {json.dumps(latest_summary_obj.key_user_quotes_from_session)}\n"
                    latest_summary_text_for_prompt += f"Counselor Reflections: {latest_summary_obj.counselor_reflections_on_session}\n"
                    latest_summary_text_for_prompt += f"Suggested Focus Points for this Probe: {json.dumps(latest_summary_obj.suggested_focus_points_for_probe)}\n"
                    latest_summary_text_for_prompt += f"Agreed Plan for Next Session: {latest_summary_obj.plan_for_next_session}"
                    append_to_log(
                        log_file_path, f"Parsed & Formatted Clinical Summary for Prompt:\n{latest_summary_text_for_prompt}")
                except Exception as e:
                    latest_summary_text_for_prompt = f"Error parsing summary: {memories_result[0]['memory']}. Error: {e}"
                    append_to_log(
                        log_file_path, f"Failed to parse summary string: {e}")
            else:
                append_to_log(log_file_path, "No SRS summary found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic and insightful AI counselor (using a large language model). You MUST use the detailed clinical summary provided to inform your response. Your goal is to address the user's current message by applying the 'plan_for_next_session' and considering the 'suggested_focus_points_for_probe' from the summary."
        prompt_content = f"**DETAILED CLINICAL SUMMARY CONTEXT (FROM PREVIOUS SESSIONS):**\n{latest_summary_text_for_prompt}\n\n**USER'S CURRENT MESSAGE (TEST PROBE):**\n{test_probe}\n\n**YOUR TASK:**\nBased on the clinical summary (especially the 'plan_for_next_session' and 'suggested_focus_points_for_probe'), and the user's current message, provide a therapeutic response. Make sure your response is congruent with the insights from the summary. Output your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response

# LocalBaselineCounselor, LocalBaselineNoMemoryCounselor, ClosedBaselineNoMemoryCounselor, ClosedBaselineCounselor
# remain unchanged as they do not use the enhanced ClinicalSummary fields.


class LocalBaselineCounselor(BaseCounselor):
    def get_response(self, user_id: str, case_data: Dict, test_probe: str, log_file_path: str) -> str:
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} Start ---")
        append_to_log(
            log_file_path, f"Model: {self.model_name}, User ID Track: {user_id}{self.user_id_track_suffix}")

        memory_context = "No relevant past conversation snippets found."
        user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
        if self.memory:
            append_to_log(log_file_path, f"\n--- Mem0 Search Call ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(log_file_path, f"Query: {test_probe}")

            memories_response = self.memory.search(
                query=test_probe,
                user_id=user_id_for_search,
                limit=5
            )
            append_to_log(
                log_file_path, f"Mem0 Raw Response: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} raw memories.")
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
                append_to_log(
                    log_file_path, f"Extracted Context for Prompt:\n{memory_context}")
            else:
                append_to_log(log_file_path, "No raw memories found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets from past sessions to inform your response."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
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
        prompt_content = f"PREVIOUS SESSION CONTEXT:\n{last_session_transcript}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
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

        prompt_content = f"You are an empathetic AI counselor. Based on the entire conversation history below, respond to the user's final message.\n\nHISTORY:\n{full_history}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(
            log_file_path, f"User/Main Prompt Content (History part omitted from log for brevity if long):\n{prompt_content.split('HISTORY:')[0]}USER'S LATEST MESSAGE:\n{test_probe}...")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
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
        user_id_for_search = f"{user_id}{self.user_id_track_suffix}"
        if self.memory:
            append_to_log(log_file_path, f"\n--- Mem0 Search Call ---")
            append_to_log(
                log_file_path, f"User ID for Search: {user_id_for_search}")
            append_to_log(log_file_path, f"Query: {test_probe}")

            memories_response = self.memory.search(
                query=test_probe,
                user_id=user_id_for_search,
                limit=5
            )
            append_to_log(
                log_file_path, f"Mem0 Raw Response: {json.dumps(memories_response, indent=2)}")
            memories_result = memories_response.get('results', [])
            if memories_result:
                append_to_log(
                    log_file_path, f"Found {len(memories_result)} raw memories.")
                memory_context = "\n".join(
                    [m['memory'] for m in memories_result])
                append_to_log(
                    log_file_path, f"Extracted Context for Prompt:\n{memory_context}")
            else:
                append_to_log(log_file_path, "No raw memories found in mem0.")
        else:
            append_to_log(log_file_path, "No memory instance provided.")

        system_prompt = "You are an empathetic AI counselor. Use the provided conversation snippets to respond to the user's latest message."
        prompt_content = f"PAST CONVERSATION SNIPPETS:\n{memory_context}\n\nUSER'S LATEST MESSAGE:\n{test_probe}\n\nProvide your response as a JSON object with one key: 'response'."

        append_to_log(log_file_path, f"\n--- LLM Call Input ---")
        append_to_log(log_file_path, f"System Prompt: {system_prompt}")
        append_to_log(
            log_file_path, f"User/Main Prompt Content:\n{prompt_content}")

        response_obj = call_api_client(
            prompt_content, CounselorResponse, model=self.model_name, system_prompt=system_prompt)
        final_response = response_obj.response if response_obj else "Error: Could not generate response."
        append_to_log(log_file_path, f"\n--- LLM Call Output ---")
        append_to_log(log_file_path, f"Generated Response: {final_response}")
        append_to_log(
            log_file_path, f"--- {self.__class__.__name__} End ---\n")
        return final_response
