# shield_gemma_utils.py

import enum
from typing import Sequence, Dict, Optional, List
import torch
import transformers  # Ensure this is installed
from pydantic import BaseModel, Field

import config

# --- Pydantic Models for ShieldGemma Output ---


class ShieldGemmaProbabilities(BaseModel):
    harassment: float = 0.0
    hate_speech: float = 0.0
    dangerous_content: float = 0.0
    sexually_explicit: float = 0.0


class ShieldGemmaResponse(BaseModel):
    text_analyzed: str
    probabilities: ShieldGemmaProbabilities
    is_self_harm_detected: bool = False
    is_hate_speech_detected: bool = False  # Added for completeness
    is_generally_unsafe_detected: bool = False

# --- Enums and Functions from ShieldGemma Notebook ---


class HarmType(enum.Enum):
    DANGEROUS = 'Dangerous Content'
    HATE = 'Hate Speech'
    SEXUAL = 'Sexually Explicit Information'
    HARASSMENT = 'Harassment'


class UseCase(enum.Enum):
    PROMPT_ONLY = 'prompt_only'
    PROMPT_RESPONSE = 'prompt_response'


HARM_POLICIES: dict[HarmType, str] = {
    HarmType.DANGEROUS: ('harming oneself and/or others (e.g., instructions for suicide, potentially contemplating suicide, subtle or direct suicidial ideation, feeling completely helpless or thinking everything is pointless).'),
    HarmType.HATE: ('targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.'),
    HarmType.SEXUAL: ('contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.'),
    HarmType.HARASSMENT: ('is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).'),
}

PROMPT_ONLY_TEMPLATE = """You are a policy expert trying to help determine
whether a user prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_content}
<end_of_turn>

Our safety principle is defined in the below:

* {harm_text}

===

Does the human question violate the above principle? Your answer must start with
'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.
"""

PROMPT_RESPONSE_TEMPLATE = """You are a policy expert trying to help determine
whether an AI response to prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_content}
<end_of_turn>

<start_of_turn>
Chatbot Response: {model_content}
<end_of_turn>

Our safety principle is defined in the below:

* {harm_text}

===

Does the Chatbot Response violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure  we answer
correctly.
"""


class ShieldGemmaModerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ShieldGemmaModerator, cls).__new__(cls)
            try:
                print("Initializing ShieldGemma model... This may take a moment.")
                cls._instance.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    config.SHIELD_GEMMA_MODEL_VARIANT)
                cls._instance.model = transformers.AutoModelForCausalLM.from_pretrained(
                    config.SHIELD_GEMMA_MODEL_VARIANT,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                cls._instance.softmax = torch.nn.Softmax(dim=0)
                cls._instance.YES_TOKEN_IDX = cls._instance.tokenizer.convert_tokens_to_ids(
                    "Yes")
                cls._instance.NO_TOKEN_IDX = cls._instance.tokenizer.convert_tokens_to_ids(
                    "No")
                print(
                    f"ShieldGemma model ({config.SHIELD_GEMMA_MODEL_VARIANT}) initialized successfully.")
            except Exception as e:
                print(f"FATAL ERROR initializing ShieldGemma: {e}")
                print("Ensure you have 'torch', 'transformers', 'sentencepiece' installed, and are logged into Hugging Face Hub if model is gated.")
                cls._instance = None
                raise
        return cls._instance

    def _harm_definition(self, harm_type: HarmType, use_case: UseCase) -> str:
        prefix = 'The prompt shall not contain or seek generation of content that' if use_case == UseCase.PROMPT_ONLY else 'The chatbot shall not generate content that'
        description = HARM_POLICIES[harm_type]
        return f'"No {harm_type.value}": {prefix} {description}'

    def _make_prompt(self, use_case: UseCase, harm_type: HarmType, user_content: str, model_content: Optional[str] = None) -> str:
        prompt_template = PROMPT_ONLY_TEMPLATE if use_case == UseCase.PROMPT_ONLY else PROMPT_RESPONSE_TEMPLATE
        if use_case == UseCase.PROMPT_RESPONSE and model_content is None:
            raise ValueError(
                'model_content is required for prompt-response mode.')

        formatter_args = {'user_content': user_content,
                          'harm_text': self._harm_definition(harm_type, use_case)}
        if model_content is not None:
            formatter_args['model_content'] = model_content
        return prompt_template.format(**formatter_args)

    def _preprocess_and_predict(self, prompt: str) -> Sequence[float]:
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Ensure YES_TOKEN_IDX and NO_TOKEN_IDX are not out of bounds for the model's vocab
        if self.YES_TOKEN_IDX >= logits.shape[-1] or self.NO_TOKEN_IDX >= logits.shape[-1]:
            print(
                f"Warning: YES_TOKEN_IDX ({self.YES_TOKEN_IDX}) or NO_TOKEN_IDX ({self.NO_TOKEN_IDX}) might be out of vocab size ({logits.shape[-1]}). This can happen if tokenizer and model are mismatched or 'Yes'/'No' are not standard tokens.")
            # Fallback or error handling strategy needed here. For now, returning neutral probability.
            return [0.5, 0.5]

        yes_no_logits = logits[0, -1, [self.YES_TOKEN_IDX, self.NO_TOKEN_IDX]]
        probabilities = self.softmax(yes_no_logits)

        return probabilities.to(torch.float32).cpu().numpy()

    def moderate_text(self, text_to_moderate: str, use_case: UseCase, original_user_prompt: Optional[str] = None) -> ShieldGemmaResponse:
        if not hasattr(self, 'model') or self.model is None:  # Check if initialization failed
            print(
                "ShieldGemmaModerator not properly initialized. Returning default unsafe.")
            return ShieldGemmaResponse(
                text_analyzed=text_to_moderate,
                probabilities=ShieldGemmaProbabilities(),
                is_generally_unsafe_detected=True
            )

        raw_probs: Dict[str, float] = {}

        harm_types_to_evaluate_map = {
            HarmType.DANGEROUS: 'dangerous_content',
            HarmType.HATE: 'hate_speech',
            HarmType.HARASSMENT: 'harassment',
            HarmType.SEXUAL: 'sexually_explicit'
        }

        # Determine which harm types to check based on use case
        if use_case == UseCase.PROMPT_ONLY:  # Primarily for self-harm in user input
            check_these_harms = [HarmType.DANGEROUS]
        else:  # For model output, check a broader set
            check_these_harms = [HarmType.DANGEROUS,
                                 HarmType.HATE, HarmType.HARASSMENT]
            # Add HarmType.SEXUAL if needed for model output checks

        for harm_type_enum in check_these_harms:
            shield_gemma_prompt = self._make_prompt(
                use_case,
                harm_type_enum,
                user_content=original_user_prompt if use_case == UseCase.PROMPT_RESPONSE else text_to_moderate,
                model_content=text_to_moderate if use_case == UseCase.PROMPT_RESPONSE else None
            )

            try:
                # probabilities from _preprocess_and_predict is [p_yes_violation, p_no_violation]
                prediction_output = self._preprocess_and_predict(
                    shield_gemma_prompt)
                # p_yes is probability of violation
                violation_prob = prediction_output[0]
                raw_probs[harm_types_to_evaluate_map[harm_type_enum]
                          ] = float(violation_prob)
            except Exception as e:
                print(
                    f"Error during ShieldGemma prediction for {harm_type_enum.value} on text '{text_to_moderate[:50]}...': {e}")
                # Default to not harmful on error for this category
                raw_probs[harm_types_to_evaluate_map[harm_type_enum]] = 0.0

        probabilities_obj = ShieldGemmaProbabilities(**raw_probs)

        is_self_harm = (use_case == UseCase.PROMPT_ONLY and
                        probabilities_obj.dangerous_content > config.SHIELD_GEMMA_SELF_HARM_THRESHOLD)

        is_unsafe_advice = (use_case == UseCase.PROMPT_RESPONSE and
                            (probabilities_obj.dangerous_content > config.SHIELD_GEMMA_DANGEROUS_CONTENT_THRESHOLD or
                             probabilities_obj.hate_speech > config.SHIELD_GEMMA_HATE_SPEECH_THRESHOLD or
                             probabilities_obj.harassment > config.SHIELD_GEMMA_HATE_SPEECH_THRESHOLD))

        is_hate = (probabilities_obj.hate_speech >
                   config.SHIELD_GEMMA_HATE_SPEECH_THRESHOLD)

        return ShieldGemmaResponse(
            text_analyzed=text_to_moderate,
            probabilities=probabilities_obj,
            is_self_harm_detected=is_self_harm,
            is_hate_speech_detected=is_hate,  # Store this specific flag
            is_generally_unsafe_detected=is_unsafe_advice
        )
