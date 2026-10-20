import json
import re

from src.llm.llm import LLM
from src.llm.prompt_builder import EntityLinkingPromptBuilder


class LLMEntityLinker:
    """
    Uses an Ollama LLM to select an entity from retrieved candidates.
    """

    def __init__(self, model_name="llama3.2:3b"):
        self.llm = LLM(model_name=model_name)
        self.prompt_builder = EntityLinkingPromptBuilder()

    def link(self, mention, context, candidates):
        """Select one retrieved candidate using structured JSON output."""

        if not candidates:
            return {
                "entity_id": "NIL",
                "reason": "No candidates were retrieved.",
                "raw_response": "",
                "valid_selection": True,
            }

        valid_ids = [
            str(candidate["entity_id"])
            for candidate in candidates
        ]

        prompt = self.prompt_builder.build_entity_linking_prompt(
            mention=mention,
            context=context,
            candidates=candidates,
        )

        schema = {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "enum": valid_ids + ["NIL"],
                },
                "reason": {"type": "string"},
            },
            "required": ["entity_id", "reason"],
        }

        try:
            raw_response = self.llm.generate_json(prompt, schema)
            result = json.loads(raw_response)

            selected_id = str(result["entity_id"])
            reason = str(result["reason"])

            if selected_id not in valid_ids and selected_id != "NIL":
                raise ValueError("Selected ID is not in candidate list.")

            return {
                "entity_id": selected_id,
                "reason": reason,
                "raw_response": raw_response,
                "valid_selection": True,
            }

        except Exception as exc:
            return {
                "entity_id": "NIL",
                "reason": f"Structured selection failed: {exc}",
                "raw_response": "",
                "valid_selection": False,
            }
