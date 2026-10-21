from __future__ import annotations

from models.base import GenericHuggingFaceVLMModel


class Gemma4Model(GenericHuggingFaceVLMModel):
    model_name = "Gemma-4-E4B-IT"
    model_id = "google/gemma-4-E4B-it"


# Backward-compatible alias while the repo transitions from the old name.
Gemma3Model = Gemma4Model
