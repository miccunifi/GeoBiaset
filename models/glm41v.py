from __future__ import annotations

from typing import Any

from models.base import GenericHuggingFaceVLMModel


class GLM41VModel(GenericHuggingFaceVLMModel):
    model_name = "GLM-4.1V-9B-Base"
    model_id = "zai-org/GLM-4.1V-9B-Base"

    def _resolve_transformers_model_class(self, transformers_module: Any) -> Any:
        model_cls = getattr(transformers_module, "Glm4vForConditionalGeneration", None)
        if model_cls is not None:
            return model_cls
        return super()._resolve_transformers_model_class(transformers_module)
