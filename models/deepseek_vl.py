from __future__ import annotations

from typing import Any

from models.base import GenericHuggingFaceVLMModel


class DeepSeekVLModel(GenericHuggingFaceVLMModel):
    model_name = "DeepSeek-VL"
    model_id = "deepseek-community/deepseek-vl-7b-chat"

    def _resolve_transformers_model_class(self, transformers_module: Any) -> Any:
        model_cls = getattr(transformers_module, "DeepseekVLHybridForConditionalGeneration", None)
        if model_cls is not None:
            return model_cls
        model_cls = getattr(transformers_module, "DeepseekVLForConditionalGeneration", None)
        if model_cls is not None:
            return model_cls
        return super()._resolve_transformers_model_class(transformers_module)
