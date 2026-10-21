from __future__ import annotations

from typing import Any

from models.base import GenericHuggingFaceVLMModel


class InternVL35Model(GenericHuggingFaceVLMModel):
    model_name = "InternVL3.5-8B-Instruct"
    model_id = "OpenGVLab/InternVL3_5-8B-HF"
    trust_remote_code = True

    def _resolve_transformers_model_class(self, transformers_module: Any) -> Any:
        model_cls = getattr(transformers_module, "InternVLForConditionalGeneration", None)
        if model_cls is not None:
            return model_cls
        return super()._resolve_transformers_model_class(transformers_module)

    def _processor_load_kwargs(self) -> dict[str, Any]:
        return {"fix_mistral_regex": True}
