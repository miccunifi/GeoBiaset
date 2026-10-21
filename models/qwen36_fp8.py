from __future__ import annotations

from typing import Any

from models.base import DEFAULT_COORDINATE_PROMPT, DEFAULT_VLM_MAX_NEW_TOKENS, total_device_vram_gb
from models.qwen3_vl import QwenVLModel


QWEN36_FP8_MODEL_ID = "Qwen/Qwen3.6-27B-FP8"
QWEN36_UNQUANTIZED_MODEL_ID = "Qwen/Qwen3.6-27B"
QWEN36_UNQUANTIZED_VRAM_THRESHOLD_GB = 50.0


def should_disable_qwen36_quantization(
    device: str = "auto",
    *,
    min_vram_gb: float = QWEN36_UNQUANTIZED_VRAM_THRESHOLD_GB,
    torch_module: Any | None = None,
) -> bool:
    available_vram_gb = total_device_vram_gb(device, torch_module=torch_module)
    return available_vram_gb is not None and available_vram_gb > min_vram_gb


class Qwen36FP8Model(QwenVLModel):
    model_name = "Qwen3.6-27B-FP8"
    model_id = QWEN36_FP8_MODEL_ID

    def __init__(
        self,
        model_id: str | None = None,
        model_name: str | None = None,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_COORDINATE_PROMPT,
        local_files_only: bool = True,
        disable_quantization_if_vram_over_50gb: bool = False,
        attention_implementation: str = "auto",
        openai_base_url: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        del openai_base_url, openai_api_key
        effective_model_id = model_id or self.model_id
        effective_model_name = model_name or self.model_name
        self.disable_quantization_if_vram_over_50gb = disable_quantization_if_vram_over_50gb
        self.detected_vram_gb = total_device_vram_gb(device)
        self.uses_unquantized_variant = False

        if (
            model_id is None
            and disable_quantization_if_vram_over_50gb
            and should_disable_qwen36_quantization(device)
        ):
            effective_model_id = QWEN36_UNQUANTIZED_MODEL_ID
            effective_model_name = "Qwen3.6-27B"
            self.uses_unquantized_variant = True

        super().__init__(
            model_id=effective_model_id,
            model_name=effective_model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            local_files_only=local_files_only,
            attention_implementation=attention_implementation,
        )

    def _model_load_kwargs(self, torch_module: Any) -> dict[str, Any]:
        if self.uses_unquantized_variant:
            return super()._model_load_kwargs(torch_module)

        kwargs: dict[str, Any] = {}
        if self.attention_implementation != "auto":
            kwargs["attn_implementation"] = self.attention_implementation
        if self.device.startswith("cuda"):
            kwargs.setdefault("device_map", {"": self.device})
        else:
            kwargs.setdefault("device_map", "auto")
        return kwargs

    def _finalize_loaded_model(self, model: Any, torch_module: Any) -> None:
        if self.uses_unquantized_variant:
            super()._finalize_loaded_model(model, torch_module)
