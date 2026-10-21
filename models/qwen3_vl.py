from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.base import (
    configure_safe_hf_download_env,
    DEFAULT_COORDINATE_PROMPT,
    DEFAULT_VLM_MAX_NEW_TOKENS,
    explain_model_load_error,
    GenericHuggingFaceVLMModel,
    ModelUnavailableError,
    find_hf_cache_root,
    find_local_hf_snapshot_dir,
    parse_lat_lon_from_text,
    resolve_runtime_device,
    should_retry_remote_on_local_failure,
)


DEFAULT_QWEN_MODEL_IDS = {
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}


def available_local_qwen_model_ids() -> list[str]:
    root = find_hf_cache_root()
    if not root.exists():
        return []

    available: list[str] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("models--Qwen--") and (path / "snapshots").exists():
            repo_id = path.name.replace("models--", "").replace("--", "/", 1)
            if list((path / "snapshots").glob("*")):
                available.append(repo_id)
    return sorted(available)


def resolve_qwen_model_id(args: Any) -> str:
    if getattr(args, "vlm_model_id", None):
        return args.vlm_model_id

    default_model_id = DEFAULT_QWEN_MODEL_IDS.get(args.method)
    if default_model_id is None:
        available = ", ".join(available_local_qwen_model_ids()) or "none"
        raise ModelUnavailableError(
            f"No default local model id is configured for method '{args.method}'. "
            f"Available local Qwen models: {available}. Use --vlm-model-id to override."
        )
    return default_model_id


def validate_qwen_snapshot(snapshot_dir: Path, model_id: str) -> None:
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing_files = [name for name in required_files if not (snapshot_dir / name).exists()]
    has_processor = (snapshot_dir / "preprocessor_config.json").exists() or (snapshot_dir / "processor_config.json").exists()
    if not has_processor:
        missing_files.append("preprocessor_config.json or processor_config.json")

    if missing_files:
        missing_summary = ", ".join(missing_files)
        raise ModelUnavailableError(
            f"Local snapshot for {model_id} is incomplete: missing {missing_summary} in {snapshot_dir}"
        )


class QwenVLModel(GenericHuggingFaceVLMModel):
    model_name = "Qwen VL"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(
        self,
        model_id: str | None = None,
        model_name: str | None = None,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_COORDINATE_PROMPT,
        local_files_only: bool = True,
        attention_implementation: str = "auto",
    ) -> None:
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            local_files_only=local_files_only,
            attention_implementation=attention_implementation,
        )

    @classmethod
    def from_args(cls, args: Any) -> "QwenVLModel":
        method = getattr(args, "method", "qwen3-vl-8b")
        model_id = resolve_qwen_model_id(args)
        display_name = "Qwen VL" if method == "qwen-custom" else {
            "qwen2-vl-7b": "Qwen2-VL-7B",
            "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
            "qwen3-vl-8b": "Qwen3-VL-8B",
        }.get(method, "Qwen VL")
        if method == "qwen-custom":
            display_name = model_id.split("/")[-1]
        return cls(
            model_id=model_id,
            model_name=display_name,
            device=args.device,
            max_new_tokens=args.vlm_max_new_tokens,
            attention_implementation=getattr(args, "attention_implementation", "auto"),
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            try:
                import torch
                from transformers import AutoProcessor
                import transformers
            except ModuleNotFoundError as exc:
                raise ModelUnavailableError(f"{self.model_name} requires torch and transformers.") from exc

            snapshot_dir = find_local_hf_snapshot_dir(self.model_id) if self.local_files_only else None
            local_only = self.local_files_only
            if self.local_files_only:
                if snapshot_dir is None:
                    print(f"Local snapshot not found for {self.model_id}. Downloading from Hugging Face...")
                    configure_safe_hf_download_env()
                    model_source = self.model_id
                    config_path = None
                    local_only = False
                else:
                    validate_qwen_snapshot(snapshot_dir, self.model_id)
                    model_source = str(snapshot_dir)
                    config_path = snapshot_dir / "config.json"
            else:
                model_source = self.model_id
                config_path = None

            model_type = None
            if config_path is not None and config_path.exists():
                model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")

            model_cls = None
            if model_type == "qwen2_5_vl":
                model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            elif model_type == "qwen3_vl":
                model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            elif model_type == "qwen2_vl":
                model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
            if model_cls is None:
                model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_cls is None:
                raise ModelUnavailableError(f"Installed transformers does not support {self.model_id}.")

            model_kwargs = self._model_load_kwargs(torch)
            try:
                processor = AutoProcessor.from_pretrained(
                    model_source,
                    local_files_only=local_only,
                    trust_remote_code=self.trust_remote_code,
                )
                model = model_cls.from_pretrained(
                    model_source,
                    local_files_only=local_only,
                    trust_remote_code=self.trust_remote_code,
                    **model_kwargs,
                )
            except Exception as exc:
                if local_only and should_retry_remote_on_local_failure(exc):
                    print(
                        f"Local snapshot for {self.model_id} looks incomplete or corrupted ({exc}). "
                        "Retrying with a fresh download from Hugging Face..."
                    )
                    configure_safe_hf_download_env()
                    try:
                        processor = AutoProcessor.from_pretrained(
                            self.model_id,
                            local_files_only=False,
                            trust_remote_code=self.trust_remote_code,
                        )
                        model = model_cls.from_pretrained(
                            self.model_id,
                            local_files_only=False,
                            trust_remote_code=self.trust_remote_code,
                            **model_kwargs,
                        )
                    except Exception as retry_exc:
                        raise ModelUnavailableError(
                            f"Could not load {self.model_name} from {self.model_id}: "
                            f"{explain_model_load_error(retry_exc)}"
                        ) from retry_exc
                else:
                    raise ModelUnavailableError(
                        f"Could not load {self.model_name} from {self.model_id}: "
                        f"{explain_model_load_error(exc)}"
                    ) from exc

            self._finalize_loaded_model(model, torch)
            model.requires_grad_(False)
            model.eval()
            self._processor = processor
            self._model = model
            self._torch = torch

        return self._processor, self._model, self._torch

    def _finalize_loaded_model(self, model: Any, torch_module: Any) -> None:
        del torch_module
        model.to(self.device)


class Qwen3VL8BModel(QwenVLModel):
    model_name = "Qwen3-VL-8B"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
