from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from models.base import (
    BaseGeoModel,
    COORDINATE_JSON_SCHEMA,
    COORDINATE_RETRY_PROMPT,
    DEFAULT_COORDINATE_PROMPT,
    DEFAULT_VLM_MAX_NEW_TOKENS,
    ModelOutput,
    ModelUnavailableError,
    find_local_hf_snapshot_dir,
    parse_lat_lon_from_text,
    resolve_runtime_device,
)


def maybe_configure_vllm_cuda_device(device: str) -> None:
    resolved_device = resolve_runtime_device(device)
    if not resolved_device.startswith("cuda:"):
        return

    requested_index = resolved_device.split(":", 1)[1]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = requested_index


class GenericVLLMVisionModel(BaseGeoModel):
    model_name = "vLLM Vision Model"
    model_id = ""
    trust_remote_code = False

    def __init__(
        self,
        model_id: str,
        model_name: str,
        *,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_COORDINATE_PROMPT,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        enforce_json: bool = True,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.device = resolve_runtime_device(device)
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.enforce_json = enforce_json
        self._llm: Any | None = None
        self._sampling_params_cls: Any | None = None

    def _metadata(self, **values: Any) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    def _resolve_model_source(self) -> str:
        if not self.model_id:
            raise ModelUnavailableError(f"{self.model_name} has no configured model id.")
        if not self.local_files_only:
            return self.model_id

        snapshot_dir = find_local_hf_snapshot_dir(self.model_id)
        if snapshot_dir is not None:
            return str(snapshot_dir)
        return self.model_id

    def _load(self) -> tuple[Any, Any]:
        if self._llm is None:
            maybe_configure_vllm_cuda_device(self.device)
            try:
                from vllm import LLM, SamplingParams
            except (ImportError, ModuleNotFoundError) as exc:
                raise ModelUnavailableError(
                    f"{self.model_name} requires vllm. Install it with `uv pip install vllm`."
                ) from exc

            model_source = self._resolve_model_source()
            llm_kwargs: dict[str, Any] = {
                "model": model_source,
                "trust_remote_code": self.trust_remote_code,
                "limit_mm_per_prompt": {"image": 1},
            }
            if self.device == "cuda":
                llm_kwargs["dtype"] = "bfloat16"

            try:
                self._llm = LLM(**llm_kwargs)
            except Exception as exc:
                raise ModelUnavailableError(
                    f"Could not load {self.model_name} from {self.model_id} with vLLM: {exc}"
                ) from exc
            self._sampling_params_cls = SamplingParams

        return self._llm, self._sampling_params_cls

    def _chat_messages(self, image: Any, prompt: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or self.prompt},
                    {"type": "image_pil", "image_pil": image},
                ],
            }
        ]

    def _sampling_params(self, sampling_params_cls: Any, use_schema: bool = True) -> Any:
        kwargs: dict[str, Any] = {"max_tokens": self.max_new_tokens, "temperature": 0.0}
        if use_schema and self.enforce_json:
            try:
                parameters = inspect.signature(sampling_params_cls).parameters
            except (TypeError, ValueError):
                parameters = {}

            if "guided_json" in parameters:
                kwargs["guided_json"] = COORDINATE_JSON_SCHEMA
            elif "guided_decoding" in parameters:
                try:
                    from vllm.sampling_params import GuidedDecodingParams
                except Exception:
                    pass
                else:
                    kwargs["guided_decoding"] = GuidedDecodingParams(json=COORDINATE_JSON_SCHEMA)

        try:
            return sampling_params_cls(**kwargs)
        except TypeError:
            return sampling_params_cls(max_tokens=self.max_new_tokens, temperature=0.0)

    def _run_chat(
        self,
        llm: Any,
        sampling_params_cls: Any,
        conversations: list[list[dict[str, Any]]],
        use_schema: bool = True,
    ) -> Any:
        sampling_params = self._sampling_params(sampling_params_cls, use_schema=use_schema)
        try:
            return llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)
        except Exception:
            if not (use_schema and self.enforce_json):
                raise
            fallback_params = self._sampling_params(sampling_params_cls, use_schema=False)
            return llm.chat(conversations, sampling_params=fallback_params, use_tqdm=False)

    def _extract_generated_text(self, result: Any) -> str:
        if hasattr(result, "outputs") and result.outputs:
            first_output = result.outputs[0]
            if hasattr(first_output, "text"):
                return str(first_output.text).strip()
        if hasattr(result, "text"):
            return str(result.text).strip()
        raise ValueError(f"Unsupported vLLM response format for {self.model_name}: {type(result)!r}")

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        return self.batch_inference([image_path], item_kwargs=[kwargs])[0]

    def batch_inference(
        self,
        image_paths: list[str | Path],
        *,
        item_kwargs: list[dict[str, Any]] | None = None,
        **shared_kwargs: Any,
    ) -> list[ModelOutput]:
        from PIL import Image

        if item_kwargs is None:
            item_kwargs = [{} for _ in image_paths]
        if len(item_kwargs) != len(image_paths):
            raise ValueError("item_kwargs must have the same length as image_paths")

        llm, sampling_params_cls = self._load()
        images: list[Any] = []
        conversations: list[list[dict[str, Any]]] = []
        for image_path in image_paths:
            with Image.open(image_path) as image_handle:
                image = image_handle.convert("RGB")
            images.append(image)
            conversations.append(self._chat_messages(image))

        try:
            results = self._run_chat(llm, sampling_params_cls, conversations, use_schema=True)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not run {self.model_name} with vLLM: {exc}"
            ) from exc

        outputs: list[ModelOutput] = []
        for image_path, image, result, current_item_kwargs in zip(image_paths, images, results, item_kwargs):
            merged_kwargs = dict(shared_kwargs)
            merged_kwargs.update(current_item_kwargs)
            decoded = self._extract_generated_text(result)
            retried = False
            try:
                pred_lat, pred_lon = parse_lat_lon_from_text(decoded)
            except ValueError as first_error:
                retry_conversations = [self._chat_messages(image, prompt=COORDINATE_RETRY_PROMPT)]
                retry_results = self._run_chat(llm, sampling_params_cls, retry_conversations, use_schema=True)
                decoded = self._extract_generated_text(retry_results[0])
                try:
                    pred_lat, pred_lon = parse_lat_lon_from_text(decoded)
                except ValueError as retry_error:
                    message = (
                        f"{first_error}; retry also failed with response: "
                        f"{decoded[:300]}"
                    )
                    raise ValueError(message) from retry_error
                retried = True
            outputs.append(
                ModelOutput(
                    model_name=self.model_name,
                    image_path=str(image_path),
                    pred_lat=pred_lat,
                    pred_lon=pred_lon,
                    raw_output=decoded,
                    metadata=self._metadata(
                        dataset_path=merged_kwargs.get("dataset_path"),
                        source_path=merged_kwargs.get("source_path"),
                        augmentation=merged_kwargs.get("augmentation"),
                        model_id=self.model_id,
                        backend="vllm",
                        output_contract_retry=retried,
                    ),
                )
            )
        return outputs
