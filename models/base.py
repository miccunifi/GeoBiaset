from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_COORDINATE_PROMPT = (
    "Estimate the geographic location of this image. Use visual clues such as architecture, vegetation, "
    "terrain, road markings, signs, weather, vehicles, shadows, coastlines, mountains, water, and text. "
    'Return exactly one JSON object with numeric "lat" and "lon" values and nothing else. '
    "Use signed decimal degrees: south and west are negative, north and east are positive. "
    "Do not copy coordinates from examples or instructions; there are no reference coordinates in this prompt. "
    "Rules: no explanation, no markdown, no prose, no reasoning, no <think> section. If uncertain, still "
    'give your best guess as latitude and longitude. The first character of your answer must be "{" and '
    'the last character must be "}".'
)
COORDINATE_RETRY_PROMPT = (
    'Your previous answer was invalid. Output only one valid JSON object with exactly two keys: "lat" and "lon". '
    "Use signed decimal-degree numbers for your best-estimate latitude and longitude. "
    "Do not include examples, markdown, prose, or reasoning."
)
COORDINATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "minimum": -90, "maximum": 90},
        "lon": {"type": "number", "minimum": -180, "maximum": 180},
    },
    "required": ["lat", "lon"],
    "additionalProperties": False,
}
COORDINATE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "geo_coordinates",
        "strict": True,
        "schema": COORDINATE_JSON_SCHEMA,
    },
}
DEFAULT_VLM_MAX_NEW_TOKENS = 32
COORDINATE_NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"


def configure_safe_hf_download_env() -> None:
    # The native hf_xet / hf_transfer backends have been less stable than the
    # regular Python download path on some Python 3.13 environments.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


configure_safe_hf_download_env()


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be used in the current local environment."""


@dataclass(frozen=True)
class ModelOutput:
    model_name: str
    image_path: str
    pred_lat: float
    pred_lon: float
    confidence: float | None = None
    raw_output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat = float(self.pred_lat)
        lon = float(self.pred_lon)
        if not -90 <= lat <= 90:
            raise ValueError(f"pred_lat must be in [-90, 90], got {lat}")
        if not -180 <= lon <= 180:
            raise ValueError(f"pred_lon must be in [-180, 180], got {lon}")
        object.__setattr__(self, "pred_lat", lat)
        object.__setattr__(self, "pred_lon", lon)
        object.__setattr__(self, "image_path", str(self.image_path))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))


class BaseGeoModel:
    model_name = "base"

    def __call__(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        return self.inference(image_path, **kwargs)

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        raise NotImplementedError

    def forward(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        return self.inference(image_path, **kwargs)

    def batch_inference(
        self,
        image_paths: list[str | Path],
        *,
        item_kwargs: list[dict[str, Any]] | None = None,
        **shared_kwargs: Any,
    ) -> list[ModelOutput]:
        if item_kwargs is None:
            item_kwargs = [{} for _ in image_paths]
        if len(item_kwargs) != len(image_paths):
            raise ValueError("item_kwargs must have the same length as image_paths")

        outputs: list[ModelOutput] = []
        for image_path, current_item_kwargs in zip(image_paths, item_kwargs):
            merged_kwargs = dict(shared_kwargs)
            merged_kwargs.update(current_item_kwargs)
            outputs.append(self.inference(image_path, **merged_kwargs))
        return outputs

    def _metadata(self, **values: Any) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}


def resolve_runtime_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_runtime_device(device: str, torch_module: Any | None = None) -> str:
    if device != "cuda":
        return device

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except Exception:
            return device
        torch = imported_torch

    try:
        current_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_index)
        return f"cuda:{current_index} ({device_name})"
    except Exception:
        return device


def resolve_cuda_device_index(device: str = "auto", torch_module: Any | None = None) -> int | None:
    resolved_device = resolve_runtime_device(device)
    if not resolved_device.startswith("cuda"):
        return None

    suffix = resolved_device.partition(":")[2]
    if suffix:
        try:
            return int(suffix)
        except ValueError:
            return None

    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except Exception:
            return None
        torch = imported_torch

    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


def total_device_vram_gb(device: str = "auto", torch_module: Any | None = None) -> float | None:
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch
        except Exception:
            return None
        torch = imported_torch

    device_index = resolve_cuda_device_index(device, torch_module=torch)
    if device_index is None:
        return None

    try:
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    except Exception:
        return None
    return float(total_bytes) / float(1024 ** 3)


def find_local_hf_snapshot_dir(repo_id: str) -> Path | None:
    configure_safe_hf_download_env()
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None

    try:
        return Path(snapshot_download(repo_id, local_files_only=True)).resolve()
    except Exception:
        return None


def find_hf_cache_root() -> Path:
    cache_root = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (
            str(Path(os.environ["HF_HOME"]).expanduser() / "hub")
            if os.environ.get("HF_HOME")
            else None
        )
        or str(Path.home() / ".cache" / "huggingface" / "hub")
    )
    return Path(cache_root).expanduser()


def should_retry_remote_on_local_failure(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True

    text = str(exc)
    retry_hints = (
        "No such file or directory",
        ".safetensors",
        "pytorch_model",
        "model.safetensors",
        "checkpoint shard",
        "looks incomplete",
        "does not appear to have a file named",
        "modeling_",
        "configuration_",
    )
    return any(hint in text for hint in retry_hints)


def explain_model_load_error(exc: Exception) -> str:
    if isinstance(exc, OSError) and exc.errno == 27:
        cache_root = find_hf_cache_root()
        return (
            "File too large while downloading or reading the model weights. "
            "This usually means the Hugging Face cache is on a filesystem with a per-file size limit. "
            f"Point HF_HUB_CACHE or HF_HOME to a local disk that supports multi-GB files, then retry. "
            f"Current cache root: {cache_root}"
        )

    text = str(exc)
    if "File too large" in text:
        cache_root = find_hf_cache_root()
        return (
            "File too large while downloading or reading the model weights. "
            "This usually means the Hugging Face cache is on a filesystem with a per-file size limit. "
            f"Point HF_HUB_CACHE or HF_HOME to a local disk that supports multi-GB files, then retry. "
            f"Current cache root: {cache_root}"
        )

    if "We couldn't connect to 'https://huggingface.co'" in text:
        cache_root = find_hf_cache_root()
        return (
            "Could not reach Hugging Face and no compatible local snapshot was available. "
            "Check internet access on that machine, or pre-download the model into the Hugging Face cache. "
            f"Current cache root: {cache_root}"
        )

    return text


def extract_json_block(text: str) -> str:
    text = text.strip()
    if "{" in text and "}" in text:
        return text[text.find("{") : text.rfind("}") + 1]
    return text


def _apply_cardinal_direction(value: float, axis: str, direction: str | None) -> float:
    if not direction:
        return value

    normalized = direction.upper()
    if axis == "lat" and normalized in {"N", "S"}:
        return abs(value) if normalized == "N" else -abs(value)
    if axis == "lon" and normalized in {"E", "W"}:
        return abs(value) if normalized == "E" else -abs(value)
    return value


def _direction_after_number(text: str, end_index: int, axis: str) -> str | None:
    direction_pattern = (
        r"^\s*(?:°|degrees?)?\s*([NS])\b"
        if axis == "lat"
        else r"^\s*(?:°|degrees?)?\s*([EW])\b"
    )
    match = re.search(direction_pattern, text[end_index:], flags=re.IGNORECASE)
    return match.group(1) if match is not None else None


def _coerce_coordinate_value(value: Any, axis: str) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    match = re.search(COORDINATE_NUMBER_PATTERN, text)
    if match is None:
        return None

    coordinate = float(match.group(0))
    direction = _direction_after_number(text, match.end(), axis)
    return _apply_cardinal_direction(coordinate, axis, direction)


def _valid_coordinate_pair(lat: float, lon: float) -> tuple[float, float] | None:
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    return lat, lon


def _coerce_coordinate_payload(payload: Any) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None

    lat_value = payload.get("lat", payload.get("latitude"))
    lon_value = payload.get("lon", payload.get("lng", payload.get("longitude")))
    if lat_value is None or lon_value is None:
        return None

    lat = _coerce_coordinate_value(lat_value, "lat")
    lon = _coerce_coordinate_value(lon_value, "lon")
    if lat is None or lon is None:
        return None

    return _valid_coordinate_pair(lat, lon)


def _json_coordinate_candidates(text: str) -> list[tuple[float, float]]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[float, float]] = []
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        coordinates = _coerce_coordinate_payload(payload)
        if coordinates is not None:
            candidates.append(coordinates)
    return candidates


def _parse_labelled_coordinate_pair(text: str) -> tuple[float, float] | None:
    number = f"({COORDINATE_NUMBER_PATTERN})"
    lat_label = r"lat(?:itude)?"
    lon_label = r"lon(?:gitude)?|lng"
    labelled_value = (
        r'["\']?\s*(?:{label})\s*["\']?\s*(?::|=|\bis\b)?\s*'
        r"{number}\s*(?:°|degrees?)?\s*([NSEW])?"
    )

    lat_then_lon = re.search(
        labelled_value.format(label=lat_label, number=number)
        + r".*?"
        + labelled_value.format(label=lon_label, number=number),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if lat_then_lon is not None:
        lat = _apply_cardinal_direction(float(lat_then_lon.group(1)), "lat", lat_then_lon.group(2))
        lon = _apply_cardinal_direction(float(lat_then_lon.group(3)), "lon", lat_then_lon.group(4))
        return _valid_coordinate_pair(lat, lon)

    lon_then_lat = re.search(
        labelled_value.format(label=lon_label, number=number)
        + r".*?"
        + labelled_value.format(label=lat_label, number=number),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if lon_then_lat is not None:
        lon = _apply_cardinal_direction(float(lon_then_lat.group(1)), "lon", lon_then_lat.group(2))
        lat = _apply_cardinal_direction(float(lon_then_lat.group(3)), "lat", lon_then_lat.group(4))
        return _valid_coordinate_pair(lat, lon)

    return None


def parse_lat_lon_from_text(text: str) -> tuple[float, float]:
    raw = text.strip().replace('""', '"')

    json_candidates = _json_coordinate_candidates(raw)
    if json_candidates:
        return json_candidates[-1]

    labelled_coordinates = _parse_labelled_coordinate_pair(raw)
    if labelled_coordinates is not None:
        return labelled_coordinates

    number_matches = list(re.finditer(COORDINATE_NUMBER_PATTERN, raw))
    if len(number_matches) == 2:
        lat = _apply_cardinal_direction(
            float(number_matches[0].group(0)),
            "lat",
            _direction_after_number(raw, number_matches[0].end(), "lat"),
        )
        lon = _apply_cardinal_direction(
            float(number_matches[1].group(0)),
            "lon",
            _direction_after_number(raw, number_matches[1].end(), "lon"),
        )
        if _valid_coordinate_pair(lat, lon) is not None:
            return lat, lon
        raise ValueError(f"Parsed invalid coordinates: lat={lat}, lon={lon}")

    raise ValueError(f"Could not parse coordinates from response: {text[:300]}")


def candidate_path_keys(path_value: str | Path) -> list[str]:
    path = Path(path_value)
    keys = [str(path), path.name]
    try:
        keys.append(str(path.expanduser().resolve(strict=False)))
    except Exception:
        pass

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


class GenericHuggingFaceVLMModel(BaseGeoModel):
    model_name = "HuggingFace VLM"
    model_id = ""
    trust_remote_code = False

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_COORDINATE_PROMPT,
        local_files_only: bool = True,
        trust_remote_code: bool | None = None,
        model_name: str | None = None,
        attention_implementation: str = "auto",
    ) -> None:
        self.model_id = model_id or self.model_id
        self.device = resolve_runtime_device(device)
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.local_files_only = local_files_only
        self.attention_implementation = attention_implementation
        if trust_remote_code is not None:
            self.trust_remote_code = trust_remote_code
        if model_name is not None:
            self.model_name = model_name
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _resolve_transformers_model_class(self, transformers_module: Any) -> Any:
        return getattr(transformers_module, "AutoModelForImageTextToText", None)

    def _resolve_transformers_fallback_model_class(self, transformers_module: Any) -> Any:
        if not self.trust_remote_code:
            return None
        return getattr(transformers_module, "AutoModel", None)

    def _processor_load_kwargs(self) -> dict[str, Any]:
        return {}

    def _model_load_kwargs(self, torch_module: Any) -> dict[str, Any]:
        dtype = torch_module.bfloat16 if self.device == "cuda" else None
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if self.attention_implementation != "auto":
            kwargs["attn_implementation"] = self.attention_implementation
        return kwargs

    def _load_processor_and_model(
        self,
        *,
        model_source: str,
        local_only: bool,
        processor_cls: Any,
        model_cls: Any,
        torch_module: Any,
    ) -> tuple[Any, Any]:
        processor = processor_cls.from_pretrained(
            model_source,
            local_files_only=local_only,
            trust_remote_code=self.trust_remote_code,
            **self._processor_load_kwargs(),
        )
        model = model_cls.from_pretrained(
            model_source,
            local_files_only=local_only,
            trust_remote_code=self.trust_remote_code,
            **self._model_load_kwargs(torch_module),
        )
        return processor, model

    def _resolve_model_source(self) -> tuple[str, bool]:
        if not self.model_id:
            raise ModelUnavailableError(f"{self.model_name} has no configured Hugging Face model id.")
        if not self.local_files_only:
            return self.model_id, False

        snapshot_dir = find_local_hf_snapshot_dir(self.model_id)
        if snapshot_dir is not None:
            return str(snapshot_dir), True

        print(f"Local snapshot not found for {self.model_id}. Downloading from Hugging Face...")
        configure_safe_hf_download_env()
        return self.model_id, False

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None:
            try:
                import torch
                from transformers import AutoProcessor
                import transformers
            except (ImportError, ModuleNotFoundError) as exc:
                missing_module = getattr(exc, "name", None) or "torch or transformers"
                raise ModelUnavailableError(
                    f"{self.model_name} requires torch and transformers to be installed. "
                    f"Missing import: {missing_module}. "
                    "On a fresh machine, run `uv sync` to install the project dependencies."
                ) from exc

            model_source, local_only = self._resolve_model_source()
            model_cls = self._resolve_transformers_model_class(transformers)
            if model_cls is None:
                raise ModelUnavailableError(f"Installed transformers does not support {self.model_name}.")
            fallback_model_cls = self._resolve_transformers_fallback_model_class(transformers)
            try:
                processor, model = self._load_processor_and_model(
                    model_source=model_source,
                    local_only=local_only,
                    processor_cls=AutoProcessor,
                    model_cls=model_cls,
                    torch_module=torch,
                )
            except Exception as exc:
                recovered = False
                if fallback_model_cls is not None and not should_retry_remote_on_local_failure(exc):
                    try:
                        processor, model = self._load_processor_and_model(
                            model_source=model_source,
                            local_only=local_only,
                            processor_cls=AutoProcessor,
                            model_cls=fallback_model_cls,
                            torch_module=torch,
                        )
                        recovered = True
                    except Exception as fallback_exc:
                        exc = fallback_exc

                if not recovered:
                    if local_only and should_retry_remote_on_local_failure(exc):
                        print(
                            f"Local snapshot for {self.model_id} looks incomplete or corrupted ({exc}). "
                            "Retrying with a fresh download from Hugging Face..."
                        )
                        configure_safe_hf_download_env()
                        try:
                            processor, model = self._load_processor_and_model(
                                model_source=self.model_id,
                                local_only=False,
                                processor_cls=AutoProcessor,
                                model_cls=fallback_model_cls or model_cls,
                                torch_module=torch,
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

            model.to(self.device)
            model.requires_grad_(False)
            model.eval()
            print(f"{self.model_name} loaded on {describe_runtime_device(self.device, torch)}")
            self._processor = processor
            self._model = model
            self._torch = torch

        return self._processor, self._model, self._torch

    def _prompt_text(self, processor: Any, prompt: str | None = None) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or self.prompt},
                    {"type": "image"},
                ],
            }
        ]
        base_kwargs = {"tokenize": False, "add_generation_prompt": True}
        for extra_kwargs in ({"enable_thinking": False}, {"thinking": False}, {}):
            try:
                return processor.apply_chat_template(conversation, **base_kwargs, **extra_kwargs)
            except TypeError:
                if extra_kwargs:
                    continue
                raise

    def _model_input_dtype(self, model: Any) -> Any | None:
        dtype = getattr(model, "dtype", None)
        if dtype is not None and getattr(dtype, "is_floating_point", False):
            return dtype

        try:
            parameter = next(model.parameters())
        except Exception:
            return None

        dtype = getattr(parameter, "dtype", None)
        if dtype is not None and getattr(dtype, "is_floating_point", False):
            return dtype
        return None

    def _build_inputs(self, processor: Any, model: Any, images: list[Any], prompt: str | None = None) -> Any:
        prompt_text = self._prompt_text(processor, prompt=prompt)
        inputs = processor(
            text=[prompt_text for _ in images],
            images=images,
            return_tensors="pt",
        ).to(self.device)
        input_dtype = self._model_input_dtype(model)
        if input_dtype is None:
            return inputs

        tensor_mapping = getattr(inputs, "data", inputs)
        if not hasattr(tensor_mapping, "items"):
            return inputs

        for key, value in list(tensor_mapping.items()):
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                tensor_mapping[key] = value.to(device=self.device, dtype=input_dtype)
        return inputs

    def _generate_decoded_batch(
        self,
        processor: Any,
        model: Any,
        torch: Any,
        images: list[Any],
        prompt: str | None = None,
    ) -> list[str]:
        inputs = self._build_inputs(processor, model, images, prompt=prompt)
        pad_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)

        return self._decode_batch(processor, inputs, outputs)

    def _parse_or_retry_decoded_output(
        self,
        decoded: str,
        image: Any,
        processor: Any,
        model: Any,
        torch: Any,
    ) -> tuple[float, float, str, bool]:
        try:
            pred_lat, pred_lon = parse_lat_lon_from_text(decoded)
            return pred_lat, pred_lon, decoded, False
        except ValueError as first_error:
            retry_decoded = self._generate_decoded_batch(
                processor,
                model,
                torch,
                [image],
                prompt=COORDINATE_RETRY_PROMPT,
            )[0]
            try:
                pred_lat, pred_lon = parse_lat_lon_from_text(retry_decoded)
            except ValueError as retry_error:
                message = (
                    f"{first_error}; retry also failed with response: "
                    f"{retry_decoded[:300]}"
                )
                raise ValueError(message) from retry_error
            return pred_lat, pred_lon, retry_decoded, True

    def _decode(self, processor: Any, inputs: Any, outputs: Any) -> str:
        return self._decode_batch(processor, inputs, outputs)[0]

    def _decode_batch(self, processor: Any, inputs: Any, outputs: Any) -> list[str]:
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            input_lengths = [int(length) for length in attention_mask.sum(dim=1).tolist()]
        else:
            input_ids = getattr(inputs, "input_ids", None)
            if input_ids is None:
                raise ValueError("Model inputs must include attention_mask or input_ids for decoding.")
            input_lengths = [int(input_ids.shape[1]) for _ in range(outputs.shape[0])]

        generated = [outputs[index, input_length:] for index, input_length in enumerate(input_lengths)]
        return [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]

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

        processor, model, torch = self._load()
        images: list[Any] = []
        for image_path in image_paths:
            with Image.open(image_path) as image_handle:
                images.append(image_handle.convert("RGB"))

        decoded_outputs = self._generate_decoded_batch(processor, model, torch, images)
        batch_outputs: list[ModelOutput] = []
        for image_path, image, decoded, current_item_kwargs in zip(image_paths, images, decoded_outputs, item_kwargs):
            merged_kwargs = dict(shared_kwargs)
            merged_kwargs.update(current_item_kwargs)
            pred_lat, pred_lon, final_decoded, retried = self._parse_or_retry_decoded_output(
                decoded,
                image,
                processor,
                model,
                torch,
            )
            batch_outputs.append(
                ModelOutput(
                    model_name=self.model_name,
                    image_path=str(image_path),
                    pred_lat=pred_lat,
                    pred_lon=pred_lon,
                    raw_output=final_decoded,
                    metadata=self._metadata(
                        dataset_path=merged_kwargs.get("dataset_path"),
                        source_path=merged_kwargs.get("source_path"),
                        augmentation=merged_kwargs.get("augmentation"),
                        model_id=self.model_id,
                        output_contract_retry=retried,
                    ),
                )
            )
        return batch_outputs
