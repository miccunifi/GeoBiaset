from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from models.base import (
    BaseGeoModel,
    COORDINATE_JSON_SCHEMA,
    COORDINATE_RESPONSE_FORMAT,
    COORDINATE_RETRY_PROMPT,
    DEFAULT_COORDINATE_PROMPT,
    DEFAULT_VLM_MAX_NEW_TOKENS,
    ModelOutput,
    ModelUnavailableError,
    parse_lat_lon_from_text,
)


def image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


class GenericOpenAICompatibleVisionModel(BaseGeoModel):
    model_name = "OpenAI-compatible Vision Model"
    model_id = ""

    def __init__(
        self,
        *,
        model_id: str,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = DEFAULT_VLM_MAX_NEW_TOKENS,
        prompt: str = DEFAULT_COORDINATE_PROMPT,
        extra_body: dict[str, Any] | None = None,
        enforce_json: bool = True,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.extra_body = dict(extra_body or {})
        self.enforce_json = enforce_json
        self._client: Any | None = None

    def _metadata(self, **values: Any) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.base_url:
            raise ModelUnavailableError(
                f"{self.model_name} must be run through an OpenAI-compatible server. "
                "Start the model server, then set OPENAI_BASE_URL, for example "
                "`OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY`."
            )
        try:
            from openai import OpenAI
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModelUnavailableError(
                f"{self.model_name} requires the openai Python package. Run `uv sync` after pulling dependencies."
            ) from exc

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _message_content(self, image_path: str | Path, prompt: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image_path)},
            },
            {"type": "text", "text": prompt or self.prompt},
        ]

    def _completion_extra_body(self, use_schema: bool = True) -> dict[str, Any] | None:
        extra_body = dict(self.extra_body)
        if use_schema and self.enforce_json:
            extra_body.setdefault("guided_json", COORDINATE_JSON_SCHEMA)
        return extra_body or None

    def _create_completion(self, client: Any, image_path: str | Path, prompt: str, use_schema: bool = True) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self._message_content(image_path, prompt=prompt)}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "extra_body": self._completion_extra_body(use_schema=use_schema),
        }
        if use_schema and self.enforce_json:
            request_kwargs["response_format"] = COORDINATE_RESPONSE_FORMAT

        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception:
            if not (use_schema and self.enforce_json):
                raise
            request_kwargs.pop("response_format", None)
            try:
                return client.chat.completions.create(**request_kwargs)
            except Exception:
                request_kwargs["extra_body"] = self.extra_body or None
                return client.chat.completions.create(**request_kwargs)

    def _decode_and_parse(self, client: Any, image_path: str | Path) -> tuple[str, float, float, bool]:
        response = self._create_completion(client, image_path, self.prompt, use_schema=True)
        decoded = self._extract_text(response)
        try:
            pred_lat, pred_lon = parse_lat_lon_from_text(decoded)
            return decoded, pred_lat, pred_lon, False
        except ValueError as first_error:
            retry_response = self._create_completion(client, image_path, COORDINATE_RETRY_PROMPT, use_schema=True)
            retry_decoded = self._extract_text(retry_response)
            try:
                pred_lat, pred_lon = parse_lat_lon_from_text(retry_decoded)
            except ValueError as retry_error:
                message = (
                    f"{first_error}; retry also failed with response: "
                    f"{retry_decoded[:300]}"
                )
                raise ValueError(message) from retry_error
            return retry_decoded, pred_lat, pred_lon, True

    def _extract_text(self, response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ValueError(f"Unsupported OpenAI-compatible response format for {self.model_name}: {response!r}") from exc
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def inference(self, image_path: str | Path, **kwargs: Any) -> ModelOutput:
        return self.batch_inference([image_path], item_kwargs=[kwargs])[0]

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

        client = self._load()
        outputs: list[ModelOutput] = []
        for image_path, current_item_kwargs in zip(image_paths, item_kwargs):
            merged_kwargs = dict(shared_kwargs)
            merged_kwargs.update(current_item_kwargs)
            try:
                decoded, pred_lat, pred_lon, retried = self._decode_and_parse(client, image_path)
            except ModelUnavailableError:
                raise
            except ValueError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"Could not run {self.model_name} through {self.base_url}: {exc}") from exc

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
                        backend="openai-compatible",
                        base_url=self.base_url,
                        output_contract_retry=retried,
                    ),
                )
            )
        return outputs
