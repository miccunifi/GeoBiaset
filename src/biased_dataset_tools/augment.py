from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

from biased_dataset_tools.classifier import (
    GROUND_TRUTH_AUGMENTATION,
    ManifestEntry,
    append_manifest_row,
    load_config,
    normalize_manifest_dataset_paths,
    read_manifest_entries,
    resolve_dataset_path,
    to_relative_dataset_path,
    write_manifest_entries,
)


# Edit this list directly, or override it with --attributes.
DEFAULT_ATTRIBUTES = ["chinese", "european", "african", "indian"]
DEFAULT_COMFYUI_SERVER_URL = "http://127.0.0.1:8188"
DEFAULT_WORKFLOW_PATH = Path("image_qwen_image_edit_2509.json")


@dataclass(frozen=True)
class WorkflowTargets:
    load_image_node_id: str
    positive_prompt_node_id: str
    save_image_node_id: str


class ComfyUIClient:
    """Small HTTP client for the few ComfyUI endpoints this script needs."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")

    def upload_input_image(self, image_path: Path) -> str:
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        upload_name = f"biased_dataset_{uuid.uuid4().hex}_{image_path.name}"
        with image_path.open("rb") as handle:
            body, header = build_multipart_form_data(
                fields={"type": "input", "overwrite": "true"},
                file_field_name="image",
                filename=upload_name,
                file_bytes=handle.read(),
                content_type=content_type,
            )

        response = self.request_json(
            "/upload/image",
            data=body,
            headers={"Content-Type": header},
        )
        return str(response["name"])

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        payload = json.dumps({"prompt": workflow, "client_id": uuid.uuid4().hex}).encode("utf-8")
        response = self.request_json(
            "/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        return str(response["prompt_id"])

    def wait_for_output_image(
        self,
        prompt_id: str,
        save_image_node_id: str,
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> tuple[bytes, dict[str, Any]]:
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            history = self.request_json(f"/history/{urllib.parse.quote(prompt_id)}")
            prompt_history = history.get(prompt_id, {}) if isinstance(history, dict) else {}
            outputs = prompt_history.get("outputs", {})
            node_output = outputs.get(save_image_node_id, {})
            images = node_output.get("images") or []
            if images:
                image_info = images[0]
                return self.download_output_image(image_info), image_info

            time.sleep(poll_interval_seconds)

        raise TimeoutError(
            f"Timed out after {timeout_seconds}s waiting for ComfyUI prompt {prompt_id}."
        )

    def download_output_image(self, image_info: dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
        )
        return self.request_bytes(f"/view?{query}")

    def request_json(
        self,
        path: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return json.loads(self.request_bytes(path, data=data, headers=headers).decode("utf-8"))

    def request_bytes(
        self,
        path: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers=headers or {},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach ComfyUI at {self.server_url}. "
                "Start the web server or pass --server-url."
            ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment the labeled dataset with ComfyUI image edits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW_PATH,
        help="Path to the ComfyUI workflow JSON file.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_COMFYUI_SERVER_URL,
        help="ComfyUI server URL.",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=DEFAULT_ATTRIBUTES,
        help="Attributes to inject into the prompts.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="How long to wait for each ComfyUI job.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="How often to poll ComfyUI history while a job is running.",
    )
    return parser.parse_args(argv)


def build_multipart_form_data(
    fields: dict[str, str],
    file_field_name: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"Boundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"\r\n'.encode(
            "utf-8"
        )
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), f"multipart/form-data; boundary={boundary}"


def load_workflow_template(workflow_path: Path) -> dict[str, Any]:
    with workflow_path.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    if not isinstance(workflow, dict):
        raise ValueError(f"Workflow at {workflow_path} is not a ComfyUI JSON object.")
    return workflow


def find_workflow_targets(workflow: dict[str, Any]) -> WorkflowTargets:
    load_image_nodes = [node_id for node_id, node in workflow.items() if node.get("class_type") == "LoadImage"]
    save_image_nodes = [node_id for node_id, node in workflow.items() if node.get("class_type") == "SaveImage"]
    prompt_nodes = [
        node_id
        for node_id, node in workflow.items()
        if node.get("class_type") == "TextEncodeQwenImageEditPlus"
        and str(node.get("inputs", {}).get("prompt", "")).strip()
    ]

    if len(load_image_nodes) != 1:
        raise ValueError("Expected exactly one LoadImage node in the workflow.")
    if len(save_image_nodes) != 1:
        raise ValueError("Expected exactly one SaveImage node in the workflow.")
    if len(prompt_nodes) != 1:
        raise ValueError("Expected exactly one non-empty TextEncodeQwenImageEditPlus prompt node.")

    return WorkflowTargets(
        load_image_node_id=load_image_nodes[0],
        positive_prompt_node_id=prompt_nodes[0],
        save_image_node_id=save_image_nodes[0],
    )


def sanitize_attribute_suffix(attribute: str) -> str:
    safe_attribute = re.sub(r"[^0-9A-Za-z]+", "_", attribute.strip()).strip("_").lower()
    if not safe_attribute:
        raise ValueError(f"Attribute {attribute!r} cannot be turned into a safe filename suffix.")
    return safe_attribute


def validate_attributes(attributes: list[str]) -> list[str]:
    clean_attributes = [attribute.strip() for attribute in attributes if attribute.strip()]
    if not clean_attributes:
        raise ValueError("Provide at least one non-empty attribute.")

    safe_suffixes = [sanitize_attribute_suffix(attribute) for attribute in clean_attributes]
    if len(set(safe_suffixes)) != len(safe_suffixes):
        raise ValueError("Attributes must map to unique filename suffixes.")

    return clean_attributes


def build_prompt(category: str, attribute: str) -> str:
    if category == "people":
        # For people images, make the requested edit strong while freezing the rest of the scene.
        return (
            f"Edit only the people in this image so they are clearly and unmistakably '{attribute}'. "
            "Make the change strong, obvious, and visually easy to recognize on the visible people. "
            "Keep the same number of people, the same poses, the same composition, the same camera viewpoint, "
            "the same background, the same location, and the same lighting. "
            "Do not add new people, do not remove people, and do not change the scene."
        )
    # For other categories, force a visible foreground addition without changing the original place.
    return (
        f"Add one clearly visible '{attribute}' person in the foreground. "
        "The added person must be prominent, realistic, and easy to notice. "
        "Keep the original background, landmark or scene, camera viewpoint, framing, lighting, and colors unchanged. "
        "Do not modify the location, do not remove existing objects, and do not add extra people beyond the single foreground person."
    )


def build_augmented_paths(
    source_dataset_path: str,
    dataset_root: Path,
    attribute: str,
) -> tuple[Path, str]:
    source_path = resolve_dataset_path(source_dataset_path, dataset_root)
    target_name = f"{source_path.stem}_{sanitize_attribute_suffix(attribute)}.jpg"
    target_path = source_path.with_name(target_name)
    return target_path, to_relative_dataset_path(target_path, dataset_root)


def save_generated_image_as_jpg(
    image_bytes: bytes,
    output_path: Path,
    source_suffix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(suffix=source_suffix or ".png", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(image_bytes)

    try:
        image = cv2.imread(str(temp_path))
        if image is None:
            raise RuntimeError("OpenCV could not decode the image returned by ComfyUI.")
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"OpenCV could not write {output_path}.")
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attributes = validate_attributes(args.attributes)

    source_root, dataset_root, manifest_path = load_config(args.config)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    workflow_path = args.workflow.expanduser().resolve()
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow JSON does not exist: {workflow_path}")

    manifest_was_normalized = normalize_manifest_dataset_paths(manifest_path, dataset_root)
    manifest_entries = read_manifest_entries(manifest_path)

    # Clean stale augmentation rows first so reruns remain resumable.
    stale_rows_removed = False
    filtered_entries: list[ManifestEntry] = []
    for entry in manifest_entries:
        if entry.augmentation == GROUND_TRUTH_AUGMENTATION:
            filtered_entries.append(entry)
            continue

        augmented_path = resolve_dataset_path(entry.dataset_path, dataset_root)
        if augmented_path.exists():
            filtered_entries.append(entry)
        else:
            stale_rows_removed = True

    if stale_rows_removed:
        manifest_entries = filtered_entries
        write_manifest_entries(manifest_path, manifest_entries)

    manifest_by_dataset_path = {entry.dataset_path: entry for entry in manifest_entries}
    ground_truth_entries = [
        entry for entry in manifest_entries if entry.augmentation == GROUND_TRUTH_AUGMENTATION
    ]
    if not ground_truth_entries:
        print("No ground-truth dataset images found in the manifest.")
        return 0

    workflow_template = load_workflow_template(workflow_path)
    workflow_targets = find_workflow_targets(workflow_template)
    comfyui = ComfyUIClient(args.server_url)

    print(f"Source root: {source_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Manifest CSV: {manifest_path}")
    print(f"Workflow JSON: {workflow_path}")
    print(f"ComfyUI server: {args.server_url}")
    print(f"Attributes: {attributes}")
    if manifest_was_normalized:
        print("Normalized existing manifest entries before augmentation.")
    if stale_rows_removed:
        print("Removed stale augmentation rows whose files were missing.")

    generated_count = 0
    skipped_count = 0
    synced_count = 0

    for entry in ground_truth_entries:
        source_image_path = resolve_dataset_path(entry.dataset_path, dataset_root)
        if not source_image_path.exists():
            print(f"[missing] {entry.dataset_path}")
            skipped_count += len(attributes)
            continue

        for attribute in attributes:
            target_path, target_relative_path = build_augmented_paths(
                source_dataset_path=entry.dataset_path,
                dataset_root=dataset_root,
                attribute=attribute,
            )
            existing_entry = manifest_by_dataset_path.get(target_relative_path)

            # If the file is already there but the CSV row is missing, repair the manifest instead of rerunning ComfyUI.
            if existing_entry is None and target_path.exists():
                repaired_entry = ManifestEntry(
                    source_path=entry.source_path,
                    dataset_path=target_relative_path,
                    lat=entry.lat,
                    lon=entry.lon,
                    category=entry.category,
                    augmentation=attribute,
                )
                append_manifest_row(
                    manifest_path,
                    {
                        "source_path": repaired_entry.source_path,
                        "dataset_path": repaired_entry.dataset_path,
                        "lat": repaired_entry.lat,
                        "lon": repaired_entry.lon,
                        "category": repaired_entry.category,
                        "augmentation": repaired_entry.augmentation,
                    },
                )
                manifest_entries.append(repaired_entry)
                manifest_by_dataset_path[target_relative_path] = repaired_entry
                synced_count += 1
                print(f"[synced] {target_relative_path}")
                continue

            if existing_entry is not None and target_path.exists():
                skipped_count += 1
                print(f"[skip] {target_relative_path}")
                continue

            prompt = build_prompt(entry.category, attribute)
            workflow = copy.deepcopy(workflow_template)
            uploaded_image_name = comfyui.upload_input_image(source_image_path)
            workflow[workflow_targets.load_image_node_id]["inputs"]["image"] = uploaded_image_name
            workflow[workflow_targets.positive_prompt_node_id]["inputs"]["prompt"] = prompt

            prompt_id = comfyui.queue_prompt(workflow)
            image_bytes, image_info = comfyui.wait_for_output_image(
                prompt_id=prompt_id,
                save_image_node_id=workflow_targets.save_image_node_id,
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )
            save_generated_image_as_jpg(
                image_bytes=image_bytes,
                output_path=target_path,
                source_suffix=Path(str(image_info.get("filename", ""))).suffix,
            )

            new_entry = ManifestEntry(
                source_path=entry.source_path,
                dataset_path=target_relative_path,
                lat=entry.lat,
                lon=entry.lon,
                category=entry.category,
                augmentation=attribute,
            )
            append_manifest_row(
                manifest_path,
                {
                    "source_path": new_entry.source_path,
                    "dataset_path": new_entry.dataset_path,
                    "lat": new_entry.lat,
                    "lon": new_entry.lon,
                    "category": new_entry.category,
                    "augmentation": new_entry.augmentation,
                },
            )
            manifest_entries.append(new_entry)
            manifest_by_dataset_path[target_relative_path] = new_entry
            generated_count += 1
            print(f"[generated] {target_relative_path}")

    print(
        f"Done. Generated: {generated_count}, synced existing files: {synced_count}, "
        f"skipped existing rows: {skipped_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
