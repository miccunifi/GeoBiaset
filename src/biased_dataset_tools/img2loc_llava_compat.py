from __future__ import annotations

import sys
from pathlib import Path


IMPORT_LINE = "from transformers import AutoProcessor, AutoModelForVision2Seq"
IMPORT_COMPAT = """from transformers import AutoProcessor
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import LlavaNextForConditionalGeneration as AutoModelForVision2Seq
"""


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m biased_dataset_tools.img2loc_llava_compat SCRIPT [ARGS...]")

    script_path = Path(sys.argv[1]).expanduser().resolve()
    source = script_path.read_text(encoding="utf-8")
    if IMPORT_LINE not in source:
        raise RuntimeError(f"Could not find expected Img2Loc import line in {script_path}")

    sys.argv = sys.argv[1:]
    source = source.replace(IMPORT_LINE, IMPORT_COMPAT, 1)
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(script_path), "exec"), globals_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
