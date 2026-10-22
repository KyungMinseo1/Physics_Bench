"""Run with python -m src.simulation from the project root."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_PROMPT_PATH, build_prompt, load_problems, load_system_prompt
from .runner import VLLMBackend, run_simulation

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict) or not isinstance(config.get("model_path"), str) or not config["model_path"].strip():
        raise ValueError("Configuration requires a non-empty model_path string")
    for key in ("vllm", "sampling"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"Configuration {key} must be a mapping")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve benchmark questions with their images using vLLM")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/annotations.jsonl")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen2.5-7b.yaml")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH, help="System prompt text file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--problem-id", action="append", help="Select an ID; repeat for multiple questions")
    parser.add_argument("--batch-size", type=int, default=101)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and build prompts without loading vLLM")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    system_prompt = load_system_prompt(args.prompt)
    problems = load_problems(args.dataset, args.image_root)
    if args.problem_id:
        requested = set(args.problem_id)
        missing = requested - {p.problem_id for p in problems}
        if missing:
            parser.error(f"Unknown problem IDs: {sorted(missing)}")
        problems = [p for p in problems if p.problem_id in requested]
    if args.limit is not None:
        problems = problems[:args.limit]
    if not problems:
        parser.error("No problems selected")
    if args.dry_run:
        for problem in problems:
            prompt = build_prompt(problem, system_prompt)
            print(json.dumps({"problem_id": problem.problem_id, "images": len(problem.images),
                              "messages": len(prompt)}, ensure_ascii=False))
        print(f"Validated {len(problems)} problems (no inference performed)")
        return 0
    config = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "outputs" / f"simulation-{stamp}.jsonl"
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new --output")
    backend = VLLMBackend(config)
    try:
        counts = run_simulation(problems, backend, output, model=config["model_path"], batch_size=args.batch_size,
                                system_prompt=system_prompt)
    finally:
        backend.shutdown()
    print(f"Saved: {output}")
    return 0 if counts["ok"] == len(problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
