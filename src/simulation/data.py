"""Read benchmark questions without exposing reference solutions to the model."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Problem:
    problem_id: str
    text: str
    choices: list[dict[str, str]]
    images: tuple[Path, ...]


def load_problems(path: Path, image_root: Path | None = None) -> list[Problem]:
    """Accept annotations.jsonl, dataset.json, or an individual annotation JSON.

    Image paths in the exported dataset are relative to the dataset root;
    individual files under annotations/ use that directory's parent by default.
    """
    root = image_root or (path.parent.parent if path.parent.name == "annotations" else path.parent)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]
    problems = []
    seen = set()
    for record in records:
        annotation = record.get("annotation", record)
        problem_id = annotation["problem_id"]
        if not isinstance(problem_id, str) or not problem_id or problem_id in seen:
            raise ValueError(f"Invalid or duplicate problem_id: {problem_id!r}")
        seen.add(problem_id)
        question = annotation["question"]
        if not isinstance(question["text"], str) or not question["text"].strip():
            raise ValueError(f"{problem_id}: empty question text")
        images = tuple((root / image["file_path"]).resolve() for image in question.get("images", []))
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(f"{problem_id}: image not found: {image}")
        problems.append(Problem(problem_id, question["text"], question.get("choices", []), images))
    return problems


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts/physics_solver.txt"


def load_system_prompt(path: Path = DEFAULT_PROMPT_PATH) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"System prompt is empty: {path}")
    return text


def build_prompt(problem: Problem, system_prompt: str | None = None) -> list[dict[str, Any]]:
    text = problem.text
    if problem.choices:
        text += "\n\n선택지:\n" + "\n".join(f"{c['id']}. {c['text']}" for c in problem.choices)
        text += "\n\n객관식 응답 규칙: answer.choice_id는 " + json.dumps(
            [c["id"] for c in problem.choices], ensure_ascii=False
        ) + " 중 정확히 하나의 문자열이어야 합니다. null이나 다른 ID는 허용되지 않습니다."
    else:
        text += "\n\n주관식 응답 규칙: answer.choice_id는 JSON null이어야 합니다."
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in problem.images:
        mime = mimetypes.guess_type(image.name)[0]
        if not mime or not mime.startswith("image/"):
            raise ValueError(f"Unsupported image type: {image}")
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return [{"role": "system", "content": load_system_prompt() if system_prompt is None else system_prompt}, {"role": "user", "content": content}]
