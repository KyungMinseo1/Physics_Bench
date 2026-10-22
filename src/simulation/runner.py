"""Batch inference and explicit output validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

from .data import Problem, build_prompt, load_system_prompt


class Backend(Protocol):
    def generate(self, prompts: list[list[dict[str, Any]]]) -> list[dict[str, Any]]: ...


class VLLMBackend:
    """Reuse the existing multiprocessing vLLM serving engine directly."""

    def __init__(self, config: dict[str, Any]) -> None:
        from vllm import SamplingParams
        from src.serving.parallel_vllm import ParallelVLLMEngine

        sampling = {"temperature": 0.0, "max_tokens": 4096, "seed": 42, **config.get("sampling", {})}
        if sampling.get("n", 1) != 1:
            raise ValueError("Simulation requires sampling.n=1")
        self.sampling_params = SamplingParams(**sampling)
        self.engine = ParallelVLLMEngine(model_path=config["model_path"], **config.get("vllm", {}))

    def generate(self, prompts: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        outputs = self.engine.run_batch(prompts, self.sampling_params)
        return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason} for o in outputs]

    def shutdown(self) -> None:
        self.engine.shutdown()


def normalize_latex_escapes(text: str) -> str:
    """Escape raw LaTeX backslashes inside JSON strings only.

    Keep already escaped pairs and JSON quote/slash/control/unicode escapes.
    Recognize commands before JSON decoding: raw \\frac and \\theta would
    otherwise silently become form-feed and tab characters.
    """
    parts: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            in_string = not in_string
        if in_string and char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            unicode_escape = re.match(r"u[0-9a-fA-F]{4}", text[index + 1:])
            command = re.match(r"[a-zA-Z]+", text[index + 1:])
            if (command and len(command.group()) > 1 and not unicode_escape) or following not in '"\\/bfnrtu':
                parts.append("\\\\")
                index += 1
                continue
            if following == "u" and not unicode_escape:
                parts.append("\\\\")
                index += 1
                continue
            parts.append(text[index:index + 2])
            index += 2
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def load_solution_json(candidate: str) -> Any:
    """Try normal parsing first, then narrowly repair two formatting errors."""
    normalized = normalize_latex_escapes(candidate)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        # Escape only literal newline/CR inside strings, preserving formatting
        # whitespace outside strings and every other JSON validation rule.
        parts: list[str] = []
        in_string = False
        escaped = False
        for char in normalized:
            if in_string and char in "\n\r":
                parts.append("\\n" if char == "\n" else "\\r")
            else:
                parts.append(char)
            if escaped:
                escaped = False
            elif in_string and char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
        repaired = "".join(parts)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            # A complete object/array followed by exactly one extra closer.
            # Do not discard another object, prose, or multiple extra brackets.
            if exc.msg == "Extra data" and repaired[exc.pos:].strip() in {"}", "]"}:
                return json.loads(repaired[:exc.pos])
            raise


def parse_solution(text: str, problem: Problem) -> dict[str, Any]:
    # Some models wrap JSON in Markdown or put a thinking section before it.
    candidate = text.rsplit("</think>", 1)[-1].strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = load_solution_json(candidate)
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object")
    steps = value.get("solution_steps")
    if not isinstance(steps, list) or not steps or any(not isinstance(s, str) or not s.strip() for s in steps):
        raise ValueError("solution_steps must contain non-empty strings")
    answer = value.get("answer")
    if not isinstance(answer, dict) or not isinstance(answer.get("text"), str) or not answer["text"].strip():
        raise ValueError("answer.text must be a non-empty string")
    if "choice_id" not in answer:
        raise ValueError("answer.choice_id is required")
    choice_id = answer["choice_id"]
    if problem.choices:
        if choice_id not in {c["id"] for c in problem.choices}:
            raise ValueError(f"Invalid answer.choice_id: {choice_id!r}")
    elif choice_id is not None:
        raise ValueError("Subjective questions require choice_id=null")
    return {"solution_steps": steps, "answer": answer}


def run_simulation(
    problems: Sequence[Problem], backend: Backend, output: Path, *, model: str, batch_size: int = 4,
    system_prompt: str | None = None,
) -> dict[str, int]:
    """Write each completed batch immediately; failed responses keep raw output."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    system_prompt = load_system_prompt() if system_prompt is None else system_prompt
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "invalid_output": 0, "truncated": 0, "error": 0}
    # Exclusive creation protects earlier experiment results.
    with output.open("x", encoding="utf-8") as handle:
        for start in range(0, len(problems), batch_size):
            batch = problems[start:start + batch_size]
            prompts = [build_prompt(p, system_prompt) for p in batch]
            try:
                responses = backend.generate(prompts)
                if len(responses) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} responses, received {len(responses)}")
            except Exception as exc:
                responses = [{"text": "", "error": f"{type(exc).__name__}: {exc}"} for _ in batch]
            for problem, response in zip(batch, responses):
                result: dict[str, Any] = {
                    "problem_id": problem.problem_id, "model": model,
                    "problem": problem.text, "choices": problem.choices,
                    "images": [str(p) for p in problem.images],
                    "raw_output": response["text"], "finish_reason": response.get("finish_reason"),
                    "solution_steps": None, "answer": None,
                }
                if "error" in response:
                    result.update(status="error", error=response["error"])
                else:
                    try:
                        result.update(parse_solution(response["text"], problem))
                        result["status"] = "ok"
                    except (ValueError, TypeError) as exc:
                        result.update(status="invalid_output", error=str(exc))
                    if response.get("finish_reason") == "length":
                        result.update(status="truncated", error="Generation reached max_tokens")
                counts[result["status"]] += 1
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Processed {min(start + batch_size, len(problems))}/{len(problems)}: {counts}", flush=True)
    return counts
