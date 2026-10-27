from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml
from .references import load_annotations, read_reference, save_reference, validate
from .judge import (COMMON, PHYSICS, RELATIONS, PhysicsVerdict, RelationVerdict,
                    call_judge, digest, image_paths, relation_counts, validate_physics)


def load_predictions(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pid = row['problem_id']
            if pid in result:
                raise ValueError(f'Duplicate prediction: {pid}')
            result[pid] = row
    return result


def evaluate(args):
    annotations = load_annotations(args.annotations)
    predictions = load_predictions(args.predictions)
    unknown = predictions.keys() - annotations.keys()
    if unknown:
        raise ValueError(f'Unknown prediction IDs: {sorted(unknown)}')
    config = yaml.safe_load(args.judge_config.read_text())
    if not isinstance(config, dict):
        raise ValueError('Judge config must be a mapping')
    answer_config = config.get('final_answer', {})
    if not isinstance(answer_config, dict):
        raise ValueError('final_answer config must be a mapping')
    answer_method = answer_config.get('method', 'rules')
    if args.task == 'answer' and answer_method not in ('rules', 'llm'):
        raise ValueError('final_answer.method must be rules or llm')
    if (args.task != 'answer' or answer_method == 'llm') and config.get('provider') not in ('gemini', 'openrouter'):
        raise ValueError('Judge provider must be gemini or openrouter')
    ids = args.problem_id or list(annotations)
    if len(ids) != len(set(ids)) or any(pid not in annotations for pid in ids):
        raise ValueError('Invalid or duplicate requested problem IDs')
    # Judge tasks compare annotations directly; answer-sidecar settings are irrelevant.
    jobs = [(pid, annotations[pid]) for pid in ids]
    answer_criteria = {}
    if args.task == 'answer':
        from .answers import ANSWER_PROMPT
        for pid, a in jobs:
            if a['question'].get('choices'):
                correct_id = a['answer'].get('choice_id')
                if correct_id not in {c['id'] for c in a['question']['choices']}:
                    raise ValueError(f'{pid}: Invalid reference choice_id')
                continue
            criteria = read_reference(args.references, a)
            validate(criteria, a)
            if criteria['review_status'] != 'complete':
                raise ValueError(f'{pid}: 주관식 정답 기준을 작성 완료로 저장하세요.')
            answer_criteria[pid] = criteria
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'task': args.task, 'judge_config': config, 'problem_ids': [j[0] for j in jobs],
                'created_at': datetime.now(timezone.utc).isoformat(),
                'prompt': ((COMMON + '\n' + ANSWER_PROMPT) if answer_method == 'llm' else None) if args.task == 'answer' else COMMON + '\n' + (PHYSICS if args.task == 'physics' else RELATIONS),
                'answer_method': answer_method if args.task == 'answer' else None,
                'answer_criteria': answer_criteria if args.task == 'answer' else None,
                'references_hash': digest([a for _, a in jobs]),
                'predictions_hash': digest(predictions),
                'relation_source': ('explicit_relations_or_claims_from_solution_steps' if args.task == 'relations' else None)}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    scores = []; errors = 0; reviews = 0
    with (args.output / 'results.jsonl').open('w') as handle:
        for pid, a in jobs:
            row = {'problem_id': pid}
            pred = predictions.get(pid)
            try:
                failed = pred is None or pred.get('status') != 'ok'
                if args.task == 'answer':
                    from .answers import ANSWER_PROMPT, AnswerVerdict, score_rules, validate_answer_verdict
                    row['question_type'] = 'objective' if a['question'].get('choices') else 'subjective'
                    row['method'] = 'choice_id' if row['question_type'] == 'objective' else answer_method
                    if failed:
                        row.update(status='ok', score=0, reason='Missing or failed model response')
                    else:
                        answer = pred.get('answer')
                        if not isinstance(answer, dict):
                            row.update(status='ok', score=0, reason='Missing or invalid answer')
                        elif row['question_type'] == 'objective':
                            row.update(status='ok', score=int(answer.get('choice_id') == a['answer']['choice_id']),
                                       predicted_choice_id=answer.get('choice_id'), reference_choice_id=a['answer']['choice_id'])
                        else:
                            criteria = answer_criteria[pid]
                            if answer_method == 'rules':
                                verdict = score_rules(answer, criteria)
                            else:
                                # Do not provide the model solution or model name to the final-answer Judge.
                                payload = {'question': a['question'], 'annotation_answer': a['answer'],
                                           'answer_criteria': criteria, 'submission_answer': answer}
                                row['request'] = payload
                                reply = call_judge(config, ANSWER_PROMPT, payload, [], AnswerVerdict)
                                row.update(reply)
                                verdict = validate_answer_verdict(reply['value'], answer, criteria)
                            row['grading'] = verdict
                            row.update(status='needs_review' if verdict['verdict'] == 'needs_review' else 'ok',
                                       score=None if verdict['verdict'] == 'needs_review' else int(verdict['verdict'] == 'correct'))
                elif args.task == 'physics':
                    core = [s['id'] for s in a['solution_steps']]
                    steps = pred.get('solution_steps') if pred else None
                    if not failed and (not isinstance(steps, list) or any(not isinstance(s, str) for s in steps)):
                        raise ValueError('Invalid solution_steps')
                    if failed:
                        row.update(status='ok', score=0, reason='Missing or failed model response')
                    else:
                        payload = {'question': a['question'], 'reference_steps': a['solution_steps'],
                                   'submission_steps': steps}
                        row['request'] = payload
                        reply = call_judge(config, PHYSICS, payload, image_paths(a, args.annotations.parent), PhysicsVerdict)
                        row.update(reply)
                        verdict = validate_physics(reply['value'], core, steps)
                        row.update(status='needs_review' if verdict['verdict'] == 'needs_review' else 'ok',
                                   score=int(verdict['verdict'] == 'correct'))
                else:
                    relations = pred.get('relations') if pred else []
                    if failed:
                        row.update(status='ok', tp=0, fp=0, fn=len(a['visual_information']), reason='Missing or failed model response')
                    else:
                        if relations is None:
                            from .extract import Extraction, FROM_SOLUTION
                            steps = pred.get('solution_steps')
                            if not isinstance(steps, list) or not steps or any(not isinstance(s, str) for s in steps):
                                raise ValueError('Simulation requires non-empty solution_steps')
                            extraction = call_judge(config, FROM_SOLUTION, {'submission_steps': steps}, [], Extraction, system_prefix='')
                            row['relation_extraction'] = extraction
                            extracted = Extraction.model_validate(extraction['value']).model_dump()
                            relations = extracted['relations']
                            for relation in relations:
                                quote = relation['description']
                                if not quote or not any(quote in step for step in steps):
                                    raise ValueError('Extracted relation must quote the simulation solution')
                            predicted_entities = extracted['entities']
                        else:
                            predicted_entities = pred.get('entities', [])
                        if not isinstance(relations, list) or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in relations):
                            raise ValueError('Invalid predicted relations')
                        payload = {'reference_entities': a['entities'], 'reference_relations': a['visual_information'],
                                   'predicted_entities': predicted_entities, 'predicted_relations': relations}
                        row['request'] = payload
                        reply = call_judge(config, RELATIONS, payload, image_paths(a, args.annotations.parent), RelationVerdict)
                        row.update(reply)
                        row.update(relation_counts(reply['value'], a['visual_information'], relations))
                        row['status'] = 'needs_review' if reply['value']['reference_issues'] else 'ok'
                if row['status'] == 'needs_review':
                    reviews += 1
                else:
                    scores.append(row)
            except Exception as exc:
                row.update(status='evaluation_error', error=f'{type(exc).__name__}: {exc}')
                errors += 1
            handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush()
            print(f"{pid}: {row['status']}", flush=True)
    summary = {'task': args.task, 'n': len(jobs), 'scored': len(scores), 'evaluation_errors': errors,
               'needs_review': reviews, 'finalized': not (errors or reviews)}
    if args.task in ('physics', 'answer'):
        summary['accuracy'] = sum(r['score'] for r in scores) / len(scores) if scores and summary['finalized'] else None
    else:
        tp, fp, fn = (sum(r[k] for r in scores) for k in ('tp', 'fp', 'fn'))
        summary.update(tp=tp, fp=fp, fn=fn)
        for name, numerator, denominator in [('precision', tp, tp+fp), ('recall', tp, tp+fn), ('f1', 2*tp, 2*tp+fp+fn)]:
            summary[name] = numerator / denominator if denominator and summary['finalized'] else None
    if args.task == 'answer':
        summary['answer_method'] = answer_method
        summary['by_question_type'] = {}
        for kind in ('objective', 'subjective'):
            subset = [r for r in scores if r['question_type'] == kind]
            expected_count = sum(bool(a['question'].get('choices')) == (kind == 'objective') for _, a in jobs)
            summary['by_question_type'][kind] = {'n': expected_count, 'scored': len(subset),
                'accuracy': sum(r['score'] for r in subset) / len(subset) if subset and len(subset) == expected_count else None}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors or reviews else 0


def main():
    parser = argparse.ArgumentParser(description='Physics evaluation references and Judge')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init', 'validate', 'serve', 'judge', 'extract'):
        p = sub.add_parser(name)
        p.add_argument('--annotations', type=Path, default=Path(__file__).resolve().parents[2] / 'dataset/annotations')
        p.add_argument('--references', type=Path, default=Path(__file__).resolve().parents[2] / 'dataset/evaluation')
        if name == 'serve':
            p.add_argument('--port', type=int, default=8010)
        if name == 'judge':
            p.add_argument('--task', choices=['physics', 'relations', 'answer'], required=True)
            p.add_argument('--predictions', type=Path, required=True)
        if name in ('judge', 'extract'):
            p.add_argument('--judge-config', type=Path, default=Path('config/evaluation/judge.yaml'))
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--problem-id', action='append')
    args = parser.parse_args()
    try:
        if args.command == 'serve':
            from .server import serve
            serve(args.annotations, args.references, args.port)
            return 0
        if args.command == 'judge':
            return evaluate(args)
        if args.command == 'extract':
            from .extract import extract
            return extract(args)
        annotations = load_annotations(args.annotations)
        for a in annotations.values():
            value = read_reference(args.references, a)
            if args.command == 'init':
                if not (args.references / (a['problem_id'] + '.yaml')).exists():
                    save_reference(args.references, value, a)
            else:
                if not (args.references / (a['problem_id'] + '.yaml')).exists():
                    raise ValueError(f"Missing reference: {a['problem_id']}")
                validate(value, a)
        print(f'{args.command}: {len(annotations)} problems')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
