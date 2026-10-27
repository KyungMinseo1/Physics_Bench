from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path
import yaml


def load_annotations(directory: Path) -> dict:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f'Annotation 폴더를 찾을 수 없습니다: {directory}')
    result = {}
    for path in sorted(directory.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        annotation = record.get('annotation', record)
        pid = annotation['problem_id']
        if pid in result:
            raise ValueError(f'Duplicate problem_id: {pid}')
        result[pid] = annotation
    if not result:
        raise ValueError(f'Annotation JSON 파일이 없습니다: {directory}')
    return result


def template(annotation: dict) -> dict:
    return {'schema_version': 2, 'problem_id': annotation['problem_id'],
            'review_status': 'complete' if annotation['question'].get('choices') else 'draft',
            'parts': []}


def normalize(value: dict, annotation: dict) -> dict:
    """Read old sidecars without requiring legacy Judge settings."""
    import copy
    result = copy.deepcopy(value)
    version = result.get('schema_version')
    if version not in (1, 2):
        raise ValueError('Unsupported schema_version')
    result['schema_version'] = 2
    result.pop('physics_setup', None)
    result.pop('relations_reviewed', None)
    for part in result.get('parts', []):
        if part.pop('relative_tolerance', 0):
            result['review_status'] = 'draft'
        part.pop('assumptions', None)
        part.setdefault('accepted_units', [])
    if annotation['question'].get('choices'):
        result['review_status'] = 'complete'
        result['parts'] = []
    return result


def validate(value: dict, annotation: dict) -> None:
    if value.get('schema_version') != 2 or value.get('problem_id') != annotation['problem_id']:
        raise ValueError('Invalid schema_version or problem_id')
    if value.get('review_status') not in ('draft', 'complete'):
        raise ValueError('Invalid review_status')
    if 'physics_setup' in value or 'relations_reviewed' in value:
        raise ValueError('Legacy Judge settings are no longer used')
    complete = value['review_status'] == 'complete'
    parts = value.get('parts')
    if not isinstance(parts, list):
        raise ValueError('parts must be a list')
    seen = set()
    for part in parts:
        pid = part.get('id')
        if not isinstance(pid, str) or not pid.strip() or pid in seen:
            raise ValueError('정답 항목 식별자가 올바르지 않습니다.')
        seen.add(pid)
        kind = part.get('kind')
        if kind not in ('numeric', 'expression') and not (kind is None and not complete):
            raise ValueError('정답 유형을 먼저 선택하세요.')
        if 'relative_tolerance' in part or 'assumptions' in part:
            raise ValueError('상대오차와 변수 가정은 더 이상 사용하지 않습니다.')
        if complete and not str(part.get('reference', '')).strip():
            raise ValueError('기준 정답을 입력하세요.')
        if kind == 'numeric':
            if not isinstance(part.get('unit_required'), bool):
                raise ValueError('단위 필수 여부를 지정하세요.')
            if complete and part['unit_required'] and not str(part.get('unit', '')).strip():
                raise ValueError('기준 단위를 입력하세요.')
            if str(part.get('reference', '')).strip() and not math.isfinite(float(part['reference'])):
                raise ValueError('기준 수치는 유한한 숫자여야 합니다.')
            number = part.get('absolute_tolerance')
            if not (number is None and not complete):
                if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
                    raise ValueError('절대 허용오차는 0 이상의 수치로 입력하세요.')
            units = part.get('accepted_units', [])
            if not isinstance(units, list):
                raise ValueError('Invalid accepted_units')
            seen_units = {str(part.get('unit', '')).strip()}
            for entry in units:
                unit = entry.get('unit', '').strip()
                factor = entry.get('factor_to_base')
                if not unit or unit in seen_units:
                    raise ValueError('추가 허용 단위는 비어 있지 않고 중복되지 않아야 합니다.')
                seen_units.add(unit)
                if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
                    raise ValueError('단위 환산 배율은 0보다 커야 합니다.')
        if kind == 'expression':
            accepted = part.get('accepted_expressions', [])
            if not isinstance(accepted, list) or any(not isinstance(e, str) or not e.strip() for e in accepted):
                raise ValueError('추가 인정 수식은 문자열 목록이어야 합니다.')
    if complete and not annotation['question'].get('choices') and not parts:
        raise ValueError('주관식 정답 기준을 입력하세요.')


def read_reference(root: Path, annotation: dict) -> dict:
    path = root / (annotation['problem_id'] + '.yaml')
    return normalize(yaml.safe_load(path.read_text()), annotation) if path.exists() else template(annotation)


def save_reference(root: Path, value: dict, annotation: dict) -> None:
    validate(value, annotation)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (annotation['problem_id'] + '.yaml')
    fd, temporary = tempfile.mkstemp(dir=root, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
