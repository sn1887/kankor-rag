from __future__ import annotations
import json
from pathlib import Path
SCHEMA = {
  '$schema': 'https://json-schema.org/draft/2020-12/schema',
  'title': 'Kankor corpus record',
  'type': 'object',
  'required': ['id', 'text', 'metadata'],
  'properties': {
    'id': {'type': 'string'},
    'text': {'type': 'string'},
    'metadata': {
      'type': 'object',
      'required': ['subject_category', 'subject', 'grade_band', 'language', 'source_type'],
      'properties': {
        'title': {'type': 'string'},
        'subject_category': {'type': 'string', 'enum': ['math', 'natural_science', 'social_science', 'languages']},
        'subject': {'type': 'string'},
        'grade_band': {'type': 'string'},
        'language': {'type': 'string', 'enum': ['ps', 'fa', 'ar', 'en']},
        'source_type': {'type': 'string', 'enum': ['explanation', 'definition', 'worked_example', 'practice_question']},
        'source_id': {'type': 'string'},
        'copyright': {'type': 'string'},
        'license': {'type': 'string'},
        'retrieval_weight': {'type': 'number'}
      },
      'additionalProperties': True
    }
  },
  'additionalProperties': False
}

def main() -> None:
    out_path = Path('docs/DATA_SCHEMA.json')
    out_path.write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}')

if __name__ == '__main__':
    main()
