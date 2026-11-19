import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from audit_cmb_study import analysis_binding, audit, digest, read, verify_decisions


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/cmb-study-evidence'


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = EVIDENCE/'test'
        cls.manifest = read(run/'manifest.json')
        cls.report = read(run/'analysis.json')
        cls.predictions = read(run/'predictions.json')
        cls.tools = [json.loads(line) for line in (run/'tool-responses.jsonl').read_text().splitlines()]
        cls.features = read(run/'decisions.json')['features']
        cls.gold = read(EVIDENCE/'dataset/test-answers.json')
        cls.donors = read(run/'control-donors.json')
        cls.intervals = {key: value['ci95'] for key, value in cls.report['comparisons'].items()}
        cls.scored = []
        for pred in cls.predictions:
            target = cls.gold[pred['id']]['answer']
            cls.scored.append({
                arm: int(re.fullmatch('[+-]?[0-9]+', answer['completion'].strip()) is not None
                         and int(answer['completion'].strip()) == target)
                for arm, answer in pred['arms'].items()})

    def verify(self, report):
        verify_decisions(report, self.manifest, self.predictions, self.tools, self.features,
                         self.scored, self.gold, self.donors, self.intervals)

    def test_preserved_reports_pass_full_independent_audit(self):
        for split, n in (('dev', 24), ('test', 96)):
            with self.subTest(split=split):
                result = audit(EVIDENCE/'dataset', EVIDENCE/split)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['n'], n)
                self.assertEqual(result['analysis_integrity_binding'], 'retrospective repository anchor')

    def test_rejects_changed_decisions_even_if_report_hash_is_rebound(self):
        self.verify(self.report)
        mutations = [
            (('schema',), 'other'), (('split',), 'dev'), (('n',), 95),
            (('dataset_anchored_at_collection',), False),
            (('comparisons', 'tool_minus_ordinary', 'delta'), .99),
            (('comparisons', 'tool_minus_ordinary', 'ci95'), [.2, .3, .4]),
            (('gates', 'geometry_allocation'), True),
            (('gates', 'information_signal'), False),
            (('gates', 'correct_context_controls'), False),
            (('repair', 'repaired'), 0), (('repair', 'regressed'), 0),
            (('per_task', 'lag', 'ordinary'), 24),
            (('geometry_available',), 0), (('shuffled_answer_agreement', 'lag'), 24),
            (('arms', 'ordinary', 'correct'), 96), (('arms', 'ordinary', 'n'), 97),
            (('arms', 'ordinary', 'llm_wall_ms'), float('nan')),
            (('arms', 'ordinary', 'prompt_plus_emitted_tokens'), 0),
            (('arms', 'deterministic_tool', 'tool_compute_ms'), 0),
            (('policies', 'geometry', 'correct'), 96), (('policies', 'geometry', 'tool_calls'), 96),
            (('policies', 'geometry', 'selected_tool_compute_ms'), 0),
            (('policies', 'geometry', 'replay_llm_wall_ms'), 0),
            (('policies', 'geometry', 'prompt_plus_emitted_tokens'), 0),
            (('bootstrap', 'draws'), 0), (('bootstrap', 'seed'), 1),
            (('bootstrap', 'reselect_policies'), False), (('geometry_available',), True),
        ]
        for keys, value in mutations:
            with self.subTest(field='.'.join(keys)):
                report = copy.deepcopy(self.report)
                target = report
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    self.verify(report)
        report = copy.deepcopy(self.report)
        del report['policies']['random']
        with self.assertRaises(ValueError):
            self.verify(report)

    def test_analysis_binding_rejects_modified_report_or_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for name in ('manifest.json', 'analysis.json'):
                (run/name).write_bytes((EVIDENCE/'test'/name).read_bytes())
            manifest = read(run/'manifest.json')
            self.assertEqual(analysis_binding(run, manifest), 'retrospective repository anchor')
            (run/'analysis.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'analysis digest mismatch'):
                analysis_binding(run, manifest)
            (run/'analysis.json').write_bytes((EVIDENCE/'test/analysis.json').read_bytes())
            manifest['n'] += 1
            (run/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'not bound'):
                analysis_binding(run, manifest)

    def test_new_runs_require_embedded_analysis_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = copy.deepcopy(self.manifest)
            manifest['n'] = 2
            (run/'manifest.json').write_text(json.dumps(manifest))
            (run/'analysis.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'not bound'):
                analysis_binding(run, manifest)
            manifest['evidence_sha256']['analysis.json'] = digest(run/'analysis.json')
            self.assertEqual(analysis_binding(run, manifest), 'run manifest')
            (run/'analysis.json').write_text('{"gates": {}}')
            with self.assertRaisesRegex(ValueError, 'analysis digest mismatch'):
                analysis_binding(run, manifest)


if __name__ == '__main__':
    unittest.main()
