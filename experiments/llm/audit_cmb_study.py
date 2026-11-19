"""Independent evidence audit; imports no collector, scorer, or policy functions.

Uses decimal raw moments for Pearson, grouped ranks for selection, and gain sums
for allocation bootstrap. It writes only the explicitly requested audit receipt.
"""
import argparse
from collections import defaultdict
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re

ARMS = ('ordinary', 'observed', 'tool', 'null', 'shuffled')
POLICIES = ('random', 'entropy', 'length', 'task_type', 'geometry', 'shuffled_geometry')
TASKS = ('lag', 'sign', 'strong', 'zero_sign')
BOOTSTRAP = {'draws': 2000, 'seed': 90210, 'strata': 'family x task', 'reselect_policies': True}
HISTORICAL_INTEGRITY = Path(__file__).resolve().parents[2]/'docs/cmb-study-evidence/analysis-integrity.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, field):
    """Compare complete structures without accepting NaN or bool-as-count values."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), field+' fields mismatch')
        for key, value in expected.items():
            compare(actual[key], value, field+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), field+' length mismatch')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, f'{field}[{i}]')
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), field+' mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, field+' mismatch')


def analysis_binding(run, manifest):
    expected = manifest['evidence_sha256'].get('analysis.json')
    if expected is not None:
        require(digest(run/'analysis.json') == expected, 'analysis digest mismatch')
        return 'run manifest'
    # These receipts predate report binding. Preserve their original bytes and
    # distinguish a retrospective repository anchor from collection-time evidence.
    catalog = read(HISTORICAL_INTEGRITY)
    require(catalog['schema'] == 'cmb.analysis-integrity/v1', 'unsupported analysis integrity schema')
    anchor = catalog['runs'].get(manifest['split'])
    require(anchor is not None and digest(run/'manifest.json') == anchor['manifest_sha256'],
            'analysis.json is not bound by the run manifest or historical anchor')
    require(digest(run/'analysis.json') == anchor['analysis_sha256'], 'historical analysis digest mismatch')
    return 'retrospective repository anchor'


def verify_decisions(report, manifest, predictions, tools, features, scored, gold, donors, intervals):
    """Independently reconstruct every reported decision and its supporting totals."""
    n = len(predictions)
    arms = {}
    for arm in ARMS:
        arms[arm] = {'correct': sum(s[arm] for s in scored), 'n': n,
                     'llm_wall_ms': sum(p['arms'][arm]['wall_ms'] for p in predictions),
                     'prompt_plus_emitted_tokens': sum(p['arms'][arm]['prompt_tokens']
                                                      + len(p['arms'][arm]['token_ids']) for p in predictions)}
    arms['deterministic_tool'] = {
        'correct': sum(t['zero_lag_sign' if p['task'] == 'zero_sign' else p['task']]
                       == gold[p['id']]['answer'] for p, t in zip(predictions, tools, strict=True)),
        'n': n, 'tool_compute_ms': sum(t['compute_ms'] for t in tools),
        'tool_batch_ms_including_startup': manifest['tool_batch_ms']}
    policies = {}
    for policy in POLICIES:
        allocation = selected(features, policy)
        policies[policy] = {
            'correct': sum(s['tool'] if i in allocation else s['observed'] for i, s in enumerate(scored)),
            'n': n, 'tool_calls': len(allocation),
            'replay_llm_wall_ms': arms['observed']['llm_wall_ms']
                                  + sum(p['arms']['tool']['wall_ms'] for i, p in enumerate(predictions) if i in allocation),
            'prompt_plus_emitted_tokens': arms['observed']['prompt_plus_emitted_tokens']
                                         + sum(p['arms']['tool']['prompt_tokens']+len(p['arms']['tool']['token_ids'])
                                               for i, p in enumerate(predictions) if i in allocation),
            'selected_tool_compute_ms': sum(t['compute_ms'] for i, t in enumerate(tools) if i in allocation)}
    comparisons = {}
    for key, ci in intervals.items():
        a, b = key.split('_minus_')
        totals = policies if a == 'geometry' else arms
        comparisons[key] = {'delta': (totals[a]['correct']-totals[b]['correct'])/n, 'ci95': ci}
    def passes(key, margin):
        return comparisons[key]['delta'] >= margin and comparisons[key]['ci95'][0] > 0
    expected = {
        'schema': 'cmb.allocation-result/v1', 'split': manifest['split'], 'n': n,
        'dataset_anchored_at_collection': 'dataset_manifest_sha256' in manifest,
        'arms': arms, 'policies': policies, 'comparisons': comparisons, 'bootstrap': BOOTSTRAP,
        'gates': {'information_signal': passes('tool_minus_ordinary', .10),
                  'correct_context_controls': passes('tool_minus_null', .10) and passes('tool_minus_shuffled', .10),
                  'geometry_allocation': passes('geometry_minus_entropy', .05) and passes('geometry_minus_task_type', .05)},
        'repair': {'repaired': sum(not s['ordinary'] and s['tool'] for s in scored),
                   'regressed': sum(s['ordinary'] and not s['tool'] for s in scored)},
        'geometry_available': sum(f['geometry_available'] for f in features),
        'shuffled_answer_agreement': {
            task: sum(gold[p['id']]['answer'] == gold[donors[p['id']]]['answer']
                      for p in predictions if p['task'] == task) for task in TASKS},
        'per_task': {task: {arm: sum(s[arm] for p, s in zip(predictions, scored, strict=True) if p['task'] == task)
                            for arm in ARMS} for task in TASKS}}
    for field, value in expected.items():
        require(field in report, 'missing report field: '+field)
        compare(report[field], value, field)


def decimal_correlation(x, y):
    with localcontext() as context:
        context.prec = 60
        a, b = [Decimal(str(v)) for v in x], [Decimal(str(v)) for v in y]
        n = Decimal(len(a))
        sx, sy = sum(a), sum(b)
        cov = n * sum(u*v for u, v in zip(a, b)) - sx*sy
        va = n * sum(u*u for u in a) - sx*sx
        vb = n * sum(v*v for v in b) - sy*sy
        require(va > 0 and vb > 0, 'undefined decimal Pearson')
        return float(cov / (va*vb).sqrt())


def measurements(row):
    a, b = row['a'], row['b']
    result = {}
    for lag in range(-row['max_lag'], row['max_lag'] + 1):
        start, end = max(0, -lag), min(len(a), len(b)-lag)
        result[lag] = decimal_correlation(a[start:end], b[start+lag:end+lag])
    lag = min(result, key=lambda v: (-abs(result[v]), v))
    sign = lambda v: int(v > 0) - int(v < 0)
    return {'lag': lag, 'correlation': result[lag], 'zero_lag_correlation': result[0],
            'strong': int(abs(result[lag]) >= .8), 'sign': sign(result[lag]),
            'zero_lag_sign': sign(result[0])}


def ranks(values):
    groups = defaultdict(list)
    for i, value in enumerate(values):
        groups[value].append(i)
    output, preceding = [0.] * len(values), 0
    for value in sorted(groups):
        indices = groups[value]
        rank = (preceding + (len(indices)-1)/2) / max(1, len(values)-1)
        for i in indices:
            output[i] = rank
        preceding += len(indices)
    return output


def selected(rows, policy):
    if policy == 'random':
        return set(random.Random(31).sample(list(range(len(rows))), len(rows)//2))
    entropy, drift = ranks([r['entropy'] for r in rows]), ranks([r['drift'] for r in rows])
    if policy == 'shuffled_geometry':
        random.Random(31).shuffle(drift)
    if policy == 'entropy':
        values = entropy
    elif policy == 'length':
        values = [r['length'] for r in rows]
    elif policy == 'task_type':
        values = [{'sign': 0, 'zero_sign': 1, 'strong': 2, 'lag': 3}[r['task']] for r in rows]
    else:
        require(policy in ('geometry', 'shuffled_geometry'), 'unknown policy')
        values = [(a+b)/2 for a, b in zip(entropy, drift)]
    keys = [(-values[i], rows[i]['id'], i) for i in range(len(rows))]
    return {item[2] for item in sorted(keys)[:len(rows)//2]}


def audit(data, run):
    require(decimal_correlation([1, 2, 3], [5, 7, 9]) == 1., 'analytic Pearson failed')
    require(decimal_correlation([1, 2, 3], [9, 7, 5]) == -1., 'analytic Pearson failed')
    require(ranks([2, 1, 2, 3]) == [.5, 0., .5, 1.], 'analytic tied ranks failed')
    manifest = read(run/'manifest.json')
    require(manifest['status'] == 'PASS', 'incomplete run')
    binding = analysis_binding(run, manifest)
    report = read(run/'analysis.json')
    compare(report['bootstrap'], BOOTSTRAP, 'bootstrap')
    for name, expected in manifest['evidence_sha256'].items():
        require(Path(name).name == name and digest(run/name) == expected, 'evidence digest mismatch: '+name)
    dataset = read(data/'dataset-manifest.json')
    for name, expected in dataset['sha256'].items():
        require(Path(name).name == name and digest(data/name) == expected, 'dataset digest mismatch: '+name)
    split = manifest['split']
    if 'dataset_manifest_sha256' in manifest:
        require(digest(data/'dataset-manifest.json') == manifest['dataset_manifest_sha256'], 'dataset anchor mismatch')
        require(digest(data/f'{split}-answers.json') == manifest['answers_sha256'], 'answer anchor mismatch')
    else:
        require(split == 'dev', 'confirmation dataset anchor absent')
    require(digest(data/f'{split}-inputs.json') == manifest['input_sha256'], 'run input digest mismatch')
    inputs, gold = read(data/f'{split}-inputs.json'), read(data/f'{split}-answers.json')
    predictions, decisions = read(run/'predictions.json'), read(run/'decisions.json')
    source_ids = [row['id'] for row in inputs]
    n = len(source_ids)
    require(len(set(source_ids)) == n, 'duplicate source')
    require([p['id'] for p in predictions] == source_ids, 'prediction ordering mismatch')
    require([(p['family'], p['task']) for p in predictions]
            == [(row['family'], row['task']) for row in inputs], 'prediction task/family mismatch')
    require(all(set(p['arms']) == set(ARMS) for p in predictions), 'prediction arm set mismatch')
    require([f['id'] for f in decisions['features']] == source_ids, 'feature ordering mismatch')
    require([f['task'] for f in decisions['features']] == [row['task'] for row in inputs], 'feature task mismatch')
    tools = [json.loads(line) for line in (run/'tool-responses.jsonl').read_text().splitlines()]
    require([t['id'] for t in tools] == source_ids, 'tool ordering mismatch')
    donors = read(run/'control-donors.json')
    expected_donors = {}
    for task in TASKS:
        ids = [row['id'] for row in inputs if row['task'] == task]
        random.Random(47).shuffle(ids)
        expected_donors.update({ident: ids[(j+1) % len(ids)] for j, ident in enumerate(ids)})
    compare(donors, expected_donors, 'control donors')
    require(manifest['n'] == n and manifest['parity_equal'] == n, 'manifest sample/parity count mismatch')
    max_pearson_error = 0.
    for row, tool in zip(inputs, tools):
        expected = measurements(row)
        target = 'zero_lag_sign' if row['task'] == 'zero_sign' else row['task']
        require(expected[target] == gold[row['id']]['answer'], 'independent target mismatch')
        for key in expected:
            error = abs(tool[key] - expected[key])
            max_pearson_error = max(max_pearson_error, error)
            require(error < 1e-10, 'independent tool measurement mismatch')
    traces, vector_count = {}, 0
    norm_error, drift_error, feature_error = 0., 0., 0.
    with gzip.open(run/'traces.jsonl.gz', 'rt') as stream:
        for line in stream:
            trace = json.loads(line)
            ob = trace['observation']
            require(trace['model_sha256'] == manifest['model']['sha256'], 'trace model mismatch')
            require(ob['llama_revision'] == manifest['model']['llama_revision'], 'trace runtime mismatch')
            require(ob['id'] not in traces, 'duplicate trace')
            traces[ob['id']] = ob
    require(len(traces) == 5*n, 'wrong number of traces')
    feature_rows = []
    for row, pred, saved in zip(inputs, predictions, decisions['features']):
        for arm, short in pred['arms'].items():
            raw = traces[row['id']+'-'+arm]
            for key in short:
                require(short[key] == raw[key], 'prediction/trace disagreement')
            require(raw['observe'] == (arm == 'observed'), 'observation arm mismatch')
        ob = traces[row['id']+'-observed']
        require(ob['token_ids'] == traces[row['id']+'-ordinary']['token_ids'], 'parity mismatch')
        steps, prior, ds = ob['observations'], None, []
        require(bool(steps), 'no observer states')
        require([s['token_id'] for s in steps if s['emitted']] == ob['token_ids'], 'emission mismatch')
        for i, state in enumerate(steps):
            vector = state['embedding']
            require(state['step'] == i and len(vector) == ob['embedding_dim'], 'vector shape mismatch')
            require(all(math.isfinite(v) for v in vector), 'nonfinite vector')
            norm = math.hypot(*vector)
            norm_error = max(norm_error, abs(norm - state['embedding_norm']))
            require(abs(norm - state['embedding_norm']) < 1e-7*max(1, norm), 'vector norm mismatch')
            expected_drift = None
            if prior is not None and norm and prior[1]:
                cosine = math.fsum(a*b for a, b in zip(vector, prior[0])) / (norm*prior[1])
                expected_drift = 1-min(1., max(-1., cosine))
            if expected_drift is None:
                require(state['cosine_drift'] is None, 'undefined drift mismatch')
            else:
                error = abs(expected_drift-state['cosine_drift'])
                drift_error = max(drift_error, error)
                require(error < 1e-7, 'vector drift mismatch')
                # Use saved float for frozen ranking; separately verified from vectors.
                ds.append(state['cosine_drift'])
            require(0 <= state['entropy_nats'] <= math.log(ob['vocab_size'])+1e-6, 'entropy range')
            vector_count += 1
            prior = (vector, norm)
        f = {'id': row['id'], 'task': row['task'], 'length': len(ob['token_ids']),
             'entropy': math.fsum(s['entropy_nats'] for s in steps)/len(steps)/math.log(ob['vocab_size']),
             'drift': math.fsum(ds)/len(ds)/2 if ds else 0., 'geometry_available': bool(ds)}
        for key in ('entropy', 'drift'):
            feature_error = max(feature_error, abs(f[key]-saved[key]))
            require(abs(f[key]-saved[key]) < 1e-12, 'feature recomputation mismatch')
        require(f['length'] == saved['length'] and f['geometry_available'] == saved['geometry_available'], 'feature metadata mismatch')
        # The exact persisted feature floats define tied rankings in the protocol.
        feature_rows.append(saved)
    require(set(report['policies']) == set(POLICIES), 'report policy set mismatch')
    require(set(decisions['selected']) == set(POLICIES), 'decision policy set mismatch')
    scored = []
    for pred in predictions:
        target = gold[pred['id']]['answer']
        scores = {}
        for arm, answer in pred['arms'].items():
            value = answer['completion'].strip()
            scores[arm] = int(re.fullmatch('[+-]?[0-9]+', value) is not None and int(value) == target)
        scored.append(scores)
    for policy in POLICIES:
        allocation = selected(feature_rows, policy)
        require(len(decisions['selected'][policy]) == n//2
                and {source_ids[i] for i in allocation} == set(decisions['selected'][policy]), 'allocation mismatch')
    strata = defaultdict(list)
    for i,row in enumerate(inputs):
        strata[row['family'],row['task']].append(i)
    samples = defaultdict(list)
    rng = random.Random(BOOTSTRAP['seed'])
    for _ in range(BOOTSTRAP['draws']):
        indices = [group[rng.randrange(len(group))] for group in strata.values() for _ in group]
        fs = [feature_rows[i] for i in indices]
        gains = [scored[i]['tool']-scored[i]['observed'] for i in indices]
        chosen_gains = {p: sum(gains[j] for j in selected(fs,p)) for p in ('geometry','entropy','task_type')}
        for baseline in ('entropy','task_type'):
            samples['geometry_minus_'+baseline].append((chosen_gains['geometry']-chosen_gains[baseline])/n)
        for arm in ('ordinary','null','shuffled'):
            samples['tool_minus_'+arm].append(sum(scored[i]['tool']-scored[i][arm] for i in indices)/n)
    intervals = {}
    for key, values in samples.items():
        ordered = sorted(values)
        intervals[key] = [ordered[math.floor(p*(len(values)-1))] for p in (.025,.975)]
    verify_decisions(report, manifest, predictions, tools, feature_rows, scored, gold, donors, intervals)
    return {'status': 'PASS', 'n': n, 'traces': len(traces), 'vectors': vector_count,
            'dataset_anchored_at_collection': 'dataset_manifest_sha256' in manifest,
            'analysis_integrity_binding': binding,
            'independent_pearson': '60-digit decimal raw-moment formula',
            'max_pearson_absolute_error': max_pearson_error,
            'max_vector_norm_absolute_error': norm_error, 'max_vector_drift_absolute_error': drift_error,
            'max_feature_absolute_error': feature_error, 'bootstrap_draws_reproduced': BOOTSTRAP['draws'],
            'checks': ['dataset/run evidence hashes', 'independent raw-input labels and tool values',
                       'trace/prediction identity and output parity', 'signed-vector norms and cosine drift',
                       'analysis integrity binding', 'feature aggregation', 'all arm and policy scores/costs',
                       'blinded policy selections and control donors', 'stratified reselected bootstrap intervals',
                       'comparison deltas, gates, repairs, per-task results and geometry counts'],
            'limits': ['Entropy is range/aggregation checked; raw logits were not retained',
                       'This audits recorded evidence; it is not a second model run'],
            'auditor_sha256': digest(Path(__file__)), 'analysis_sha256': digest(run/'analysis.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'run', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.run)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
