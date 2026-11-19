import { describe, expect, it } from 'vitest';
import {
  correlateEdgesFromHits,
  filterCorrelateEdgesByPValue,
  replayCorrelateEdgesFromLedger,
} from './correlateEdges';
import type { CorrelateHit } from './correlates';
import { scanCorrelates } from './correlates';
import type { MeaningNode, IngestedDataset } from '../ingest/types';
import type { LedgerRecord } from '../compute/ledger';

function hit(partial: Partial<CorrelateHit> = {}): CorrelateHit {
  return {
    id: 'hit-a',
    seedId: 'seed-a',
    seedLabel: 'Seed A',
    targetId: 'target-b',
    targetLabel: 'Target B',
    metric: 'pearson',
    score: 0.73,
    lag: 0,
    nullMean: 0.01,
    nullStd: 0.11,
    zToy: 3.2,
    pValue: 0.005,
    n: 32,
    epistemic: 'DERIVED/MEANING-MAP',
    seedNodeId: 'node-a',
    targetNodeId: 'node-b',
    note: 'test',
    ...partial,
  };
}

describe('correlateEdges', () => {
  it('builds stable edge ids for same source/target/metric/lag', () => {
    const a = correlateEdgesFromHits([hit()], 'led-1', '2026-01-01T00:00:00.000Z')[0]!;
    const b = correlateEdgesFromHits(
      [hit({ id: 'hit-b', score: 0.9, pValue: 0.001 })],
      'led-2',
      '2026-01-02T00:00:00.000Z'
    )[0]!;
    expect(a.id).toBe(b.id);
  });

  it('filters edges by p-value threshold', () => {
    const edges = correlateEdgesFromHits(
      [hit({ pValue: 0.002 }), hit({ id: 'hit-c', targetNodeId: 'node-c', pValue: 0.02 })],
      'led-1',
      '2026-01-01T00:00:00.000Z'
    );
    const filtered = filterCorrelateEdgesByPValue(edges, 0.01);
    expect(filtered).toHaveLength(1);
    expect(filtered[0]!.targetId).toBe('node-b');
  });

  it('replays edges from correlate ledger history', () => {
    const records: LedgerRecord[] = [
      {
        id: 'led-correlate-1',
        at: Date.parse('2026-01-01T00:00:00.000Z'),
        kind: 'correlate',
        detail: { hits: [hit()] },
      },
      {
        id: 'led-correlate-2',
        at: Date.parse('2026-01-01T00:00:01.000Z'),
        kind: 'correlate',
        detail: { hits: [hit({ score: 0.91, pValue: 0.001 })] },
      },
    ];
    const replayed = replayCorrelateEdgesFromLedger(records);
    expect(replayed).toHaveLength(1);
    expect(replayed[0]!.ledgerRef).toBe('led-correlate-2');
    expect(replayed[0]!.score).toBeCloseTo(0.91);
    expect(replayCorrelateEdgesFromLedger([...records].reverse())).toEqual(replayed);
  });

  it('keeps the newest snapshot entry when scans share a timestamp', () => {
    const records: LedgerRecord[] = [
      { id: 'new', at: 1000, kind: 'correlate', detail: { hits: [hit({ score: .9 })] } },
      { id: 'old', at: 1000, kind: 'correlate', detail: { hits: [hit({ score: .5 })] } },
    ];
    const replayed = replayCorrelateEdgesFromLedger(records);
    expect(replayed).toHaveLength(1);
    expect(replayed[0]!.ledgerRef).toBe('new');
    expect(replayed[0]!.score).toBe(.9);
  });

  it('allows a scanned series edge through the default threshold and reports paired n', () => {
    const series = Array.from({ length: 32 }, (_, i) => Math.sin(i*i*.7) + Math.cos(i*2.3));
    const target = [7, -2, ...series.slice(0, -2)];
    const dataset: IngestedDataset = {
      id: 'target', name: 'Target', type: 'generator', format: 'test', enabled: true,
      epistemic: 'DERIVED/MEANING-MAP', ingestedAt: 0,
      series: target.map((value, t) => ({ t, value })),
    };
    const node: MeaningNode = {
      id: 'target-node', label: 'Target', kind: 'time-series-window', datasetId: 'target',
      epistemic: 'DERIVED/MEANING-MAP', features: new Float32Array(),
      x: 0, y: 0, pullA: 0, pullB: 0, provenance: 'test',
    };
    const { hits } = scanCorrelates({ id: 'seed', label: 'Seed', kind: 'dataset-series',
      nodeId: 'seed-node', epistemic: 'DERIVED/MEANING-MAP', series }, [dataset], [node],
      { Cl: [], ellFocus: 2, ellMax: 2, coherenceScore: 0, coherenceZ: 0, timePhase: 0 },
      { maxLag: 3, topN: 100 });
    const match = hits.find(h => h.targetNodeId === node.id)!;
    expect(match.lag).toBe(2);
    expect(match.n).toBe(30);
    expect(match.pValue).toBeLessThan(.01);
    expect(filterCorrelateEdgesByPValue(correlateEdgesFromHits([match], 'ledger', '2026-01-01'))).toHaveLength(1);
  });

  it('allows a scanned cosine edge through the default threshold', () => {
    const features = Float32Array.from({ length: 32 }, (_, i) => Math.sin(i*i*.7) + Math.cos(i*2.3));
    const node: MeaningNode = {
      id: 'target-node', label: 'Target', kind: 'text-claim', datasetId: 'target',
      epistemic: 'DERIVED/MEANING-MAP', features,
      x: 0, y: 0, pullA: 0, pullB: 0, provenance: 'test',
    };
    const { hits } = scanCorrelates({ id: 'seed', label: 'Seed', kind: 'meaning-node',
      nodeId: 'seed-node', epistemic: 'DERIVED/MEANING-MAP', features }, [], [node],
      { Cl: [], ellFocus: 2, ellMax: 2, coherenceScore: 0, coherenceZ: 0, timePhase: 0 });
    expect(hits.find(h => h.targetNodeId === node.id)!.pValue).toBeLessThan(.01);
  });
});
