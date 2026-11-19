import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { replayCorrelateEdgesFromLedger } from '../math/correlateEdges';

const KEY = 'cmb.correlate-ledger.v1';
class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { this.values.set(key, value); }
  removeItem(key: string) { this.values.delete(key); }
  clear() { this.values.clear(); }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
}

function detail(score = .9) {
  return { hits: [{ metric: 'pearson', lag: 0, score, pValue: .005, n: 32,
    seedNodeId: 'source', targetNodeId: 'target' }] };
}

describe('browser correlate ledger persistence', () => {
  let storage: MemoryStorage;
  beforeEach(() => {
    vi.resetModules();
    storage = new MemoryStorage();
    vi.stubGlobal('window', { localStorage: storage });
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

  it('rehydrates the latest correlate evidence after a fresh module load', async () => {
    const first = await import('./ledger');
    first.ledgerAppend({ id: 'old', at: 1000, kind: 'correlate', detail: detail(.5) });
    first.ledgerAppend({ id: 'new', at: 1001, kind: 'correlate', detail: detail(.9) });
    first.ledgerAppend({ kind: 'note', detail: { message: 'memory only' } });
    vi.resetModules();
    const reloaded = await import('./ledger');
    expect(reloaded.ledgerSnapshot().map(r => r.id)).toEqual(['new', 'old']);
    const [edge] = replayCorrelateEdgesFromLedger(reloaded.ledgerSnapshot());
    expect(edge.ledgerRef).toBe('new');
    expect(edge.score).toBe(.9);
    reloaded.ledgerClear();
    vi.resetModules();
    expect((await import('./ledger')).ledgerSnapshot()).toEqual([]);
    expect(storage.getItem(KEY)).toBeNull();
  });

  it('bounds persisted history and survives many unrelated live events', async () => {
    const ledger = await import('./ledger');
    for (let i = 0; i < 205; i++) ledger.ledgerAppend({ id: String(i), at: i, kind: 'correlate', detail: detail() });
    for (let i = 0; i < 205; i++) ledger.ledgerAppend({ kind: 'compute', detail: {} });
    vi.resetModules();
    const records = (await import('./ledger')).ledgerSnapshot();
    expect(records).toHaveLength(200);
    expect(records[0].id).toBe('204');
    expect(records[199].id).toBe('5');
  });

  it('bounds serialized bytes by removing the oldest records', async () => {
    const ledger = await import('./ledger');
    for (let i = 0; i < 4; i++) ledger.ledgerAppend({ id: String(i), at: i, kind: 'correlate',
      detail: { ...detail(), provenance: '\u00e9'.repeat(400_000) } });
    expect(new TextEncoder().encode(storage.getItem(KEY)!).byteLength).toBeLessThanOrEqual(2 * 1024 * 1024);
    vi.resetModules();
    expect((await import('./ledger')).ledgerSnapshot().map(r => r.id)).toEqual(['3', '2']);
  });

  it.each(['{', '{"schema":"other","records":[]}',
    '{"schema":"cmb.correlate-ledger/v1","records":[{"id":"bad","kind":"correlate","at":1,"detail":{"hits":[null]}}]}',
    JSON.stringify({ schema: 'cmb.correlate-ledger/v1', records: [{ id: 'invalid-date', kind: 'correlate', at: 1e100, detail: detail() }] }),
    ' '.repeat(2 * 1024 * 1024 + 1)])('ignores corrupt or oversized storage (%#)', async text => {
    storage.setItem(KEY, text);
    expect((await import('./ledger')).ledgerSnapshot()).toEqual([]);
  });

  it('keeps the live ledger usable when storage access throws', async () => {
    vi.stubGlobal('window', { get localStorage() { throw new Error('disabled'); } });
    const ledger = await import('./ledger');
    expect(() => ledger.ledgerAppend({ kind: 'correlate', detail: detail() })).not.toThrow();
    expect(ledger.ledgerSnapshot()).toHaveLength(1);
    expect(() => ledger.ledgerClear()).not.toThrow();
  });

  it('keeps the live ledger usable when a quota write fails', async () => {
    vi.spyOn(storage, 'setItem').mockImplementation(() => { throw new Error('quota'); });
    const ledger = await import('./ledger');
    ledger.ledgerAppend({ kind: 'correlate', detail: detail() });
    expect(ledger.ledgerSnapshot()).toHaveLength(1);
    expect(storage.getItem(KEY)).toBeNull();
  });

  it('preserves older evidence when a new record is oversized or cyclic', async () => {
    const ledger = await import('./ledger');
    ledger.ledgerAppend({ id: 'good', at: 1, kind: 'correlate', detail: detail() });
    ledger.ledgerAppend({ id: 'large', at: 2, kind: 'correlate', detail: { ...detail(), text: 'x'.repeat(2 * 1024 * 1024) } });
    const cyclic: Record<string, unknown> = detail();
    cyclic.self = cyclic;
    ledger.ledgerAppend({ id: 'cyclic', at: 3, kind: 'correlate', detail: cyclic });
    ledger.ledgerAppend({ id: 'next-good', at: 4, kind: 'correlate', detail: detail() });
    vi.resetModules();
    expect((await import('./ledger')).ledgerSnapshot().map(r => r.id)).toEqual(['next-good', 'good']);
  });
});
