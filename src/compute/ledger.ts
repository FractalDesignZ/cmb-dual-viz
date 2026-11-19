/**
 * Evidence Ledger — append-only ingest / correlate / compute / dressing records.
 * Browser: bounded correlate history survives reload; other events stay in memory.
 * Node scripts: JSONL under artifacts/ledger/.
 */

import type { Epistemic, OpName, TPPolicy } from './types';

export interface LedgerRecord {
  id: string;
  at: number; // epoch ms
  kind: 'ingest' | 'correlate' | 'compute' | 'perf' | 'note' | 'dressing';
  op?: OpName | string;
  tp?: TPPolicy;
  epistemic?: Epistemic | string;
  device?: 'cpu' | 'webgpu';
  ms?: number;
  /** Dressing checklist fields (kind === 'dressing') */
  nodeId?: string;
  from?: string;
  to?: string;
  note?: string;
  detail: Record<string, unknown>;
}

const RING_MAX = 200;
const CORRELATE_STORAGE_KEY = 'cmb.correlate-ledger.v1';
const CORRELATE_STORAGE_BYTES = 2 * 1024 * 1024;

function browserStorage(): Storage | undefined {
  try { return typeof window === 'undefined' ? undefined : window.localStorage; }
  catch { return undefined; }
}

function validCorrelateRecord(value: unknown): value is LedgerRecord {
  if (!value || typeof value !== 'object') return false;
  const rec = value as Partial<LedgerRecord>;
  if (rec.kind !== 'correlate' || typeof rec.id !== 'string' || !rec.id || rec.id.length > 256 ||
      typeof rec.at !== 'number' || !Number.isSafeInteger(rec.at) || rec.at < 0 || rec.at > 8.64e15 ||
      !rec.detail || typeof rec.detail !== 'object' || !Array.isArray(rec.detail.hits) || rec.detail.hits.length > 128) return false;
  return rec.detail.hits.every((hit: unknown) => {
    if (!hit || typeof hit !== 'object') return false;
    const h = hit as Record<string, unknown>;
    return typeof h.metric === 'string' && ['pearson', 'spearman', 'lagged-pearson', 'cosine'].includes(h.metric) &&
      typeof h.score === 'number' && Number.isFinite(h.score) &&
      typeof h.pValue === 'number' && Number.isFinite(h.pValue) && h.pValue >= 0 && h.pValue <= 1 &&
      typeof h.lag === 'number' && Number.isSafeInteger(h.lag) &&
      typeof h.n === 'number' && Number.isSafeInteger(h.n) && h.n > 0 &&
      (h.seedNodeId === undefined || typeof h.seedNodeId === 'string') &&
      (h.targetNodeId === undefined || typeof h.targetNodeId === 'string');
  });
}

function readCorrelateHistory(): LedgerRecord[] {
  try {
    const text = browserStorage()?.getItem(CORRELATE_STORAGE_KEY);
    if (!text || new TextEncoder().encode(text).byteLength > CORRELATE_STORAGE_BYTES) return [];
    const stored: unknown = JSON.parse(text);
    if (!stored || typeof stored !== 'object') return [];
    const { schema, records } = stored as { schema?: unknown; records?: unknown };
    if (schema !== 'cmb.correlate-ledger/v1' || !Array.isArray(records) || records.length > RING_MAX ||
        !records.every(validCorrelateRecord)) return [];
    const seen = new Set<string>();
    return records.sort((a, b) => b.at - a.at).filter(rec => {
      if (seen.has(rec.id)) return false;
      seen.add(rec.id);
      return true;
    });
  } catch { return []; }
}

const correlateHistory = readCorrelateHistory();
const ring: LedgerRecord[] = correlateHistory.slice();
let seq = 0;

function nextId(kind: string): string {
  let id: string;
  do { id = `led-${kind}-${Date.now()}-${++seq}`; }
  while (ring.some(rec => rec.id === id) || correlateHistory.some(rec => rec.id === id));
  return id;
}

function persistCorrelateRecord(rec: LedgerRecord): void {
  if (!validCorrelateRecord(rec)) return;
  // Snapshot serializable bytes before updating history, so a mutable/cyclic or
  // oversized caller record cannot poison later writes or erase older evidence.
  let snapshot: LedgerRecord;
  try {
    const one = JSON.stringify({ schema: 'cmb.correlate-ledger/v1', records: [rec] });
    if (new TextEncoder().encode(one).byteLength > CORRELATE_STORAGE_BYTES) return;
    snapshot = JSON.parse(one).records[0] as LedgerRecord;
  } catch { return; }
  const previous = correlateHistory.findIndex(item => item.id === rec.id);
  if (previous >= 0) correlateHistory.splice(previous, 1);
  correlateHistory.unshift(snapshot);
  correlateHistory.length = Math.min(correlateHistory.length, RING_MAX);
  try {
    let text = JSON.stringify({ schema: 'cmb.correlate-ledger/v1', records: correlateHistory });
    while (new TextEncoder().encode(text).byteLength > CORRELATE_STORAGE_BYTES && correlateHistory.length) {
      correlateHistory.pop();
      text = JSON.stringify({ schema: 'cmb.correlate-ledger/v1', records: correlateHistory });
    }
    browserStorage()?.setItem(CORRELATE_STORAGE_KEY, text);
  } catch { /* Keep the live ledger usable when browser storage is unavailable. */ }
}

/** Append a ledger record (always in-memory; optionally mirrored by callers to JSONL). */
export function ledgerAppend(
  partial: Omit<LedgerRecord, 'id' | 'at' | 'detail'> & {
    id?: string;
    at?: number;
    detail?: Record<string, unknown>;
  }
): LedgerRecord {
  const rec: LedgerRecord = {
    id: partial.id ?? nextId(partial.kind),
    at: partial.at ?? Date.now(),
    kind: partial.kind,
    op: partial.op,
    tp: partial.tp,
    epistemic: partial.epistemic,
    device: partial.device ?? 'cpu',
    ms: partial.ms,
    nodeId: partial.nodeId,
    from: partial.from,
    to: partial.to,
    note: partial.note,
    detail: partial.detail ?? {},
  };
  ring.unshift(rec);
  if (ring.length > RING_MAX) ring.length = RING_MAX;
  if (rec.kind === 'correlate') persistCorrelateRecord(rec);
  return rec;
}

/**
 * Append a dressing transition in the canonical shape:
 * `{ kind: 'dressing', nodeId, from, to, at, note }`
 */
export function ledgerAppendDressing(entry: {
  nodeId: string;
  from: string;
  to: string;
  at?: number;
  note?: string;
}): LedgerRecord {
  const at = entry.at ?? Date.now();
  return ledgerAppend({
    kind: 'dressing',
    nodeId: entry.nodeId,
    from: entry.from,
    to: entry.to,
    note: entry.note ?? '',
    at,
    epistemic: 'RESEARCH/DERIVED',
    detail: {
      kind: 'dressing',
      nodeId: entry.nodeId,
      from: entry.from,
      to: entry.to,
      at,
      note: entry.note ?? '',
    },
  });
}

export function ledgerSnapshot(): LedgerRecord[] {
  return ring.slice();
}

export function ledgerClear(): void {
  ring.length = 0;
  correlateHistory.length = 0;
  try { browserStorage()?.removeItem(CORRELATE_STORAGE_KEY); } catch { /* Storage may be disabled. */ }
  seq = 0;
}

/** Serialize one record as a JSONL line. */
export function ledgerToJsonl(rec: LedgerRecord): string {
  return JSON.stringify(rec);
}

/** Format many records as JSONL body. */
export function ledgerDumpJsonl(recs: LedgerRecord[] = ring): string {
  return recs.map(ledgerToJsonl).join('\n') + (recs.length ? '\n' : '');
}
