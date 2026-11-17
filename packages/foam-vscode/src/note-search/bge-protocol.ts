export const bgeRevision = '46ce10a0ac77ca4bab460d6676ba8260b33e68dc';
export const bgeIndexKey = bgeRevision + ':q8:baai-29fbe26:512:v1';
export const bgeDimension = 1024;

export type SparseWeights = [number, number][];
export interface Representation {
  dense: Float32Array;
  sparse: SparseWeights;
}
export interface QueryRepresentation extends Representation {
  tokens?: Float32Array;
}
export interface RerankResult {
  scores: number[];
  candidateEncodingMs: number;
  maxSimMs: number;
  cacheHits: number;
}
export interface BgeRetrievalModel {
  key: string;
  encodePassages(texts: string[]): Promise<Representation[]>;
  encodeQuery(text: string, multiVector: boolean): Promise<QueryRepresentation>;
  rerank(texts: string[], query: QueryRepresentation): Promise<RerankResult>;
}
export type BgeRequest =
  | { kind: 'prepare' }
  | { kind: 'passages'; texts: string[] }
  | { kind: 'query'; text: string; multiVector: boolean }
  | { kind: 'rerank'; texts: string[]; tokens: Float32Array }
  | { kind: 'clear-candidates' };
export type BgeResponse =
  | void
  | Representation[]
  | QueryRepresentation
  | RerankResult;

export function sparseWeights(
  ids: ArrayLike<number>,
  weights: ArrayLike<number>,
  specialIds: ReadonlySet<number>
): SparseWeights {
  const best = new Map<number, number>();
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const weight = weights[i];
    if (!specialIds.has(id) && weight > (best.get(id) ?? 0))
      best.set(id, weight);
  }
  return [...best];
}
