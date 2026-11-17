import { hash, type Resource, type URI } from '@foam/core';
import { passages, type Passage, type RankedPassage } from './search';

import type { BgeRetrievalModel, Representation } from './bge-protocol';

export type BgeMode = 'dense' | 'sparse' | 'fusion' | 'rerank';
const candidateLimit = 20;

export function dot(a: Float32Array, b: Float32Array) {
  if (a.length !== b.length) throw new Error('Embedding dimensions differ.');
  let score = 0;
  for (let i = 0; i < a.length; i++) score += a[i] * b[i];
  return score;
}

interface IndexedNote {
  checksum: string;
  chunks: Passage[];
}

export class BgePassageIndex {
  private notes = new Map<string, IndexedNote>();
  private vectors = new Map<string, Representation>();
  private modelKey?: string;
  private version = 0;

  set(note: Resource, source: string) {
    this.setEntry(note.uri, source, () => passages(note, source));
  }

  setPassage(uri: URI, text: string) {
    this.setEntry(uri, text, () => [
      {
        uri,
        text,
        input: text,
        heading: '',
        start: 0,
        end: 1,
      },
    ]);
  }

  private setEntry(uri: URI, source: string, chunks: () => Passage[]) {
    const checksum = hash(source);
    if (this.notes.get(uri.path)?.checksum === checksum) return;
    this.notes.set(uri.path, { checksum, chunks: chunks() });
    this.version++;
  }

  delete(uri: URI) {
    if (this.notes.delete(uri.path)) this.version++;
  }

  async rank(
    query: string,
    model: BgeRetrievalModel,
    mode: BgeMode = 'rerank'
  ) {
    if (this.modelKey !== model.key) {
      this.modelKey = model.key;
      this.vectors.clear();
    }
    while (true) {
      const version = this.version;
      const entries = [...this.notes.values()];
      const rows = entries.flatMap(({ checksum, chunks }) =>
        chunks.map(chunk => ({ chunk, checksum }))
      );
      const inputs = new Set(rows.map(row => row.chunk.input));
      for (const key of this.vectors.keys()) {
        if (!inputs.has(key)) this.vectors.delete(key);
      }
      const start = performance.now();
      const pending = [...inputs].filter(input => !this.vectors.has(input));
      for (let i = 0; i < pending.length; i += 8) {
        const batch = pending.slice(i, i + 8);
        const values = await model.encodePassages(batch);
        if (values.length !== batch.length)
          throw new Error('Incomplete BGE-M3 passage embeddings.');
        batch.forEach((input, position) =>
          this.vectors.set(input, values[position])
        );
      }
      const indexingMs = performance.now() - start;
      const queryStart = performance.now();
      const encoded = await model.encodeQuery(query, mode === 'rerank');
      const rankingStart = performance.now();
      const lexicalQuery = new Map(encoded.sparse);
      const top = (score: (value: Representation) => number, floor: number) =>
        rows
          .map((row, index) => ({
            index,
            score: score(this.vectors.get(row.chunk.input)!),
          }))
          .filter(item => item.score > floor)
          .sort((a, b) => b.score - a.score || a.index - b.index)
          .slice(0, candidateLimit);
      const lists: ReturnType<typeof top>[] = [];
      if (mode !== 'sparse')
        lists.push(top(value => dot(encoded.dense, value.dense), -Infinity));
      if (mode !== 'dense')
        lists.push(
          top(
            value =>
              value.sparse.reduce(
                (sum, [id, weight]) =>
                  sum + weight * (lexicalQuery.get(id) ?? 0),
                0
              ),
            0
          )
        );
      const fusion = new Map<number, number>();
      for (const list of lists) {
        list.forEach(({ index }, rank) =>
          fusion.set(index, (fusion.get(index) ?? 0) + 1 / (61 + rank))
        );
      }
      const candidates = [...fusion.entries()]
        .map(([index, score]) => ({ index, score }))
        .sort((a, b) => b.score - a.score || a.index - b.index)
        .slice(0, candidateLimit);
      const recallMs = performance.now() - rankingStart;
      const candidateResults = candidates.map(({ index, score }) => ({
        ...rows[index].chunk,
        checksum: rows[index].checksum,
        score,
      }));
      const reranked =
        mode === 'rerank' && candidates.length
          ? await model.rerank(
              candidates.map(({ index }) => rows[index].chunk.input),
              encoded
            )
          : undefined;
      if (version !== this.version) continue;
      if (reranked && reranked.scores.length !== candidates.length)
        throw new Error('Incomplete BGE-M3 reranking.');
      const ordered = reranked
        ? candidateResults
            .map((row, i) => ({ ...row, score: reranked.scores[i] }))
            .sort((a, b) => b.score - a.score)
        : candidateResults;
      const results: RankedPassage[] = [];
      for (const item of ordered) {
        if (
          results.some(
            previous =>
              previous.uri.path === item.uri.path &&
              previous.start < item.end &&
              item.start < previous.end
          )
        )
          continue;
        results.push(item);
        if (results.length === 5) break;
      }
      return {
        results,
        candidates: candidateResults,
        timings: {
          passagesEncoded: pending.length,
          indexingMs,
          queryEncodingMs: rankingStart - queryStart,
          recallMs,
          candidateEncodingMs: reranked?.candidateEncodingMs ?? 0,
          maxSimMs: reranked?.maxSimMs ?? 0,
          cacheHits: reranked?.cacheHits ?? 0,
        },
      };
    }
  }
}
