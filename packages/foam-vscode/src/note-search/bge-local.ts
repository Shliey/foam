import { Worker } from 'node:worker_threads';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import {
  bgeIndexKey,
  bgeRevision,
  type BgeRequest,
  type BgeResponse,
  type BgeRetrievalModel,
  type QueryRepresentation,
} from './bge-protocol';

export class BgeLocal implements BgeRetrievalModel {
  readonly key = bgeIndexKey;
  private worker?: Worker;
  private nextId = 0;
  private requests = new Map<
    number,
    { resolve: (value: BgeResponse) => void; reject: (error: Error) => void }
  >();
  constructor(private file: string, private cache: string) {}

  private get modelDirectory() {
    return join(this.cache, 'bge-m3', bgeRevision);
  }

  getProviderInfo() {
    return {
      name: 'BGE-M3',
      type: 'local' as const,
      model: { name: 'bge-m3', dimensions: 1024 },
    };
  }

  async isAvailable() {
    return [
      'tokenizer.json',
      'tokenizer_config.json',
      'onnx/model_quantized.onnx',
    ].every(file => existsSync(join(this.modelDirectory, file)));
  }

  async embed(text: string) {
    const { dense } = await this.encodeQuery(
      text.slice(0, 6000).normalize(),
      false
    );
    return Array.from(dense);
  }

  prepare() {
    return this.request<void>({ kind: 'prepare' });
  }

  encodePassages(texts: string[]) {
    return this.request<
      Awaited<ReturnType<BgeRetrievalModel['encodePassages']>>
    >({ kind: 'passages', texts });
  }

  encodeQuery(text: string, multiVector: boolean) {
    return this.request<QueryRepresentation>({
      kind: 'query',
      text,
      multiVector,
    });
  }

  rerank(texts: string[], query: QueryRepresentation) {
    if (!query.tokens) throw new Error('Missing query token embeddings.');
    return this.request<Awaited<ReturnType<BgeRetrievalModel['rerank']>>>({
      kind: 'rerank',
      texts,
      tokens: query.tokens,
    });
  }

  clearCandidates() {
    return this.request<void>({ kind: 'clear-candidates' });
  }

  private request<T extends BgeResponse>(request: BgeRequest): Promise<T> {
    if (!this.worker) {
      const worker = new Worker(this.file, {
        workerData: { cache: this.modelDirectory },
      });
      this.worker = worker;
      worker.on('message', ({ id, value, error }) => {
        const response = this.requests.get(id);
        if (!response) return;
        this.requests.delete(id);
        if (error) response.reject(new Error(error));
        else response.resolve(value);
      });
      const stop = (error: Error) => {
        if (this.worker !== worker) return;
        this.worker = undefined;
        for (const response of this.requests.values()) response.reject(error);
        this.requests.clear();
      };
      worker.on('error', stop);
      worker.on('exit', code =>
        stop(new Error('BGE-M3 worker exited (' + code + ').'))
      );
    }
    const id = this.nextId++;
    const worker = this.worker;
    return new Promise<T>((resolve, reject) => {
      this.requests.set(id, { resolve: value => resolve(value as T), reject });
      try {
        worker.postMessage({ id, request });
      } catch (error) {
        this.requests.delete(id);
        reject(error);
      }
    });
  }

  dispose() {
    const worker = this.worker;
    this.worker = undefined;
    for (const request of this.requests.values())
      request.reject(new Error('BGE-M3 worker closed.'));
    this.requests.clear();
    void worker?.terminate();
  }
}
