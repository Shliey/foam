import { Worker } from 'node:worker_threads';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import type { EmbeddingProvider } from '../ai/services/embedding-provider';

export class LocalEmbedding implements EmbeddingProvider {
  private worker?: Worker;
  private nextId = 0;
  private ready = false;
  private pending = new Map<
    number,
    { resolve: (vectors: number[][]) => void; reject: (error: Error) => void }
  >();

  constructor(private path: string, private cache: string) {}

  getProviderInfo() {
    return {
      name: 'Transformers.js',
      type: 'local' as const,
      model: { name: 'multilingual-e5-small', dimensions: 384 },
    };
  }

  async isAvailable() {
    return (
      this.ready ||
      existsSync(
        join(
          this.cache,
          'Xenova/multilingual-e5-small/761b726/onnx/model_quantized.onnx'
        )
      )
    );
  }

  async prepare() {
    await this.request([], 'passage', true);
    this.ready = true;
  }

  async embed(text: string) {
    const [vector] = await this.embedBatch(
      [text.slice(0, 6000).normalize()],
      'passage'
    );
    return vector;
  }

  embedBatch = async (texts: string[], role: 'query' | 'passage') => {
    if (!(await this.isAvailable()))
      throw new Error('Run Foam: Download Local Embedding Model first.');
    return this.request(texts, role, false);
  };

  private request(
    texts: string[],
    role: 'query' | 'passage',
    prepare: boolean
  ): Promise<number[][]> {
    if (!this.worker) {
      const worker = new Worker(this.path, {
        workerData: { cache: this.cache },
      });
      this.worker = worker;
      worker.on('message', ({ id, vectors, error }) => {
        const request = this.pending.get(id);
        if (!request) return;
        this.pending.delete(id);
        if (error) request.reject(new Error(error));
        else request.resolve(vectors);
      });
      const reset = (error: Error) => {
        if (this.worker !== worker) return;
        this.worker = undefined;
        for (const request of this.pending.values()) request.reject(error);
        this.pending.clear();
      };
      worker.on('error', reset);
      worker.on('exit', code =>
        reset(new Error('Embedding worker exited: ' + code))
      );
    }
    const id = this.nextId++;
    const worker = this.worker;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      try {
        worker.postMessage({ id, texts, role, prepare });
      } catch (error) {
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  dispose() {
    const worker = this.worker;
    this.worker = undefined;
    for (const request of this.pending.values())
      request.reject(new Error('Embedding worker closed'));
    this.pending.clear();
    void worker?.terminate();
  }
}
