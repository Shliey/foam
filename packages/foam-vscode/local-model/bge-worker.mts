import { AutoTokenizer, env } from '@huggingface/transformers';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { parentPort, workerData } from 'node:worker_threads';
import {
  bgeDimension as width,
  bgeRevision,
  sparseWeights,
  type BgeRequest,
  type QueryRepresentation,
  type RerankResult,
} from '../src/note-search/bge-protocol.js';

const port = parentPort;
if (!port) throw new Error('Start BGE-M3 inside a Node worker.');

const cache: string = workerData.cache;
const limit = 512;
const cacheLimit = 64 * 1024 * 1024;
const revisionUrl =
  'https://huggingface.co/Xenova/bge-m3/resolve/' + bgeRevision + '/';
const assets = [
  [
    'tokenizer.json',
    '6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790',
  ],
  [
    'tokenizer_config.json',
    '7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4',
  ],
  [
    'onnx/model_quantized.onnx',
    '0826f8c1ab9edf1801db86c61919d4d108e8bfc0b809ec823ad366882ff0b77d',
  ],
] as const;
const head = (name: string) =>
  join(__dirname, 'bge-heads', 'bge-' + name + '.onnx');

async function validFile(path: string, expected: string) {
  const hash = createHash('sha256');
  try {
    for await (const chunk of createReadStream(path)) hash.update(chunk);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  }
  return hash.digest('hex') === expected;
}

async function ensureAssets(allowDownload: boolean) {
  if (allowDownload) await mkdir(join(cache, 'onnx'), { recursive: true });
  for (const [name, sha] of assets) {
    const path = join(cache, name);
    if (await validFile(path, sha)) continue;
    if (!allowDownload)
      throw new Error(
        'BGE-M3 assets are missing or invalid. Run Foam: Download Local Embedding Model.'
      );
    const temporary = path + '.' + randomUUID() + '.part';
    try {
      const response = await fetch(revisionUrl + name, {
        signal: AbortSignal.timeout(600_000),
      });
      if (!response.ok || !response.body)
        throw new Error('BGE-M3 download failed (' + response.status + ').');
      await pipeline(
        response.body,
        createWriteStream(temporary, { flags: 'wx' })
      );
      if (!(await validFile(temporary, sha)))
        throw new Error('BGE-M3 checksum mismatch for ' + name);
      await rename(temporary, path);
    } finally {
      await rm(temporary, { force: true });
    }
  }
}

async function load(allowDownload: boolean) {
  await ensureAssets(allowDownload);
  env.allowRemoteModels = false;
  const session = (path: string, threads: number) =>
    InferenceSession.create(path, {
      executionProviders: ['cpu'],
      intraOpNumThreads: threads,
      interOpNumThreads: 1,
    });
  const [tokenizer, encoder, sparse, colbert, maxsim] = await Promise.all([
    AutoTokenizer.from_pretrained(cache, { local_files_only: true }),
    session(join(cache, 'onnx', 'model_quantized.onnx'), 4),
    session(head('sparse'), 1),
    session(head('colbert'), 1),
    session(head('maxsim'), 1),
  ]);
  if (!encoder.outputNames.includes('last_hidden_state'))
    throw new Error('Unexpected BGE-M3 encoder output.');
  return { tokenizer, encoder, sparse, colbert, maxsim };
}

let runtime: ReturnType<typeof load> | undefined;
type Runtime = Awaited<ReturnType<typeof load>>;

function normalize(row: Float32Array) {
  const norm = Math.hypot(...row);
  if (norm > 0) for (let i = 0; i < row.length; i++) row[i] /= norm;
  return row;
}

function windows(text: string, tokenizer: Runtime['tokenizer']): string[] {
  if (tokenizer.encode(text, { add_special_tokens: true }).length <= limit)
    return [text];
  const chars = Array.from(text);
  const middle = Math.floor(chars.length / 2);
  return [
    ...windows(chars.slice(0, middle).join(''), tokenizer),
    ...windows(chars.slice(middle).join(''), tokenizer),
  ];
}

function concatenate(rows: Float32Array[]) {
  const merged = new Float32Array(
    rows.reduce((sum, row) => sum + row.length, 0)
  );
  let offset = 0;
  for (const row of rows) {
    merged.set(row, offset);
    offset += row.length;
  }
  return merged;
}

const sparseSpecialIds = new Set([0, 1, 2, 3]);
const matrix = (data: Float32Array) =>
  new Tensor('float32', data, [data.length / width, width]);

async function encodeBatch(
  texts: string[],
  model: Runtime,
  representation: 'retrieval' | 'all' | 'tokens'
) {
  const inputs = model.tokenizer(texts, { padding: true, truncation: false });
  const ids = inputs.input_ids.data as BigInt64Array;
  const mask = inputs.attention_mask.data as BigInt64Array;
  const shape = inputs.input_ids.dims;
  const result = await model.encoder.run({
    input_ids: new Tensor('int64', ids, shape),
    attention_mask: new Tensor('int64', mask, shape),
  });
  const state = result.last_hidden_state;
  const [batch, length, dim] = state.dims;
  if (
    batch !== texts.length ||
    dim !== width ||
    length > limit ||
    state.type !== 'float32'
  ) {
    throw new Error('Unexpected BGE-M3 hidden state shape.');
  }
  const hidden = state.data as Float32Array;
  const projections = await Promise.all([
    representation === 'tokens'
      ? undefined
      : model.sparse.run({ hidden: matrix(hidden) }),
    representation === 'retrieval'
      ? undefined
      : model.colbert.run({ hidden: matrix(hidden) }),
  ]);
  const sparse = projections[0]?.output.data as Float32Array | undefined;
  const colbert = projections[1]?.output.data as Float32Array | undefined;

  return texts.map((_, row): QueryRepresentation => {
    const offset = row * length;
    let real = 0;
    while (real < length && mask[offset + real] === 1n) real++;
    const dense =
      representation === 'tokens'
        ? new Float32Array(0)
        : normalize(hidden.slice(offset * width, (offset + 1) * width));
    const lexical = sparse
      ? sparseWeights(
          Array.from(ids.subarray(offset, offset + real), Number),
          sparse.subarray(offset, offset + real),
          sparseSpecialIds
        )
      : [];
    const tokens = colbert?.slice(
      (offset + 1) * width,
      (offset + real) * width
    );
    if (tokens)
      for (let n = 0; n < tokens.length; n += width)
        normalize(tokens.subarray(n, n + width));
    return { dense, sparse: lexical, tokens };
  });
}

async function encode(
  texts: string[],
  model: Runtime,
  representation: 'retrieval' | 'all' | 'tokens'
) {
  const groups = texts.map(text => windows(text, model.tokenizer));
  const flattened = groups.flat();
  const values: QueryRepresentation[] = [];
  for (let i = 0; i < flattened.length; i += 4) {
    values.push(
      ...(await encodeBatch(flattened.slice(i, i + 4), model, representation))
    );
  }
  let offset = 0;
  return groups.map(group => {
    const slices = values.slice(offset, offset + group.length);
    offset += group.length;
    if (slices.length === 1) return slices[0];
    const dense = new Float32Array(slices[0].dense.length);
    const weights = new Map<number, number>();
    for (const item of slices) {
      item.dense.forEach((value, i) => {
        dense[i] += value;
      });
      for (const [id, value] of item.sparse)
        weights.set(id, Math.max(weights.get(id) ?? 0, value));
    }
    return {
      dense: normalize(dense),
      sparse: [...weights],
      tokens:
        representation === 'retrieval'
          ? undefined
          : concatenate(slices.map(item => item.tokens!)),
    };
  });
}

const tokenCache = new Map<string, Float32Array>();
let cacheBytes = 0;
function cacheTokens(text: string, tokens: Float32Array) {
  if (tokens.byteLength > cacheLimit) return;
  while (cacheBytes + tokens.byteLength > cacheLimit) {
    const oldest = tokenCache.keys().next().value!;
    cacheBytes -= tokenCache.get(oldest)!.byteLength;
    tokenCache.delete(oldest);
  }
  tokenCache.set(text, tokens);
  cacheBytes += tokens.byteLength;
}

async function rerank(
  texts: string[],
  query: Float32Array,
  model: Runtime
): Promise<RerankResult> {
  const start = performance.now();
  const resolved = new Map<string, Float32Array>();
  let cacheHits = 0;
  for (const text of new Set(texts)) {
    const value = tokenCache.get(text);
    if (!value) continue;
    tokenCache.delete(text);
    tokenCache.set(text, value);
    resolved.set(text, value);
    cacheHits++;
  }
  const missing = [...new Set(texts)].filter(text => !resolved.has(text));
  const encoded = await encode(missing, model, 'tokens');
  missing.forEach((text, index) => {
    const tokens = encoded[index].tokens!;
    resolved.set(text, tokens);
    cacheTokens(text, tokens);
  });
  const scoring = performance.now();
  const scores = [];
  const queryTensor = matrix(query);
  for (const text of texts) {
    const { score } = await model.maxsim.run({
      query: queryTensor,
      passage: matrix(resolved.get(text)!),
    });
    scores.push(Number(score.data[0]));
  }
  return {
    scores,
    cacheHits,
    candidateEncodingMs: scoring - start,
    maxSimMs: performance.now() - scoring,
  };
}

async function processRequest(request: BgeRequest) {
  if (request.kind === 'clear-candidates') {
    tokenCache.clear();
    cacheBytes = 0;
    return;
  }
  runtime ??= load(request.kind === 'prepare').catch(error => {
    runtime = undefined;
    throw error;
  });
  const model = await runtime;
  switch (request.kind) {
    case 'prepare':
      return;
    case 'passages':
      return encode(request.texts, model, 'retrieval');
    case 'query':
      return (
        await encode(
          [request.text],
          model,
          request.multiVector ? 'all' : 'retrieval'
        )
      )[0];
    case 'rerank':
      return rerank(request.texts, request.tokens, model);
  }
}

let queue = Promise.resolve();
port.on('message', ({ id, request }: { id: number; request: BgeRequest }) => {
  queue = queue.then(async () => {
    try {
      port.postMessage({ id, value: await processRequest(request) });
    } catch (error) {
      port.postMessage({
        id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
});
