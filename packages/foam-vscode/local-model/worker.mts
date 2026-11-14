import { parentPort, workerData } from 'node:worker_threads';
import {
  env,
  pipeline,
  type FeatureExtractionPipeline,
} from '@huggingface/transformers';

const port = parentPort;
if (!port) throw new Error('Start this module using a Node worker.');

interface Request {
  id: number;
  texts: string[];
  role: 'passage' | 'query';
  prepare: boolean;
}

const model = 'Xenova/multilingual-e5-small';
const revision = '761b726';
let extractor: Promise<FeatureExtractionPipeline> | undefined;

function windows(
  text: string,
  prefix: string,
  encode: FeatureExtractionPipeline
): string[] {
  const input = prefix + text;
  const { input_ids } = encode.tokenizer(input, {
    return_tensor: false,
    truncation: false,
  });
  if ((input_ids as unknown as number[]).length <= 512) return [input];

  // Rare long-token passages keep all of their content and source coordinates.
  const chars = Array.from(text);
  const middle = Math.floor(chars.length / 2);
  return [
    ...windows(chars.slice(0, middle).join(''), prefix, encode),
    ...windows(chars.slice(middle).join(''), prefix, encode),
  ];
}

async function processRequest({ texts, role, prepare }: Request) {
  env.cacheDir = workerData.cache;
  env.allowRemoteModels = prepare;
  extractor ??= pipeline('feature-extraction', model, {
    revision,
    dtype: 'q8',
    device: 'cpu',
  }).catch(error => {
    extractor = undefined;
    throw error;
  });
  const encode = await extractor;
  if (prepare) return [];

  const inputs = texts.map(text => windows(text, role + ': ', encode));
  const flattened = inputs.flat();
  const vectors: number[][] = [];
  for (let i = 0; i < flattened.length; i += 8) {
    const tensor = await encode(flattened.slice(i, i + 8), {
      pooling: 'mean',
      normalize: true,
    });
    vectors.push(...(tensor.tolist() as number[][]));
  }
  let offset = 0;
  return inputs.map(group => {
    const start = offset;
    offset += group.length;
    if (group.length === 1) return vectors[start];
    const pooled = vectors[start].map((_, dim) =>
      vectors.slice(start, offset).reduce((sum, vector) => sum + vector[dim], 0)
    );
    const norm = Math.hypot(...pooled) || 1;
    return pooled.map(value => value / norm);
  });
}

let queue = Promise.resolve();
port.on('message', (request: Request) => {
  queue = queue.then(async () => {
    try {
      port.postMessage({
        id: request.id,
        vectors: await processRequest(request),
      });
    } catch (error) {
      port.postMessage({
        id: request.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
});
