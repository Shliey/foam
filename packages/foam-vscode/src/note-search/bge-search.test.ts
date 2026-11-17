import { createMarkdownParser, URI } from '@foam/core';
import { BgePassageIndex } from './bge-search';
import {
  sparseWeights,
  type BgeRetrievalModel,
  type Representation,
} from './bge-protocol';

const parser = createMarkdownParser();
const uri = URI.file('/runs.md');
const source = '## Dense\n\nImplicit topic\n\n## Sparse\n\nDB_POOL_MAX';
const encoded = (
  dense: number[],
  sparse: [number, number][] = []
): Representation => ({ dense: Float32Array.from(dense), sparse });

function fixture(): BgeRetrievalModel {
  return {
    key: 'fake-bge-v1',
    encodePassages: async texts =>
      texts.map(text =>
        text.includes('DB_POOL_MAX')
          ? encoded([0, 1], [[27, 0.5]])
          : encoded([1, 0])
      ),
    encodeQuery: async () => encoded([1, 0], [[27, 2]]),
    rerank: async texts => ({
      scores: texts.map(text => (text.includes('DB_POOL_MAX') ? 1 : -1)),
      candidateEncodingMs: 0,
      maxSimMs: 0,
      cacheHits: 0,
    }),
  };
}

it('aggregates positive token weights by maximum, excluding special IDs', () => {
  expect(
    sparseWeights(
      [0, 8, 8, 9, 2, 10],
      [5, 0.2, 0.9, 0.6, 1, 0],
      new Set([0, 2])
    )
  ).toEqual([
    [8, 0.9],
    [9, 0.6],
  ]);
});

it('recalls by either BGE-M3 route and reranks only the fused candidates', async () => {
  const index = new BgePassageIndex();
  index.set(parser.parse(uri, source), source);
  const model = fixture();
  expect((await index.rank('limit', model, 'dense')).results[0].heading).toBe(
    'Dense'
  );
  expect(
    (await index.rank('limit', model, 'sparse')).results.map(
      item => item.heading
    )
  ).toEqual(['Sparse']);
  const fusion = await index.rank('limit', model, 'fusion');
  const final = await index.rank('limit', model);
  expect(final.candidates).toEqual(fusion.candidates);
  expect(final.results.map(item => item.heading)).toEqual(['Sparse', 'Dense']);
});

it('re-encodes changed chunks and separates vectors from different models', async () => {
  const index = new BgePassageIndex();
  const model = fixture();
  index.set(parser.parse(uri, source), source);
  expect((await index.rank('limit', model)).timings.passagesEncoded).toBe(2);
  const changed = source.replace('DB_POOL_MAX', 'New limit value');
  index.set(parser.parse(uri, changed), changed);
  expect((await index.rank('limit', model)).timings.passagesEncoded).toBe(1);
  expect(
    (await index.rank('limit', { ...model, key: 'v2' })).timings.passagesEncoded
  ).toBe(2);
});

it('propagates unavailable model errors and discards a deleted result during reranking', async () => {
  const index = new BgePassageIndex();
  index.set(parser.parse(uri, source), source);
  const model = fixture();
  await expect(
    index.rank('term', {
      ...model,
      encodeQuery: async () => {
        throw new Error('Download model');
      },
    })
  ).rejects.toThrow('Download model');

  const rerank = model.rerank;
  model.rerank = async (...args) => {
    index.delete(uri);
    return rerank(...args);
  };
  expect((await index.rank('term', model)).results).toEqual([]);
});

it('indexes an externally segmented passage without changing its text or ID', async () => {
  const index = new BgePassageIndex();
  const text = '正文片段\n' + 'long original passage '.repeat(100);
  let indexed: string[] = [];
  index.setPassage(uri, text);
  const model = fixture();
  const rank = await index.rank(
    'term',
    {
      ...model,
      encodePassages: async values => {
        indexed = values;
        return model.encodePassages(values);
      },
    },
    'dense'
  );
  expect(indexed).toEqual([text]);
  expect(rank.results[0].uri).toEqual(uri);
  expect(rank.results[0].text).toBe(text);
});
