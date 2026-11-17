const { createHash } = require('node:crypto');
const { readFileSync, writeFileSync } = require('node:fs');
const { join } = require('node:path');
const { URI } = require('@foam/core');
const { BgeLocal } = require('../out/note-search/bge-local');
const { BgePassageIndex } = require('../out/note-search/bge-search');
const { PassageIndex } = require('../out/note-search/search');
const { startE5 } = require('./bge-e5-runner.cjs');
const { metrics, summarize } = require('./cmteb-bge-eval.cjs');

async function evaluateDu(index, data, rank) {
  for (const doc of data.corpus)
    index.setPassage(URI.file('/' + doc.id), doc.text);
  const start = performance.now();
  await rank(data.queries[0].text);
  const firstIndexAndQueryMs = performance.now() - start;
  const samples = [];
  for (const row of data.queries) {
    const started = performance.now();
    const hit = await rank(row.text);
    const ms = performance.now() - started;
    const top = hit.results.map(item => item.uri.path.slice(1));
    const candidates = hit.candidates.map(item => item.uri.path.slice(1));
    samples.push({
      id: row.id,
      top,
      candidates,
      ms,
      metrics: metrics(top, row.labels, candidates, 5),
    });
  }
  return { firstIndexAndQueryMs, ...summarize(samples) };
}

async function evaluateT2(rows, e5) {
  const samples = [];
  for (const row of rows) {
    const texts = row.candidates.map(item => item.text);
    const labels = Object.fromEntries(
      row.candidates.map(({ grade }, index) => [index, grade])
    );
    const started = performance.now();
    const [query] = await e5.embedBatch([row.text], 'query');
    const scores = [];
    for (let offset = 0; offset < texts.length; offset += 8) {
      const vectors = await e5.embedBatch(
        texts.slice(offset, offset + 8),
        'passage'
      );
      for (const vector of vectors)
        scores.push(
          vector.reduce((sum, value, i) => sum + value * query[i], 0)
        );
    }
    const top = scores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .map(item => item.index);
    samples.push({
      id: row.id,
      top,
      ms: performance.now() - started,
      metrics: metrics(top, labels),
    });
  }
  return summarize(samples);
}

async function main() {
  const [fixture, bgeCache, e5Cache, report, output] = process.argv.slice(2);
  if (!output)
    throw new Error(
      'Usage: node cmteb-baselines.cjs <fixture.json> <bge-cache> <e5-cache> <bge-report.json> <output.json>'
    );
  const input = readFileSync(fixture);
  const fixtureSha256 = createHash('sha256').update(input).digest('hex');
  const data = JSON.parse(input);
  const result = JSON.parse(readFileSync(report, 'utf8'));
  if (
    fixtureSha256 !==
    (result.fixtureSha256 ??
      '2129690b7c5de4d2d918abde1f25be7220ae88c62170ac70cc2634692ba60c6b')
  ) {
    throw new Error('Fixture differs from the original BGE-M3 run.');
  }
  result.fixtureSha256 = fixtureSha256;
  const model = new BgeLocal(
    join(__dirname, '../local-model/bge-worker.cjs'),
    bgeCache
  );
  const e5 = startE5(e5Cache);
  try {
    await e5.embedBatch([], 'passage');
    console.error('DuRetrieval: E5 + BM25');
    const oldIndex = new PassageIndex();
    const old = await evaluateDu(oldIndex, data.du, async text => {
      const hit = await oldIndex.rank(text, e5.embedBatch);
      if (!hit.semantic) throw new Error(hit.modelError || 'E5 unavailable.');
      return hit;
    });
    console.error('T2Reranking: E5 dense');
    const cross = await evaluateT2(data.t2, e5);
    console.error('DuRetrieval: E5 + BGE-M3 sparse');
    await model.encodePassages([]);
    const hybridIndex = new BgePassageIndex();
    const mixed = await evaluateDu(hybridIndex, data.du, text =>
      hybridIndex.rank(
        text,
        {
          key: 'E5+' + model.key,
          async encodePassages(texts) {
            const dense = await e5.embedBatch(texts, 'passage');
            const bge = await model.encodePassages(texts);
            return bge.map((value, i) => ({
              ...value,
              dense: Float32Array.from(dense[i]),
            }));
          },
          async encodeQuery(query) {
            const [dense] = await e5.embedBatch([query], 'query');
            const bge = await model.encodeQuery(query, false);
            return { ...bge, dense: Float32Array.from(dense) };
          },
        },
        'fusion'
      )
    );
    result.du.strategies = {
      ...result.du.strategies,
      e5Bm25: old,
      e5BgeSparse: mixed,
    };
    result.t2.strategies = { ...result.t2.strategies, e5Dense: cross };
    writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  } finally {
    model.dispose();
    e5.dispose();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
