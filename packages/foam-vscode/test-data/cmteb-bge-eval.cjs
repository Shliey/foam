const { readFileSync, writeFileSync } = require('node:fs');
const { join, resolve } = require('node:path');
const { cpus } = require('node:os');
const { URI } = require('@foam/core');
const { BgeLocal } = require('../out/note-search/bge-local');
const { BgePassageIndex, dot } = require('../out/note-search/bge-search');

const modes = ['dense', 'sparse', 'fusion', 'rerank'];
const mean = values =>
  values.reduce((sum, value) => sum + value, 0) / values.length;
const dcg = values =>
  values.reduce(
    (sum, grade, i) => sum + (2 ** grade - 1) / Math.log2(i + 2),
    0
  );
function metrics(ranking, labels, candidates = ranking, rrCutoff = 10) {
  const grades = ranking.map(id => Math.max(0, labels[id] ?? 0));
  const ideal = Object.values(labels)
    .filter(grade => grade > 0)
    .sort((a, b) => b - a);
  const target = grades.slice(0, rrCutoff).findIndex(grade => grade > 0);
  return {
    hitAt1: Number(grades[0] > 0),
    hitAt5: Number(grades.slice(0, 5).some(grade => grade > 0)),
    ndcgAt5: dcg(grades.slice(0, 5)) / dcg(ideal.slice(0, 5)),
    ['mrrAt' + rrCutoff]: target < 0 ? 0 : 1 / (target + 1),
    recallAt20:
      candidates.slice(0, 20).filter(id => labels[id] > 0).length /
      ideal.length,
  };
}
function stats(values) {
  const ranked = values.slice().sort((a, b) => a - b);
  return {
    p50: ranked[Math.ceil(ranked.length * 0.5) - 1],
    p95: ranked[Math.ceil(ranked.length * 0.95) - 1],
  };
}
function summarize(samples) {
  const averages = Object.fromEntries(
    Object.keys(samples[0].metrics).map(key => [
      key,
      mean(samples.map(row => row.metrics[key])),
    ])
  );
  return {
    n: samples.length,
    ...averages,
    latencyMs: stats(samples.map(row => row.ms)),
    samples,
  };
}

async function evaluateDu(du, model) {
  const index = new BgePassageIndex();
  for (const doc of du.corpus)
    index.setPassage(URI.file('/' + doc.id), doc.text);
  const start = performance.now();
  const first = await index.rank(du.queries[0].text, model, 'dense');
  const firstIndexAndQueryMs = performance.now() - start;
  console.error('DuRetrieval index ready.');
  const byMode = Object.fromEntries(modes.map(mode => [mode, []]));
  for (const [i, row] of du.queries.entries()) {
    if (i % 16 === 0)
      console.error('DuRetrieval queries: ' + i + '/' + du.queries.length);
    for (const mode of [
      ...modes.slice(i % modes.length),
      ...modes.slice(0, i % modes.length),
    ]) {
      if (mode === 'rerank') await model.clearCandidates();
      const started = performance.now();
      const hit = await index.rank(row.text, model, mode);
      const ms = performance.now() - started;
      const top = hit.results.map(result => result.uri.path.slice(1));
      const candidates = hit.candidates.map(result => result.uri.path.slice(1));
      byMode[mode].push({
        id: row.id,
        top,
        candidates,
        ms,
        timings: hit.timings,
        metrics: metrics(top, row.labels, candidates, 5),
      });
    }
  }
  return {
    corpusSize: du.corpus.length,
    queryCount: du.queries.length,
    fullCorpusSize: du.sourceCorpusSize,
    sourceQueryCount: du.sourceQueriesSize,
    judgedPassages: du.judgedPassages,
    firstIndexAndQueryMs,
    initialEncodingMs: first.timings.indexingMs,
    strategies: Object.fromEntries(
      modes.map(mode => [mode, summarize(byMode[mode])])
    ),
  };
}

async function evaluateT2(rows, model) {
  const denseRows = [];
  const maxsimRows = [];
  for (const [i, row] of rows.entries()) {
    if (i % 25 === 0)
      console.error('T2Reranking queries: ' + i + '/' + rows.length);
    const texts = row.candidates.map(item => item.text);
    const labels = Object.fromEntries(
      row.candidates.map(({ grade }, index) => [index, grade])
    );
    const order = scores =>
      scores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score || a.index - b.index)
        .map(({ index }) => index);
    let started = performance.now();
    const query = await model.encodeQuery(row.text, false);
    const scores = [];
    for (let offset = 0; offset < texts.length; offset += 8) {
      const encoded = await model.encodePassages(
        texts.slice(offset, offset + 8)
      );
      scores.push(...encoded.map(passage => dot(query.dense, passage.dense)));
    }
    const dense = order(scores);
    denseRows.push({
      id: row.id,
      top: dense,
      ms: performance.now() - started,
      metrics: metrics(dense, labels),
    });
    await model.clearCandidates();
    started = performance.now();
    const withTokens = await model.encodeQuery(row.text, true);
    const {
      scores: tokenScores,
      candidateEncodingMs,
      maxSimMs,
    } = await model.rerank(texts, withTokens);
    const maxsim = order(tokenScores);
    maxsimRows.push({
      id: row.id,
      top: maxsim,
      ms: performance.now() - started,
      candidateEncodingMs,
      maxSimMs,
      metrics: metrics(maxsim, labels),
    });
  }
  return {
    candidates: stats(rows.map(row => row.candidates.length)),
    strategies: { dense: summarize(denseRows), maxsim: summarize(maxsimRows) },
  };
}

async function main() {
  const [fixture, cache, output] = process.argv.slice(2);
  if (!fixture || !cache || !output)
    throw new Error(
      'Usage: node cmteb-bge-eval.cjs <fixture.json> <bge-cache> <output.json>'
    );
  const data = JSON.parse(readFileSync(fixture, 'utf8'));
  const model = new BgeLocal(
    join(__dirname, '../local-model/bge-worker.cjs'),
    resolve(cache)
  );
  try {
    const start = performance.now();
    await model.encodePassages([]);
    const modelLoadMs = performance.now() - start;
    console.error('Evaluating DuRetrieval...');
    const du = await evaluateDu(data.du, model);
    console.error('Evaluating T2Reranking_zh2en...');
    const t2 = await evaluateT2(data.t2, model);
    const result = {
      seed: data.seed,
      files: data.files,
      modelKey: model.key,
      modelLoadMs,
      hardware: {
        cpu: cpus()[0].model.trim(),
        platform: process.platform,
        node: process.version,
      },
      du,
      t2: { ...t2, sourceQueries: data.t2SourceSize },
    };
    writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  } finally {
    model.dispose();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
