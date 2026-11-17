const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { cpus, totalmem } = require('node:os');
const { join, resolve } = require('node:path');
const { URI, createMarkdownParser } = require('@foam/core');
const { BgeLocal } = require('../out/note-search/bge-local');
const { BgePassageIndex } = require('../out/note-search/bge-search');
const { startE5 } = require('./bge-e5-runner.cjs');
const { PassageIndex, passages } = require('../out/note-search/search');
const previous = require('./section-search-eval.cjs');
const holdout = require('./bge-judgments.cjs');

const strategies = {
  A: 'E5 dense + BM25',
  B: 'E5 dense + BGE-M3 sparse',
  C: 'BGE-M3 dense',
  D: 'BGE-M3 sparse',
  E: 'BGE-M3 dense + sparse',
  F: 'BGE-M3 dense + sparse + MaxSim',
};
const labels = [
  ...previous.labels.map(([query, name, heading]) => ({
    category: 'regression',
    query,
    name,
    heading,
  })),
  ...holdout.judgments,
];
const parser = createMarkdownParser();
const documents = [
  ...Object.entries(previous.notes).map(([name, sections]) => ({
    name,
    source: sections
      .map(([title, body]) => '## ' + title + '\n\n' + body)
      .join('\n\n'),
  })),
  ...holdout.notes,
].map(({ name, source }) => ({
  name,
  source,
  resource: parser.parse(URI.file('/' + name + '.md'), source),
}));

function stats(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    p50: sorted[Math.ceil(sorted.length * 0.5) - 1],
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
  };
}

function summary(rows) {
  const byCategory = {};
  for (const [category, group] of [
    ['all', rows],
    ['holdout', rows.filter(row => row.category !== 'regression')],
    ...[...new Set(rows.map(row => row.category))].map(category => [
      category,
      rows.filter(row => row.category === category),
    ]),
  ]) {
    byCategory[category] = {
      n: group.length,
      hitAt1: group.filter(row => row.rank === 1).length,
      hitAt5: group.filter(row => row.rank > 0).length,
      recallAt20: group.filter(row => row.candidateRank > 0).length,
      warmMs: stats(group.map(row => row.warmMs)),
      ...(group[0].coldCandidateMs === undefined
        ? {}
        : {
            coldCandidateMs: stats(group.map(row => row.coldCandidateMs)),
          }),
    };
  }
  return byCategory;
}

async function run(mode, bgeCache, e5Cache) {
  const bge = new BgeLocal(
    join(__dirname, '../local-model/bge-worker.cjs'),
    resolve(bgeCache)
  );
  const e5 =
    mode === 'A' || mode === 'B' ? startE5(resolve(e5Cache)) : undefined;
  const index = mode === 'A' ? new PassageIndex() : new BgePassageIndex();
  const model =
    mode === 'B'
      ? {
          key: 'e5-dense+bge-sparse',
          async encodePassages(texts) {
            const dense = await e5.embedBatch(texts, 'passage');
            const rep = await bge.encodePassages(texts);
            return rep.map((value, i) => ({
              ...value,
              dense: Float32Array.from(dense[i]),
            }));
          },
          async encodeQuery(text) {
            const [dense] = await e5.embedBatch([text], 'query');
            const rep = await bge.encodeQuery(text, false);
            return { ...rep, dense: Float32Array.from(dense) };
          },
        }
      : bge;
  const bgeMode = { C: 'dense', D: 'sparse', F: 'rerank' }[mode] ?? 'fusion';
  const rank = async query => {
    const result = await (mode === 'A'
      ? index.rank(query, e5.embedBatch)
      : index.rank(query, model, bgeMode));
    if (mode === 'A' && !result.semantic) throw new Error(result.modelError);
    return result;
  };
  const rss = () => process.memoryUsage.rss() + (e5?.rss() ?? 0);
  let peakRssBytes = rss();
  const monitor = setInterval(() => {
    peakRssBytes = Math.max(peakRssBytes, rss());
  }, 10);
  try {
    let start = performance.now();
    if (mode === 'A' || mode === 'B') await e5.embedBatch([], 'passage');
    if (mode !== 'A') await bge.encodePassages([]);
    const coldModelLoadMs = performance.now() - start;
    start = performance.now();
    documents.forEach(doc => index.set(doc.resource, doc.source));
    const first = await rank(labels[0].query);
    const firstIndexAndQueryMs = performance.now() - start;
    const samples = [];
    for (const label of labels) {
      const match = row =>
        row.uri.path === '/' + label.name + '.md' &&
        row.heading === label.heading;
      let cold;
      let coldCandidateMs;
      if (mode === 'F') {
        await bge.clearCandidates();
        start = performance.now();
        cold = await rank(label.query);
        coldCandidateMs = performance.now() - start;
      }
      start = performance.now();
      const result = await rank(label.query);
      const warmMs = performance.now() - start;
      samples.push({
        ...label,
        rank: result.results.findIndex(match) + 1,
        candidateRank: result.candidates.findIndex(match) + 1,
        ...(cold ? { coldCandidateMs, coldTimings: cold.timings } : {}),
        warmMs,
        timings: result.timings,
        returned: result.results.map(row => ({
          note: row.uri.path,
          heading: row.heading,
          line: row.start + 1,
        })),
      });
    }
    const edited = documents.find(doc => doc.name === 'holdout-storage');
    const source = edited.source.replace(
      'DB_POOL_MAX caps',
      'DB_POOL_MAX_REVISED caps'
    );
    start = performance.now();
    index.set(parser.parse(edited.resource.uri, source), source);
    const update = await rank('DB_POOL_MAX_REVISED');
    const changedNoteAndQueryMs = performance.now() - start;
    return {
      label: strategies[mode],
      coldModelLoadMs,
      firstIndexAndQueryMs,
      firstTimings: first.timings,
      changedNoteAndQueryMs,
      updateTimings: update.timings,
      peakRssBytes: Math.max(peakRssBytes, rss()),
      groups: summary(samples),
      samples,
    };
  } finally {
    clearInterval(monitor);
    bge.dispose();
    e5?.dispose();
  }
}

async function main() {
  const [bgeCache, e5Cache, mode] = process.argv.slice(2);
  if (!bgeCache || !e5Cache)
    throw new Error(
      'Usage: node bge-search-eval.cjs <bge-cache> <e5-cache> [A-F]'
    );
  if (mode) {
    if (!(mode in strategies)) throw new Error('Unknown evaluation strategy.');
    console.log(JSON.stringify(await run(mode, bgeCache, e5Cache)));
    return;
  }
  const child = promisify(execFile);
  const results = {};
  for (const mode of Object.keys(strategies)) {
    console.error('Evaluating ' + mode + ' ' + strategies[mode]);
    const { stdout } = await child(
      process.execPath,
      [__filename, bgeCache, e5Cache, mode],
      { maxBuffer: 4 * 1024 * 1024, timeout: 300000, windowsHide: true }
    );
    results[mode] = JSON.parse(stdout);
  }
  const report = {
    platform: process.platform,
    cpu: cpus()[0].model,
    logicalCores: cpus().length,
    systemRamBytes: totalmem(),
    node: process.version,
    docs: documents.length,
    chunks: documents.reduce(
      (count, doc) => count + passages(doc.resource, doc.source).length,
      0
    ),
    queries: labels.length,
    results,
  };
  console.log(JSON.stringify(report, null, 2));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
