const assert = require('node:assert/strict');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join, resolve } = require('node:path');
const { execFileSync } = require('node:child_process');
const { BgeLocal } = require('../out/note-search/bge-local');

async function main() {
  if (!process.argv[2])
    throw new Error('Pass the prepared BGE-M3 model cache.');
  const worker = join(__dirname, '../local-model/bge-worker.cjs');
  const empty = mkdtempSync(join(tmpdir(), 'foam-bge-empty-'));
  const unavailable = new BgeLocal(worker, empty);
  const model = new BgeLocal(worker, resolve(process.argv[2]));
  try {
    await assert.rejects(
      unavailable.encodeQuery('test', false),
      /Download Local Embedding Model/
    );
    unavailable.dispose();

    const text = 'DB_POOL_MAX 连接数';
    const encoded = await model.encodeQuery(text, true);
    assert.equal(encoded.dense.length, 1024);
    assert.ok(Math.abs(Math.hypot(...encoded.dense) - 1) < 1e-5);
    assert.ok(
      encoded.sparse.length > 0 &&
        encoded.sparse.every(([id, weight]) => id > 3 && weight > 0)
    );
    assert.ok(
      encoded.tokens.length > 1024 && encoded.tokens.length % 1024 === 0
    );
    const long = await model.encodeQuery(
      'database connection pool limit '.repeat(150),
      true
    );
    assert.ok(long.tokens.length > 512 * 1024);

    const candidates = [
      'set DB_POOL_MAX to limit connections',
      'roll out a new version gradually',
    ];
    const cold = await model.rerank(candidates, encoded);
    const warm = await model.rerank(candidates, encoded);
    assert.ok(cold.scores[0] > cold.scores[1]);
    assert.deepEqual(cold.scores, warm.scores);
    assert.equal(cold.cacheHits, 0);
    assert.equal(warm.cacheHits, 2);
  } finally {
    model.dispose();
    unavailable.dispose();
    rmSync(empty, { recursive: true, force: true });
  }

  execFileSync(process.execPath, [__filename, '--maxsim'], {
    stdio: 'inherit',
    windowsHide: true,
  });
  console.log(
    'BGE-M3 offline weights, sparse tokens, multi-window encoding, candidate cache and MaxSim verified.'
  );
}

async function checkMaxSim() {
  const {
    InferenceSession,
    Tensor,
  } = require('../local-model/node_modules/onnxruntime-node');
  const test = await InferenceSession.create(
    join(__dirname, '../local-model/bge-heads/bge-maxsim.onnx'),
    {
      executionProviders: ['cpu'],
      intraOpNumThreads: 1,
    }
  );
  try {
    const row = (a, b) => [a, b, ...Array(1022).fill(0)];
    const tensor = rows =>
      new Tensor('float32', Float32Array.from(rows.flat()), [
        rows.length,
        1024,
      ]);
    const { score } = await test.run({
      query: tensor([row(1, 0), row(0, 1)]),
      passage: tensor([row(-1, 0), row(-0.6, 0.8)]),
    });
    // max(-1, -.6) and max(0, .8), then average.
    assert.ok(Math.abs(Number(score.data[0]) - 0.1) < 1e-6);
  } finally {
    await test.release();
  }
}

(process.argv[2] === '--maxsim' ? checkMaxSim() : main()).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
