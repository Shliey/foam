const { fork } = require('node:child_process');
const { join, resolve } = require('node:path');
const { LocalEmbedding } = require('../out/note-search/local');

function startE5(cache) {
  // ORT's Windows binding cannot share a native environment across two Node worker isolates.
  const process = fork(__filename, [cache], { windowsHide: true });
  let lastRss = 0;
  let pending;
  process.on('message', result => {
    lastRss = result.rss;
    const request = pending;
    pending = undefined;
    if (result.error) request?.reject(new Error(result.error));
    else request?.resolve(result.vectors);
  });
  process.on('error', error => {
    pending?.reject(error);
  });
  process.on('exit', code => {
    pending?.reject(new Error('E5 process exited (' + code + ').'));
  });
  return {
    embedBatch(texts, role) {
      return new Promise((resolve, reject) => {
        if (pending) return reject(new Error('E5 requests must be serial.'));
        pending = { resolve, reject };
        process.send({ texts, role }, error => {
          if (error) reject(error);
        });
      });
    },
    rss: () => lastRss,
    dispose: () => process.kill(),
  };
}

if (require.main === module) {
  const e5 = new LocalEmbedding(
    join(__dirname, '../local-model/worker.cjs'),
    resolve(process.argv[2])
  );
  process.on('message', async ({ texts, role }) => {
    try {
      const vectors = await e5.embedBatch(texts, role);
      process.send({ vectors, rss: process.memoryUsage.rss() });
    } catch (error) {
      process.send({ error: error.message, rss: process.memoryUsage.rss() });
    }
  });
}

module.exports = { startE5 };
