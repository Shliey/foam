const { join, resolve } = require('node:path');
const { performance } = require('node:perf_hooks');
const { URI, createMarkdownParser } = require('@foam/core');
const { LocalEmbedding } = require('../out/note-search/local');
const { PassageIndex, passages } = require('../out/note-search/search');

const notes = {
  ordering: [
    [
      'Contracts',
      'Registration, invoicing, and cancellation use an asynchronous queue.\n\n'.repeat(
        140
      ),
    ],
    [
      'Retry safety',
      'Send Idempotency-Key when retrying POST /v1/orders. The server stores the response with the key for 24 hours.',
    ],
    [
      'Message delivery',
      'A transactional outbox preserves events when workers fail after a database write.',
    ],
  ],
  auth: [
    ['Access tokens', '短期访问令牌只在内存中保留。泄露后凭据最多有效五分钟。'],
    [
      'Signing key lifecycle',
      'Roll over the JWT signing key by publishing both keys in JWKS until old tokens expire.',
    ],
  ],
  deploy: [
    ['Progressive rollout', '新版先接收少量流量。观察错误率后逐步放大流量。'],
    [
      'Abort conditions',
      'If HTTP-5xx rises above 2% for five minutes, automatically send all traffic to the previous revision.',
    ],
  ],
  store: [
    [
      'Recovery drill',
      '每个月用独立环境恢复一次备份，记录恢复耗时和最后一条成功落盘的数据。',
    ],
    [
      'Pool settings',
      'Change DB_POOL_MAX to limit open connections. The writer and readers have separate limits.',
    ],
  ],
  search: [
    [
      'Exact identifiers',
      'Support exact matching for identifiers, API paths, and error strings.',
    ],
    [
      'Source navigation',
      'Ranking results include note coordinates so the reader can open the original paragraph.',
    ],
  ],
  net: [
    [
      'DNS caching',
      '降低 DNS TTL 后，递归解析器会更早向权威服务器请求新记录。',
    ],
    [
      'Certificates',
      'ACME HTTP-01 renewals require port 80, and the validation route bypasses application authentication.',
    ],
  ],
  billing: [
    [
      'Refund timeline',
      '退款以支付渠道实际到账时间为准。银行处理期间显示处理中。',
    ],
    [
      'Reconciliation',
      'Reconcile settlements against invoices once a day. Alert on duplicate payment events.',
    ],
  ],
  cache: [
    [
      'Expiration',
      'Cache entries expire after 60 seconds. Temporary jitter avoids simultaneous expiry.',
    ],
    [
      'Stale responses',
      'When the upstream is down, previously saved responses remain useful for up to five minutes.',
    ],
  ],
};

const labels = [
  ['Idempotency-Key', 'ordering', 'Retry safety'],
  ['客户端重复下单如何避免生成两条记录', 'ordering', 'Retry safety'],
  [
    'keep event after write if publisher crashes',
    'ordering',
    'Message delivery',
  ],
  ['换签名密钥时怎么继续验证旧令牌', 'auth', 'Signing key lifecycle'],
  ['怎么把新版本先给少数用户试用', 'deploy', 'Progressive rollout'],
  ['HTTP-5xx', 'deploy', 'Abort conditions'],
  ['怎样验证手头的备份能用', 'store', 'Recovery drill'],
  ['DB_POOL_MAX', 'store', 'Pool settings'],
  ['点击检索结果跳转到笔记的出处', 'search', 'Source navigation'],
  ['HTTP-01 port 80', 'net', 'Certificates'],
  ['退款还有多久才能看到钱', 'billing', 'Refund timeline'],
  ['serve existing data during origin outage', 'cache', 'Stale responses'],
];

async function main() {
  if (!process.argv[2])
    throw new Error('Pass the directory that contains the pinned model cache.');
  const model = new LocalEmbedding(
    join(__dirname, '../local-model/worker.cjs'),
    resolve(process.argv[2])
  );
  try {
    const index = new PassageIndex();
    const parser = createMarkdownParser();
    const documents = Object.entries(notes).map(([name, sections]) => {
      const uri = URI.file('/' + name + '.md');
      const source = sections
        .map(([title, body]) => '## ' + title + '\n\n' + body)
        .join('\n\n');
      return { name, uri, source, resource: parser.parse(uri, source) };
    });
    documents.forEach(doc => index.set(doc.resource, doc.source));
    const counts = documents.map(
      doc => passages(doc.resource, doc.source).length
    );
    let start = performance.now();
    // Same embeddings and scoring, with the original whole-note 6000-character cap.
    const documentVectors = await model.embedBatch(
      documents.map(doc =>
        [doc.resource.title, doc.source]
          .filter(Boolean)
          .join('\n\n')
          .slice(0, 6000)
          .normalize()
      ),
      'passage'
    );
    const wholeColdIndexMs = performance.now() - start;
    start = performance.now();
    await index.rank(labels[0][0], model.embedBatch);
    const chunkWarmModelIndexAndFirstQueryMs = performance.now() - start;
    const report = {
      notes: documents.length,
      chunks: counts.reduce((a, b) => a + b),
      questions: labels.length,
      wholeColdIndexMs,
      chunkWarmModelIndexAndFirstQueryMs,
      modes: {},
    };
    for (const mode of ['whole', 'lexical', 'dense', 'hybrid']) {
      const samples = [];
      for (const [query, name, heading] of labels) {
        const target = '/' + name + '.md';
        start = performance.now();
        let results;
        if (mode === 'whole') {
          const [vector] = await model.embedBatch([query], 'query');
          results = documentVectors
            .map((row, i) => ({
              uri: documents[i].uri,
              score: row.reduce(
                (sum, value, dim) => sum + value * vector[dim],
                0
              ),
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
        } else {
          const ranked = await index.rank(query, model.embedBatch, mode);
          if (mode !== 'lexical' && !ranked.semantic)
            throw new Error(ranked.modelError);
          results = ranked.results;
        }
        const ms = performance.now() - start;
        const noteRank =
          results.findIndex(item => item.uri.path === target) + 1;
        const sectionRank =
          mode === 'whole'
            ? null
            : results.findIndex(
                item => item.uri.path === target && item.heading === heading
              ) + 1;
        samples.push({ query, name, heading, noteRank, sectionRank, ms });
      }
      const latency = samples.map(sample => sample.ms).sort((a, b) => a - b);
      report.modes[mode] = {
        noteHitsAt5: samples.filter(sample => sample.noteRank).length,
        sectionHitsAt5:
          mode === 'whole'
            ? null
            : samples.filter(sample => sample.sectionRank).length,
        sectionHitsAt1:
          mode === 'whole'
            ? null
            : samples.filter(sample => sample.sectionRank === 1).length,
        warmMedianMs: latency[Math.floor(latency.length / 2)],
        samples,
      };
    }
    console.log(JSON.stringify(report, null, 2));
  } finally {
    model.dispose();
  }
}

module.exports = { notes, labels };

if (require.main === module)
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
