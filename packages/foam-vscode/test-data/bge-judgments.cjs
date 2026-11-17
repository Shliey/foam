// Each row is [category, query, note, heading, source passage]. Labels are fixed before evaluation.
const groups = [
  [
    'orders',
    [
      [
        'exact',
        'Idempotency-Key',
        'Replay protection',
        'Require Idempotency-Key on POST /v2/orders. The server returns the stored response when the same key is replayed.',
      ],
      [
        'paraphrase',
        'When can customers still cancel a purchase?',
        'Cancellation window',
        'Customers may void an unshipped order until the warehouse starts packing. A packed parcel cannot be recalled through self-service.',
      ],
      [
        'cross-language',
        '下单后库存什么时候扣减？',
        'Inventory reservation',
        'Reserve stock at checkout and decrease sellable inventory once payment is confirmed. Expired reservations release the units.',
      ],
    ],
  ],
  [
    'storage',
    [
      [
        'exact',
        'DB_POOL_MAX',
        'Connection limit',
        'DB_POOL_MAX caps concurrent sessions per application worker. Raising it can exhaust the database connection budget.',
      ],
      [
        'paraphrase',
        '怎么知道备份恢复真的能用？',
        '恢复演练',
        '每月将归档还原到隔离实例，把还原结果与线上校验和比对，并记录恢复点和恢复耗时。',
      ],
      [
        'cross-language',
        'How do readers avoid seeing stale data just after an update?',
        '写后读一致性',
        '写入后立即发起的读取在一分钟内路由至主实例，降低复制延迟造成的旧值读取。',
      ],
    ],
  ],
  [
    'delivery',
    [
      [
        'exact',
        'GET /v1/events/:id',
        'Event lookup',
        'GET /v1/events/:id returns the delivery attempt history. The path accepts an event identifier, never a subscriber ID.',
      ],
      [
        'paraphrase',
        'How do we avoid losing messages after saving to the database?',
        'Transactional outbox',
        'Commit the business row and outgoing event in the same transaction. A relay publishes the outbox entry and acknowledges it afterward.',
      ],
      [
        'cross-language',
        '对端一直没有确认，多久后我们再投递？',
        'Retry schedule',
        'An unacknowledged webhook is sent again after 30 seconds. Each subsequent attempt doubles the wait up to 15 minutes.',
      ],
    ],
  ],
  [
    'deploy',
    [
      [
        'exact',
        'HTTP-5xx',
        'Abort threshold',
        'If HTTP-5xx exceeds 2% for five minutes, the controller restores traffic to the previous revision.',
      ],
      [
        'paraphrase',
        '怎么先让少量用户试试新版？',
        '分批放量',
        '首批仅给百分之一的会话分配新版本，观察错误指标后逐步提高流量占比。',
      ],
      [
        'cross-language',
        'How do we return to the previous release without redeploying it?',
        '双环境回切',
        '发布期间保留两套运行环境，失败时把入口负载均衡的目标切回原实例组。',
      ],
    ],
  ],
  [
    'security',
    [
      [
        'exact',
        'JWKS_ROTATE_DAYS',
        'Rotation cadence',
        'JWKS_ROTATE_DAYS defines the interval between planned signing-key rotations. Rotate immediately if a private key is compromised.',
      ],
      [
        'paraphrase',
        'How do old credentials stay valid when we replace a signing key?',
        'Overlap period',
        'Publish both public keys in the key set. Keep the retiring key available until every token carrying its key ID has expired.',
      ],
      [
        'cross-language',
        '账号被盗以后怎么让正在登录的设备全部下线？',
        'Session revocation',
        'Invalidate every refresh token for the account and bump the session generation counter. Devices must authenticate again.',
      ],
    ],
  ],
  [
    'network',
    [
      [
        'exact',
        'ACME-HTTP-01',
        'Certificate validation',
        'ACME-HTTP-01 validation listens on port 80. Serve the challenge route before applying application authentication.',
      ],
      [
        'paraphrase',
        '改域名记录前怎样让旧地址早点过期？',
        'DNS 缓存窗口',
        '迁移前提前降低解析记录的 TTL，等待递归解析器缓存陆续失效后再切换实际地址。',
      ],
      [
        'cross-language',
        'How do we preserve the original client address behind the proxy?',
        '代理来源地址',
        '仅信任网关写入的转发头，从受信任代理链提取真实请求源 IP，并剔除客户端伪造的字段。',
      ],
    ],
  ],
  [
    'billing',
    [
      [
        'exact',
        'PAYMENT_CAPTURE_TIMEOUT',
        'Capture timeout',
        'PAYMENT_CAPTURE_TIMEOUT means the authorization exists but capture returned no final state. Query the payment gateway before repeating capture.',
      ],
      [
        'paraphrase',
        '退款以后卡里的钱多久能回来？',
        '退款入账',
        '支付机构接收退回请求后仍需等发卡行处理。状态更新为已到账的时间由银行决定。',
      ],
      [
        'cross-language',
        'How do we find invoices that disagree with actual payments?',
        '对账差异',
        '每天把结算清单和内部账单按交易标识配对，对金额不一致或只在一侧存在的记录告警。',
      ],
    ],
  ],
  [
    'observability',
    [
      [
        'exact',
        'traceparent',
        'Trace context header',
        'Propagate traceparent across service boundaries and background jobs to associate downstream spans with the same request.',
      ],
      [
        'paraphrase',
        'How can on-call tell if a few people are waiting much longer than normal?',
        'Tail latency',
        'Track the request duration histogram and alert on the 99th percentile. Median latency hides slow outliers.',
      ],
      [
        'cross-language',
        '怎么避免告警太多，把同一故障合在一起？',
        'Alert grouping',
        'Group notifications by service, symptom and environment. Repeated occurrences update a single incident instead of paging every time.',
      ],
    ],
  ],
  [
    'cache',
    [
      [
        'exact',
        'STALE_IF_ERROR_SECONDS',
        'Stale-on-error setting',
        'STALE_IF_ERROR_SECONDS is the maximum time a stored response can be served after an origin error.',
      ],
      [
        'paraphrase',
        '怎么避免一大批缓存同一时刻失效？',
        '过期时间抖动',
        '给统一的生存周期加上独立随机扰动，降低同批键同时回源造成的峰值。',
      ],
      [
        'cross-language',
        'What happens if thousands of callers miss the same key at once?',
        '并发回源合并',
        '同一个键只允许首个请求拉取源数据，后续并发请求挂起并共享首个请求的回源结果。',
      ],
    ],
  ],
  [
    'builds',
    [
      [
        'exact',
        'BUILD_LOCK_SHA256',
        'Lockfile digest',
        'Compare BUILD_LOCK_SHA256 with the recorded dependency snapshot digest before restoring a build cache.',
      ],
      [
        'paraphrase',
        'How do we run tests only for projects touched by a change?',
        'Dependency-aware testing',
        'Compute the affected package closure from changed paths and reverse dependency edges. Execute suites only for packages in that closure.',
      ],
      [
        'cross-language',
        '如何让同一个版本在另一台电脑上生成一样的包？',
        'Reproducible builds',
        'Pin compiler and package versions, fix archive timestamps and sort inputs. Publish the digest so another machine can compare outputs.',
      ],
    ],
  ],
  [
    'search',
    [
      [
        'exact',
        'INDEX_SCHEMA_MISMATCH',
        'Schema mismatch recovery',
        'INDEX_SCHEMA_MISMATCH means cached metadata uses an older index layout. Discard the derived data and rebuild from the saved notes.',
      ],
      [
        'paraphrase',
        '搜到了以后怎么跳到当时命中的那一段？',
        '原文定位',
        '用结果里的文件 URI 和起止行打开对应的笔记选区，跳转前核对源文件内容有没有变化。',
      ],
      [
        'cross-language',
        'How can I keep searching after I delete an old document?',
        '增量删除',
        '收到笔记移除事件后，立即从文档表删掉对应片段，下次查询时这些片段不再参与候选排序。',
      ],
    ],
  ],
  [
    'queue',
    [
      [
        'exact',
        'QUEUE_VISIBILITY_MS',
        'Visibility timeout',
        'QUEUE_VISIBILITY_MS hides a borrowed task for a bounded interval so a different consumer can retry if no acknowledgement arrives.',
      ],
      [
        'paraphrase',
        'What should we do with jobs that always fail?',
        'Poison message handling',
        'After the last allowed attempt, move the task to a quarantine queue with its error context so it can be investigated without blocking healthy traffic.',
      ],
      [
        'cross-language',
        '怎么保证同一个用户的任务按顺序执行？',
        'Ordering by partition',
        'Use the account identifier as the partition key and process each partition serially so writes for one account preserve arrival order.',
      ],
    ],
  ],
];

const notes = groups.map(([name, rows]) => ({
  name: 'holdout-' + name,
  source: rows
    .map(([, , heading, body]) => {
      const prelude =
        heading === 'Poison message handling'
          ? 'The staging queue publishes daily summary messages to internal dispatch workers.\n\n'.repeat(
              100
            )
          : '';
      return prelude + '## ' + heading + '\n\n' + body;
    })
    .join('\n\n'),
}));

const judgments = groups.flatMap(([name, rows]) =>
  rows.map(([category, query, heading]) => ({
    category,
    query,
    heading,
    name: 'holdout-' + name,
  }))
);

module.exports = { notes, judgments };
