# Foam BGE-M3 public benchmark

**BGE-M3 = Dense Retrieval + Learned Sparse Retrieval + MaxSim Multi-vector Reranking**

## Datasets and protocol

- **DuRetrieval dev subset.** Select 64 official queries. Include the 322 passages with judgments for those queries and add deterministic background passages from the official corpus until the pool has 1,024 passages. Use every original passage as one searchable item and preserve its source ID. Compare dense, learned sparse, RRF fusion, and fusion followed by MaxSim over the same first 20 fused candidates. Show the first five results. Report macro Recall@20 over all known positives and nDCG@5 with unjudged passages scored as zero. [Dataset](https://huggingface.co/datasets/C-MTEB/DuRetrieval) · [Judgments](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)
- **T2Reranking_zh2en dev subset.** Choose 100 official Chinese queries with at least one positive and one negative English passage. Deduplicate and deterministically shuffle each provided candidate set. Compare BGE-M3 dense score against MaxSim over the identical full candidate set for each query. Use binary relevance and report MRR@10 and nDCG@5. [Dataset](https://huggingface.co/datasets/C-MTEB/T2Reranking_zh2en)
- **Timing.** Measure end-to-end under local CPU inference after model load. DuRetrieval builds its shared dense/sparse index once. MaxSim starts each measured query with an empty candidate-token cache. In T2, both methods encode the supplied candidates during the measurement. The reported figures apply to these two subsets and this machine. DuRetrieval includes known positives by construction and uses a much smaller background pool than the original benchmark.

## Results

Windows CPU, AMD Ryzen 9 8945HX, Node v22.23.5, BGE-M3 q8. DuRetrieval initial encoding took 222.6 seconds.

### DuRetrieval dev subset

| Strategy | Hit@1 | Hit@5 | Recall@20 | nDCG@5 | P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 63/64 | 64/64 | 0.9991 | 0.9848 | 14.4 / 16.6 |
| sparse | 62/64 | 64/64 | 0.9611 | 0.9359 | 15.0 / 18.9 |
| fusion | 64/64 | 64/64 | 0.9956 | 0.9840 | 16.2 / 17.8 |
| rerank | 63/64 | 64/64 | 0.9956 | 0.9845 | 9916.1 / 16489.0 |

Fusion reached 64/64 first-position hits. Adding MaxSim moved one query's first relevant result down and raised nDCG@5 by about 0.0005 across the sample. Candidate caching was cleared per reranked query, producing a P50 of 9.92 seconds. The near-ceiling quality on this small pool limits conclusions about full-corpus retrieval.

### T2Reranking_zh2en dev subset

| Score | Hit@1 | Hit@5 | MRR@10 | nDCG@5 | P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 54/100 | 96/100 | 0.7139 | 0.5773 | 1727.7 / 4058.3 |
| maxsim | 51/100 | 93/100 | 0.6932 | 0.5553 | 1843.4 / 4367.2 |

The paired test used a median of 14 English candidates per query. MaxSim ranked a relevant passage first for three fewer queries, reduced MRR@10 by 0.0207, and increased median end-to-end latency by 116 ms. Dense scoring was stronger on this sample.

[Machine-readable per-query results](../../packages/foam-vscode/test-data/cmteb-results.json)

## Reproduction

From the repository root, use Node 22, Yarn 1, and Python with PyArrow. Foam's **Download Local Embedding Model** command prepares the local model cache. Set `$modelCache` to the `embedding-models` folder in the Foam extension's VS Code global storage.

```powershell
yarn install --frozen-lockfile
npm ci --prefix packages/foam-vscode/local-model
yarn workspace @foam/core build
yarn workspace @foam/graph-view build
yarn workspace foam-vscode compile
yarn workspace foam-vscode build
python -m pip install pyarrow
$work = Join-Path ([System.IO.Path]::GetTempPath()) 'foam-cmteb'
New-Item -ItemType Directory -Force $work | Out-Null
$data = Join-Path $work 'data'
$fixture = Join-Path $work 'fixture.json'
python packages/foam-vscode/test-data/prepare-cmteb.py --download $data $fixture
$modelCache = Join-Path $env:APPDATA 'Code/User/globalStorage/foam.foam-vscode/embedding-models'
node packages/foam-vscode/test-data/cmteb-bge-eval.cjs $fixture $modelCache (Join-Path $work 'results.json')
```

The preparation script pins the dataset revisions and verifies each Parquet file against its SHA-256 hash. It emits a deterministic fixture using seed `foam-cmteb-v1`. The report JSON contains dataset references, aggregate results, IDs or candidate positions, and timings.
