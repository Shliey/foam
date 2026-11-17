# Foam BGE-M3 public benchmark

**BGE-M3 = Dense Retrieval + Learned Sparse Retrieval + MaxSim Multi-vector Reranking**

## Datasets and protocol

- **DuRetrieval dev subset.** Select 64 official queries. Include the 322 passages with judgments for those queries and add deterministic background passages from the official corpus until the pool has 1,024 passages. Use every original passage as one searchable item and preserve its source ID. Compare E5 + BM25, E5 + BGE-M3 sparse, and the four BGE-M3 strategies. The MaxSim path reranks the same first 20 fused candidates. Show the first five results. Report macro Recall@20 over all known positives and nDCG@5 with unjudged passages scored as zero. [Dataset](https://huggingface.co/datasets/C-MTEB/DuRetrieval) · [Judgments](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)
- **T2Reranking_zh2en dev subset.** Choose 100 official Chinese queries with at least one positive and one negative English passage. Deduplicate and deterministically shuffle each provided candidate set. Compare E5 dense, BGE-M3 dense, and MaxSim over the identical full candidate set for each query. Use binary relevance and report MRR@10 and nDCG@5. [Dataset](https://huggingface.co/datasets/C-MTEB/T2Reranking_zh2en)
- **Timing.** Measure end-to-end under local CPU inference after model load. DuRetrieval builds its shared dense/sparse index once. MaxSim starts each measured query with an empty candidate-token cache. In T2, both methods encode the supplied candidates during the measurement. The reported figures apply to these two subsets and this machine. DuRetrieval includes known positives by construction and uses a much smaller background pool than the original benchmark.

## Results

Windows CPU, AMD Ryzen 9 8945HX, Node v22.23.5, BGE-M3 q8. DuRetrieval initial encoding took 222.6 seconds.

### DuRetrieval dev subset

| Strategy | Hit@1 | Hit@5 | Recall@20 | nDCG@5 | P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| E5 + BM25 | 62/64 | 64/64 | 0.9827 | 0.9647 | 6.8 / 9.0 |
| E5 + BGE-M3 sparse | 63/64 | 64/64 | 0.9837 | 0.9649 | 22.8 / 27.1 |
| BGE-M3 dense | 63/64 | 64/64 | 0.9991 | 0.9848 | 14.4 / 16.6 |
| BGE-M3 sparse | 62/64 | 64/64 | 0.9611 | 0.9359 | 15.0 / 18.9 |
| BGE-M3 dense + sparse | 64/64 | 64/64 | 0.9956 | 0.9840 | 16.2 / 17.8 |
| BGE-M3 dense + sparse + MaxSim | 63/64 | 64/64 | 0.9956 | 0.9845 | 9916.1 / 16489.0 |

Keeping E5 fixed and replacing BM25 with BGE-M3 sparse raised Hit@1 from 62/64 to 63/64. nDCG@5 rose from 0.9647 to 0.9649, while median query latency rose from 6.8 ms to 22.8 ms. Replacing E5 with BGE-M3 dense under the same sparse route increased nDCG@5 to 0.9840 and reached 64/64 at rank one. Compared with E5 + BM25, BGE-M3 fusion gained 1.93 percentage points in nDCG@5 and added 9.4 ms to median latency. BGE-M3 dense alone had the highest Recall@20 of the six strategies. The near-ceiling quality on this pool limits conclusions about the full corpus. Adding MaxSim moved one query's first relevant result down and raised nDCG@5 by about 0.0005. Candidate caching was cleared per reranked query, giving a P50 of 9.92 seconds.

### T2Reranking_zh2en dev subset

| Score | Hit@1 | Hit@5 | MRR@10 | nDCG@5 | P50 / P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| E5 dense | 53/100 | 95/100 | 0.7160 | 0.5627 | 313.4 / 803.3 |
| BGE-M3 dense | 54/100 | 96/100 | 0.7139 | 0.5773 | 1727.7 / 4058.3 |
| BGE-M3 MaxSim | 51/100 | 93/100 | 0.6932 | 0.5553 | 1843.4 / 4367.2 |

The paired test used a median of 14 English candidates per query. BGE-M3 dense gained one first-position hit and 1.46 percentage points in nDCG@5 over E5 dense. E5 had a slightly higher MRR@10 and a lower median latency (313 ms versus 1,728 ms). MaxSim reduced both ranking metrics relative to BGE-M3 dense and added 116 ms median latency. Model comparisons include the actual Node/ONNX configuration and the E5 subprocess IPC overhead. The baselines were measured in a separate run on the same machine, with the identical fixture verified by SHA-256.

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

```powershell
$e5Cache = Join-Path $work 'e5-models'
node -e 'const {LocalEmbedding}=require("./packages/foam-vscode/out/note-search/local");const {resolve}=require("node:path");const m=new LocalEmbedding(resolve("packages/foam-vscode/local-model/worker.cjs"), resolve(process.argv[1]));m.prepare().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>m.dispose())' $e5Cache
node packages/foam-vscode/test-data/cmteb-baselines.cjs $fixture $modelCache $e5Cache (Join-Path $work 'results.json') (Join-Path $work 'results-with-baselines.json')
```

The preparation script pins the dataset revisions and verifies each Parquet file against its SHA-256 hash. It emits a deterministic fixture using seed `foam-cmteb-v1`. The report JSON contains dataset references, aggregate results, IDs or candidate positions, and timings.
