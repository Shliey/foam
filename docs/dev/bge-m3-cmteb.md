# Foam BGE-M3 public benchmark

**BGE-M3 = Dense Retrieval + Learned Sparse Retrieval + MaxSim Multi-vector Reranking**

## DuRetrieval dev subset

The comparison uses the same 64 official queries and 1,024 public passages, including all 322 judged passages for the selected queries. Each source passage is indexed as one item. Both strategies return five passages. Recall@20 measures candidate retrieval, and nDCG@5 measures final ranking. [Dataset](https://huggingface.co/datasets/C-MTEB/DuRetrieval) · [Judgments](https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels)

| Strategy | Hit@1 | Hit@5 | Recall@20 | nDCG@5 | Query P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original E5 + BM25 | 62/64 | 64/64 | 98.27% | 0.9647 | 6.8 / 9.0 ms |
| BGE-M3 dense + learned sparse + MaxSim | 63/64 | 64/64 | 99.56% | 0.9845 | 9.92 / 16.49 s |

The full BGE-M3 pipeline found a relevant passage first for **one additional query**, increased candidate recall by **1.28 percentage points**, and improved nDCG@5 by **1.98 percentage points** on this fixed subset. Both strategies found a relevant passage in the top five for every query.

Timings are from one Windows CPU (AMD Ryzen 9 8945HX), with both indexes built before timed queries. The BGE-M3 candidate-token cache was cleared for each timed query, so its latency includes encoding the candidates for MaxSim. The subset includes known positives by construction and has a smaller candidate pool than the full corpus.

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
