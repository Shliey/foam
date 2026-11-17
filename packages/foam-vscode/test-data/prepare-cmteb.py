"""Create a deterministic local evaluation fixture from C-MTEB Parquet."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from pyarrow.parquet import read_table

FILES = {
    "corpus": {
        "url": "https://huggingface.co/datasets/C-MTEB/DuRetrieval/resolve/a1a333e290fe30b10f3f56498e3a0d911a693ced/data/corpus-00000-of-00001-19b9e924cb33e4d5.parquet",
        "sha256": "d4b4eb51b63549ef0851a15fc63c2a61b703dce95e3727b535a08f7ba1d14424"
    },
    "queries": {
        "url": "https://huggingface.co/datasets/C-MTEB/DuRetrieval/resolve/a1a333e290fe30b10f3f56498e3a0d911a693ced/data/queries-00000-of-00001-7c7edb40be6b560c.parquet",
        "sha256": "62ac55e764bffd4ffceb0aa51e7a536a0e5932f23c8606566906db6a9efb4b94"
    },
    "qrels": {
        "url": "https://huggingface.co/datasets/C-MTEB/DuRetrieval-qrels/resolve/497b7bd1bbb25cb3757ff34d95a8be50a3de2279/data/dev-00000-of-00001-d3c385852a7c0c9d.parquet",
        "sha256": "c87e7c16f535a98b29ee0ebf6977639c793e3bd149c04634a1810273cfd3c3e5"
    },
    "t2": {
        "url": "https://huggingface.co/datasets/C-MTEB/T2Reranking_zh2en/resolve/d78ad1103cca3f03e075ab472f5ca7858e6aad07/data/dev-00000-of-00001-780e7df94f8dab2b.parquet",
        "sha256": "69a7b684e4fcd41ad85212b1f8b383811f9a01b6b3295463426a811a31510ef4"
    }
}
SEED = "foam-cmteb-v1"


def order(group, key):
    return sha256(f"{SEED}/{group}/{key}".encode()).hexdigest()


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, asset in FILES.items():
        path = root / (name + ".parquet")
        if path.exists() and sha256(path.read_bytes()).hexdigest() == asset["sha256"]:
            continue
        partial = path.with_suffix(".parquet.part")
        try:
            with urlopen(asset["url"], timeout=90) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256(partial.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Unexpected checksum for {path}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


def load(root, name):
    path = root / (name + ".parquet")
    if sha256(path.read_bytes()).hexdigest() != FILES[name]["sha256"]:
        raise ValueError(f"Unexpected checksum for {path}")
    return read_table(path).to_pylist()


def prepare(root):
    corpus = {row["id"]: row["text"] for row in load(root, "corpus")}
    queries = {row["id"]: row["text"] for row in load(root, "queries")}
    labels = defaultdict(dict)
    for row in load(root, "qrels"):
        labels[row["qid"]][row["pid"]] = row["score"]

    eligible = sorted(qid for qid, scores in labels.items()
                      if queries.get(qid) and any(score > 0 for score in scores.values()))
    qids = sorted(eligible, key=lambda qid: order("du-query", qid))[:64]
    required = {pid for qid in qids for pid in labels[qid]}
    if required - corpus.keys():
        raise ValueError("Judged passages missing from corpus")
    pool = set(required)
    unjudged = sorted(set(corpus) - required)
    pool.update(sorted(unjudged, key=lambda pid: order("du-doc", pid))[:1024 - len(pool)])
    du = {
        "corpus": [{"id": pid, "text": corpus[pid]} for pid in sorted(pool)],
        "queries": [{"id": qid, "text": queries[qid], "labels": labels[qid]} for qid in qids],
        "sourceCorpusSize": len(corpus),
        "sourceQueriesSize": len(queries),
        "judgedPassages": len(required),
    }

    items = []
    rows = load(root, "t2")
    t2_ids = sorted(range(len(rows)), key=lambda i: order("t2-query", i))
    for index in t2_ids:
        row = rows[index]
        positive = {text.strip() for text in row["positive"] if text.strip()}
        negative = {text.strip() for text in row["negative"] if text.strip()} - positive
        if not row["query"].strip() or not positive or not negative:
            continue
        candidates = sorted(positive | negative)
        candidates.sort(key=lambda text: order(f"t2-doc-{index}", text))
        items.append({
            "id": index, "text": row["query"],
            "candidates": [{"text": text, "grade": int(text in positive)} for text in candidates],
        })
        if len(items) == 100:
            break
    if len(qids) != 64 or len(pool) != 1024 or len(items) != 100:
        raise ValueError("Dataset did not produce the expected subset size")
    return {"seed": SEED, "files": FILES, "du": du, "t2": items, "t2SourceSize": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("parquet_dir", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.list_files:
        print(json.dumps(FILES, indent=2))
    elif args.parquet_dir and args.output:
        if args.download:
            download(args.parquet_dir)
        args.output.write_text(json.dumps(prepare(args.parquet_dir), ensure_ascii=False), encoding="utf-8")
    else:
        parser.error("provide parquet_dir and output, or use --list-files")
