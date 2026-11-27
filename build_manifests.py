#!/usr/bin/env python3
"""Build frozen CUAD manifests without copying contract text into the repo.

Dependencies: duckdb and tiktoken. Parquet files are downloaded to a local cache
directory and can be removed after the manifests have been built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

import duckdb
import tiktoken


REPO = "theatticusproject/cuad-qa"
REVISION = "1f61b0333e6d0bc676c913ad9386562a9348e06e"
FILES = {
    "test": ["default/test/0000.parquet"],
    "train": [
        "default/train/0000.parquet",
        "default/train/0001.parquet",
        "default/train/0002.parquet",
    ],
}
TEST_N = 30
SEED = 261126
PERIPLUS_N = 10
QUESTION_TYPE = re.compile(r'related to\s+"([^"]+)"', re.IGNORECASE)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_files(cache: Path) -> dict[str, list[Path]]:
    cache.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for split, files in FILES.items():
        result[split] = []
        for relpath in files:
            destination = cache / split / Path(relpath).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                url = (
                    f"https://huggingface.co/datasets/{REPO}/resolve/"
                    f"{REVISION}/{relpath}"
                )
                request = urllib.request.Request(
                    url, headers={"User-Agent": "rag-index-breakeven-research/1.0"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    destination.write_bytes(response.read())
            result[split].append(destination)
    return result


def read_split(paths: list[Path], encoder) -> dict[str, dict]:
    con = duckdb.connect()
    path_sql = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    cursor = con.execute(
        f"SELECT title, context, question, id, answers "
        f"FROM read_parquet([{path_sql}]) ORDER BY title, id"
    )
    documents: dict[str, dict] = {}
    while batch := cursor.fetchmany(128):
        for title, context, question, qid, answers in batch:
            if title not in documents:
                documents[title] = {
                    "title": title,
                    "context": context,
                    "context_chars": len(context),
                    "context_tokens_gpt4o": len(encoder.encode(context)),
                    "context_sha256": sha256(context),
                    "questions": [],
                }
            document = documents[title]
            if document["context_sha256"] != sha256(context):
                raise ValueError(f"Inconsistent context for contract {title!r}")
            match = QUESTION_TYPE.search(question)
            if not match:
                raise ValueError(f"Cannot parse question category for {qid!r}")
            texts = answers["text"] or []
            starts = answers["answer_start"] or []
            if len(texts) != len(starts):
                raise ValueError(f"Answer text/offset mismatch for {qid!r}")
            for text, start in zip(texts, starts):
                if context[start : start + len(text)] != text:
                    raise ValueError(f"Gold answer offset mismatch for {qid!r}")
            document["questions"].append(
                {
                    "id": qid,
                    "category": match.group(1),
                    "answerable": bool(texts),
                    "answer_count": len(texts),
                }
            )
    con.close()
    return documents


def pick_stratified(items: list[dict], groups: int, per_group: int, seed: int) -> list[dict]:
    ordered = sorted(items, key=lambda x: (x["context_tokens_gpt4o"], x["title"]))
    if len(ordered) < groups * per_group:
        raise ValueError("Not enough documents for the requested stratified sample")
    rng = random.Random(seed)
    chosen = []
    for group in range(groups):
        start = round(group * len(ordered) / groups)
        end = round((group + 1) * len(ordered) / groups)
        stratum = ordered[start:end]
        chosen.extend(rng.sample(stratum, per_group))
    return sorted(chosen, key=lambda x: (x["context_tokens_gpt4o"], x["title"]))


def clean_document(document: dict) -> dict:
    return {
        key: value
        for key, value in document.items()
        if key != "context"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    paths = fetch_files(args.cache_dir)
    encoder = tiktoken.encoding_for_model("gpt-4o")
    train = read_split(paths["train"], encoder)
    test = read_split(paths["test"], encoder)
    if len(train) != 408 or len(test) != 102:
        raise ValueError(f"Unexpected contract counts: train={len(train)}, test={len(test)}")

    all_documents = {**train, **test}
    paper_documents = sorted(
        all_documents.values(),
        key=lambda x: (-x["context_tokens_gpt4o"], x["title"]),
    )[:PERIPLUS_N]
    paper_titles = {document["title"] for document in paper_documents}
    eligible_test = [doc for title, doc in test.items() if title not in paper_titles]
    confirmatory = pick_stratified(eligible_test, groups=3, per_group=10, seed=SEED)

    # Four independent train contracts, one from each token-length quartile.
    pilot_pool = [doc for title, doc in train.items() if title not in paper_titles]
    pilot_docs = pick_stratified(pilot_pool, groups=4, per_group=1, seed=SEED + 1)
    pilot_questions = []
    pilot_rng = random.Random(SEED + 2)
    for document in pilot_docs:
        answerable = [q for q in document["questions"] if q["answerable"]]
        no_answer = [q for q in document["questions"] if not q["answerable"]]
        if len(answerable) < 2 or len(no_answer) < 3:
            raise ValueError(f"Not enough answerable/no-answer questions for pilot: {document['title']}")
        selected = pilot_rng.sample(answerable, 2) + pilot_rng.sample(no_answer, 3)
        pilot_questions.extend(
            {"contract_title": document["title"], **question}
            for question in sorted(selected, key=lambda q: q["id"])
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": REPO,
        "revision": REVISION,
        "tokenizer": "tiktoken encoding_for_model(gpt-4o)",
        "seed": SEED,
        "selection_rule": "Exclude the exact ten longest CUAD contracts used by Periplus; divide remaining test contracts into three token-length strata and sample ten per stratum with Python random.Random(seed).",
        "excluded_periplus_contracts": [clean_document(d) for d in paper_documents],
        "excluded_periplus_contracts_in_test_split": sum(
            document["title"] in test for document in paper_documents
        ),
    }
    (output_dir / "test_manifest.json").write_text(
        json.dumps(
            {
                **common,
                "split": "test",
                "contract_count": len(confirmatory),
                "question_count": sum(len(d["questions"]) for d in confirmatory),
                "contracts": [clean_document(d) for d in confirmatory],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "pilot_manifest.json").write_text(
        json.dumps(
            {
                **common,
                "split": "train",
                "contract_count": len(pilot_docs),
                "question_count": len(pilot_questions),
                "selection_rule": "Exclude the exact ten Periplus contracts, divide remaining train contracts into four token-length strata, and sample one per stratum with seed+1.",
                "question_rule": "Select two answerable and three no-answer questions per contract using seed+2.",
                "contracts": [clean_document(d) for d in pilot_docs],
                "questions": pilot_questions,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote test manifest: {len(confirmatory)} contracts, "
        f"{sum(len(d['questions']) for d in confirmatory)} questions"
    )
    print(f"Wrote pilot manifest: {len(pilot_docs)} contracts, {len(pilot_questions)} questions")
    print(
        "Periplus top-ten contracts excluded from test sample: "
        f"{sum(document['title'] in test for document in paper_documents)}"
    )


if __name__ == "__main__":
    main()
