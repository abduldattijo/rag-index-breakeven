#!/usr/bin/env python3
"""Run the frozen CUAD pilot or test sample entirely on the Colab GPU.

The script downloads pinned CUAD Parquet files and model revisions. By default
it saves per-question metrics only; raw answers are included only when the
explicit RAG_PILOT_DEBUG=1 flag is set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import urllib.request
from pathlib import Path
import platform

import duckdb
import numpy as np
import torch
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


DATASET = "theatticusproject/cuad-qa"
DATASET_REVISION = "1f61b0333e6d0bc676c913ad9386562a9348e06e"
SPLIT_FILES = {
    "train": [
        "default/train/0000.parquet",
        "default/train/0001.parquet",
        "default/train/0002.parquet",
    ],
    "test": ["default/test/0000.parquet"],
}
ANSWER_MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
ANSWER_REVISION = "e4cc03482199411cf0051723b3303e2c6a93c8b6"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INPUT_PRICE_PER_MILLION = 0.20
OUTPUT_PRICE_PER_MILLION = 0.20
CHUNK_SIZE = 480
CHUNK_OVERLAP = 64
TOP_K = 5
MAX_NEW_TOKENS = 768


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "rag-index-breakeven/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        destination.write_bytes(response.read())


def get_rows(cache: Path, manifest: dict) -> dict[str, dict]:
    contracts = {row["title"]: row for row in manifest["contracts"]}
    titles = set(contracts)
    questions = manifest.get("questions") or [
        question
        for contract in manifest["contracts"]
        for question in contract["questions"]
    ]
    qids = {question["id"] for question in questions}
    paths = []
    split = manifest["split"]
    for relpath in SPLIT_FILES[split]:
        local = cache / split / Path(relpath).name
        download(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{relpath}",
            local,
        )
        paths.append(local)
    path_list = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    con = duckdb.connect()
    sql = f"SELECT title, context, question, id, answers FROM read_parquet([{path_list}])"
    found = {}
    for title, context, question, qid, answers in con.execute(sql).fetchall():
        if title in titles and qid in qids:
            found[qid] = {
                "title": title,
                "context": context,
                "question": question,
                "references": answers["text"] or [],
                "reference_spans": [
                    {"text": text, "start": int(start)}
                    for text, start in zip(answers["text"] or [], answers["answer_start"] or [])
                ],
            }
    con.close()
    if set(found) != qids:
        raise RuntimeError(f"Missing manifest rows: {len(qids - set(found))}")
    for qid, row in found.items():
        contract = contracts[row["title"]]
        import hashlib
        digest = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
        if digest != contract["context_sha256"]:
            raise RuntimeError(f"Context hash mismatch for {row['title']!r}")
        expected = next(
            question for question in contract["questions"]
            if question["id"] == qid
        )
        if bool(row["references"]) != bool(expected["answerable"]):
            raise RuntimeError(f"Answerability mismatch for {qid!r}")
    return found


def chunks_for(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_input_tokens: int = 512,
) -> list[dict]:
    encoded = tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True, verbose=False,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not token_ids:
        return [{"text": text, "start_char": 0, "end_char": len(text)}]
    chunks = []
    start = 0
    while start < len(token_ids):
        token_end = min(start + chunk_size, len(token_ids))
        start_char = offsets[start][0]
        while token_end > start:
            end_char = offsets[token_end - 1][1]
            chunk_text = text[start_char:end_char]
            measured = len(tokenizer(
                chunk_text, add_special_tokens=True, truncation=False
            )["input_ids"])
            if measured <= max_input_tokens:
                break
            token_end -= max(1, measured - max_input_tokens)
        if token_end <= start:
            raise RuntimeError("Could not construct a chunk within the embedder token limit")
        end_char = offsets[token_end - 1][1]
        chunks.append(
            {
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        if token_end >= len(token_ids):
            break
        start = max(start + 1, token_end - overlap)
    return chunks


def normalize(text: str) -> str:
    import re
    import string

    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    gold = normalize(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    common = {}
    for token in pred:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold:
        if common.get(token, 0):
            overlap += 1
            common[token] -= 1
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, references: list[str]) -> tuple[float, float]:
    if not references:
        no_answer = normalize(prediction) in {
            "noanswer", "no answer", "notfound", "not found",
            "norelevantclause", "no relevant clause", "notpresent", "not present"
        }
        return float(no_answer), float(no_answer)
    normalized_prediction = normalize(prediction)
    exact = max(float(normalized_prediction == normalize(ref)) for ref in references)
    f1 = max(token_f1(prediction, ref) for ref in references)
    return exact, f1


def span_recall(answer_spans: list[dict], retrieved_intervals: list[tuple[int, int]]) -> float | None:
    """Maximum character coverage of any one annotated answer span."""
    if not answer_spans:
        return None
    merged = []
    for start, end in sorted(retrieved_intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    recalls = []
    for span in answer_spans:
        start = span["start"]
        end = start + len(span["text"])
        covered = sum(max(0, min(end, right) - max(start, left)) for left, right in merged)
        recalls.append(covered / max(1, end - start))
    return max(recalls)


def answer(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    encoded = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in encoded.items()}
    input_tokens = int(encoded["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.perf_counter() - start
    new_tokens = generated[0, input_tokens:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "prediction": prediction,
        "input_tokens": input_tokens,
        "output_tokens": int(new_tokens.numel()),
        "latency_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(os.environ.get("RAG_MANIFEST", "/content/pilot_manifest.json")),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/rag_index_pilot"))
    parser.add_argument(
        "--output-jsonl", type=Path,
        default=Path(os.environ.get("RAG_OUTPUT_JSONL", "/content/pilot_metrics.jsonl")),
    )
    parser.add_argument(
        "--summary-json", type=Path,
        default=(
            Path(os.environ["RAG_SUMMARY_JSON"])
            if os.environ.get("RAG_SUMMARY_JSON") else None
        ),
    )
    # Colab exec may append the Jupyter kernel's own -f argument.
    args, _unknown = parser.parse_known_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = manifest.get("questions") or [
        question
        for contract in manifest["contracts"]
        for question in contract["questions"]
    ]
    if len({question["id"] for question in questions}) != len(questions):
        raise RuntimeError("Manifest contains duplicate question IDs")
    rows = get_rows(args.cache_dir / "parquet", manifest)
    if os.environ.get("RAG_AUDIT_ONLY") == "1":
        print(json.dumps({
            "split": manifest["split"],
            "contracts": len(manifest["contracts"]),
            "questions": len(questions),
            "answerable": sum(bool(rows[q["id"]]["references"]) for q in questions),
            "no_answer": sum(not rows[q["id"]]["references"] for q in questions),
            "context_hashes_and_answerability": "verified",
        }, sort_keys=True), flush=True)
        return

    print("Loading the pinned embedding model", flush=True)
    embedder = SentenceTransformer(
        EMBED_MODEL,
        revision=EMBED_REVISION,
        device="cuda",
    )
    embed_tokenizer = embedder.tokenizer
    if CHUNK_SIZE > embedder.max_seq_length:
        raise ValueError(
            f"Chunk size {CHUNK_SIZE} exceeds embedding model input limit "
            f"{embedder.max_seq_length}"
        )

    by_title: dict[str, list[tuple[dict, dict]]] = {}
    for question in questions:
        row = rows[question["id"]]
        by_title.setdefault(row["title"], []).append((question, row))

    if os.environ.get("RAG_CHUNK_AUDIT") == "1":
        observed = []
        for items in by_title.values():
            chunks = chunks_for(
                items[0][1]["context"], embed_tokenizer,
                max_input_tokens=embedder.max_seq_length,
            )
            observed.extend(
                len(embed_tokenizer(chunk["text"], add_special_tokens=True)["input_ids"])
                for chunk in chunks
            )
        if max(observed) > embedder.max_seq_length:
            raise RuntimeError("A chunk exceeds the embedding model's input limit")
        print(json.dumps({
            "contracts": len(by_title),
            "chunks": len(observed),
            "max_chunk_input_tokens_including_special_tokens": max(observed),
            "embedder_max_seq_length": embedder.max_seq_length,
            "truncation_risk": False,
        }, sort_keys=True), flush=True)
        return

    print("Loading the pinned answer model", flush=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        ANSWER_MODEL,
        revision=ANSWER_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.generation_config.max_length = None
    tokenizer = MistralCommonBackend.from_pretrained(
        ANSWER_MODEL, revision=ANSWER_REVISION
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    done = set()
    if args.output_jsonl.exists():
        old_rows = [
            json.loads(line)
            for line in args.output_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        methods_by_qid: dict[str, set[str]] = {}
        for old in old_rows:
            methods_by_qid.setdefault(old["qid"], set()).add(old["method"])
        done = {
            qid for qid, methods in methods_by_qid.items()
            if methods == {"full_document", "dense_rag"}
        }
        rows_out = [old for old in old_rows if old["qid"] in done]
    with args.output_jsonl.open("w", encoding="utf-8") as checkpoint:
        for old in rows_out:
            checkpoint.write(json.dumps(old, sort_keys=True) + "\n")
    total_questions = len(questions)
    processed_questions = len(done)
    index_build_seconds = 0.0
    index_bytes = 0
    index_text_bytes = 0
    for title, items in by_title.items():
        context = items[0][1]["context"]
        build_start = time.perf_counter()
        chunks = chunks_for(
            context, embed_tokenizer, max_input_tokens=embedder.max_seq_length
        )
        doc_vectors = embedder.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        build_seconds = time.perf_counter() - build_start
        index_build_seconds += build_seconds
        index_bytes += int(doc_vectors.nbytes)
        index_text_bytes += sum(len(chunk["text"].encode("utf-8")) for chunk in chunks)

        for question_meta, row in items:
            if question_meta["id"] in done:
                continue
            prompt_prefix = (
                "Answer the question using only the contract text. "
                "Return every distinct verbatim passage needed to fully answer it, with no explanation or preamble. "
                "If the contract does not contain an answer, return exactly NO_ANSWER. "
                "Do not invent or paraphrase contract wording.\n\n"
            )
            question_text = f"Question: {row['question']}\n\n"
            full_prompt = (
                prompt_prefix + "Contract:\n" + context + "\n\n" + question_text + "Answer:"
            )

            query = "Represent this sentence for searching relevant passages: " + row["question"]
            retrieval_start = time.perf_counter()
            query_vector = embedder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )[0].astype(np.float32)
            similarities = doc_vectors @ query_vector
            top_indices = np.argsort(-similarities, kind="stable")[:TOP_K]
            retrieved = "\n\n--- Retrieved passage ---\n\n".join(
                chunks[i]["text"] for i in top_indices
            )
            retrieved_intervals = [
                (chunks[i]["start_char"], chunks[i]["end_char"]) for i in top_indices
            ]
            retrieval_seconds = time.perf_counter() - retrieval_start
            rag_prompt = (
                prompt_prefix + "Retrieved contract passages:\n" + retrieved + "\n\n"
                + question_text + "Answer:"
            )

            pair_records = []
            for method, prompt in (("full_document", full_prompt), ("dense_rag", rag_prompt)):
                result = answer(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
                exact, f1 = score(result["prediction"], row["references"])
                record = {
                        "qid": question_meta["id"],
                        "contract_title": title,
                        "method": method,
                        "answerable": bool(row["references"]),
                        "exact_match": exact,
                        "token_f1": f1,
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "generation_seconds": result["latency_seconds"],
                        "retrieval_seconds": retrieval_seconds if method == "dense_rag" else 0.0,
                        "end_to_end_seconds": result["latency_seconds"] + (retrieval_seconds if method == "dense_rag" else 0.0),
                        "gold_span_coverage": (
                            span_recall(row["reference_spans"], retrieved_intervals)
                            if method == "dense_rag"
                            else None
                        ),
                    }
                if os.environ.get("RAG_PILOT_DEBUG") == "1":
                    record["prediction"] = result["prediction"]
                rows_out.append(record)
                pair_records.append(record)
            with args.output_jsonl.open("a", encoding="utf-8") as checkpoint:
                for record in pair_records:
                    checkpoint.write(json.dumps(record, sort_keys=True) + "\n")
                checkpoint.flush()
            processed_questions += 1
            if processed_questions % 10 == 0 or processed_questions == total_questions:
                print(f"{processed_questions}/{total_questions} questions processed", flush=True)

    aggregate = {}
    for method in ("full_document", "dense_rag"):
        method_rows = [row for row in rows_out if row["method"] == method]
        aggregate[method] = {}
        for subset_name, subset in (
            ("all", method_rows),
            ("answerable", [r for r in method_rows if r["answerable"]]),
            ("no_answer", [r for r in method_rows if not r["answerable"]]),
        ):
            aggregate[method][subset_name] = {
                "n": len(subset),
                "exact_match": round(float(np.mean([r["exact_match"] for r in subset])), 6) if subset else None,
                "token_f1": round(float(np.mean([r["token_f1"] for r in subset])), 6) if subset else None,
            }
        input_tokens = sum(row["input_tokens"] for row in method_rows)
        output_tokens = sum(row["output_tokens"] for row in method_rows)
        aggregate[method]["input_tokens"] = input_tokens
        aggregate[method]["output_tokens"] = output_tokens
        aggregate[method]["price_equivalent_usd"] = round(
            input_tokens * INPUT_PRICE_PER_MILLION / 1_000_000
            + output_tokens * OUTPUT_PRICE_PER_MILLION / 1_000_000,
            8,
        )
        aggregate[method]["median_latency_seconds"] = round(
            float(np.median([row["end_to_end_seconds"] for row in method_rows])), 4
        )

    result = {
        "split": manifest["split"],
        "question_count_expected": len(questions),
        "question_count_completed": processed_questions,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "pilot_only": manifest["split"] != "test",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
        "duckdb_version": duckdb.__version__,
        "numpy_version": np.__version__,
        "mistral_common_version": importlib.metadata.version("mistral-common"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "dataset_revision": DATASET_REVISION,
        "answer_model": ANSWER_MODEL,
        "answer_model_revision": ANSWER_REVISION,
        "embedding_model": EMBED_MODEL,
        "embedding_model_revision": EMBED_REVISION,
        "chunk_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "api_price_equivalent_usd_per_million_input_and_output_tokens": 0.20,
        "answer_generation_cost_is_price_equivalent_not_an_api_bill": True,
        "embedding_cost_is_local_compute_not_included_in_price_equivalent": True,
        "index_build_seconds": round(index_build_seconds, 4),
        "index_vector_bytes": index_bytes,
        "index_text_bytes": index_text_bytes,
        "index_payload_bytes": index_bytes + index_text_bytes,
        "dense_rag_gold_span_coverage": round(
            float(np.mean([r["gold_span_coverage"] for r in rows_out if r["method"] == "dense_rag" and r["gold_span_coverage"] is not None])), 6
        ),
        "results": aggregate,
    }
    if result["question_count_completed"] != total_questions:
        raise RuntimeError(
            f"Incomplete output: {result['question_count_completed']} of {total_questions} questions"
        )
    result["per_question_metrics_path"] = str(args.output_jsonl)
    summary_path = args.summary_json or args.output_jsonl.with_name("metrics_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN_SUMMARY_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
