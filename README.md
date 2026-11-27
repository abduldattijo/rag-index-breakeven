# RAG index cost and quality test on CUAD

Full contract prompting against dense retrieval on 30 contracts (1,230 questions) from the
CUAD test split. Both used Ministral 3 14B Instruct, the same instructions and greedy decoding.

## Result

| | Full contract | Dense RAG |
|---|---:|---:|
| Token F1, 385 answerable questions | 28.3 percent | 19.7 percent |
| F1, 845 questions with no answer | 95.5 percent | 96.3 percent |
| Input tokens | 13,157,509 | 3,057,801 |
| Output tokens | 119,582 | 50,005 |
| Token price equivalent, US dollars | 2.66 | 0.62 |

Dense RAG minus full prompting on answerable questions is -8.5 points (95 percent interval
-11.0 to -6.0). The quality limit was -5 points. The price is token counts times a published
rate (0.20 US dollars per million input or output tokens, checked 27 November 2026). It is not
a bill and leaves out GPU time, index building and storage.

## Files

| File | What it does |
|---|---|
| `build_manifests.py` | Picks the contracts and questions |
| `test_manifest.json`, `pilot_manifest.json` | The picked IDs and text hashes (no contract text) |
| `run_local_pilot.py` | Retrieval, answer generation and scoring (used for the pilot and the test) |
| `analyze_results.py` | Rebuilds `results/analysis.json` from the saved rows |
| `plot_results.py` | Draws the figure |
| `results/test_metrics.jsonl` | One row per question and method: scores, token counts, times |
| `results/earlier_run/` | An earlier full run, set aside. Same scores and token counts, different times |
| `results/manual_audit.md` | An AI assistant's review of 20 question pairs. The reply text was not saved |

## Recheck the numbers (no GPU)

```bash
pip install numpy matplotlib
python analyze_results.py
python plot_results.py
```

## Run the test again

This used an NVIDIA A100 with 80 GB of memory on Google Colab. Run from this folder.

```bash
pip install -r requirements.txt
python run_local_pilot.py --manifest test_manifest.json --cache-dir cache \
  --output-jsonl rerun/test_metrics.jsonl --summary-json rerun/run_summary.json
```

The script downloads the pinned dataset and model revisions. If the output file already exists,
questions finished by both methods are skipped.

To rebuild the manifests: `python build_manifests.py --cache-dir cache`

## Data

[CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa) by The Atticus Project,
licensed CC BY 4.0, revision `1f61b0333e6d0bc676c913ad9386562a9348e06e`. The dataset card makes
no promise about the license status of the underlying contracts. No contract text is stored here.
The manifests use contract titles and question IDs from the dataset.

Hendrycks, Burns, Chen and Ball, [CUAD: An Expert-Annotated NLP Dataset for Legal Contract
Review](https://arxiv.org/abs/2103.06268).

## License

The code is under the MIT License (see `LICENSE`). The CUAD data keeps its own CC BY 4.0 license.
