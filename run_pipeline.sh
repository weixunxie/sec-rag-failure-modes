#!/usr/bin/env bash
# Re-run the full SEC RAG evaluation from scratch (requires OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY in .env
# and network access to data.sec.gov, api.openai.com and your Qdrant cluster).
#
#   ./run_pipeline.sh            # everything: download -> chunk/embed -> references -> retrieval -> eval -> figures
#   ./run_pipeline.sh --after-review   # skip steps 1-3 once reference_answers.csv has been reviewed
#
# Estimated cost ~$0.25 (embeddings + gpt-4o-mini generation/judging + 40 gpt-4o reference answers), 30-45 min.
set -euo pipefail
cd "$(dirname "$0")"
run_nb () { jupyter nbconvert --to notebook --execute --inplace "notebooks/$1" --ExecutePreprocessor.timeout=7200; }

if [[ "${1:-}" != "--after-review" ]]; then
  # Archive first-run outputs so the notebooks do not silently reload them
  ARCHIVE="data/processed/v1_archive"
  mkdir -p "$ARCHIVE"
  for f in filing_metadata.csv filing_chunks.parquet retrieval_results.parquet generation_results.csv evaluation_results.csv summary_results.csv; do
    [[ -f "data/processed/$f" ]] && git mv -f "data/processed/$f" "$ARCHIVE/$f" 2>/dev/null || mv -f "data/processed/$f" "$ARCHIVE/$f" 2>/dev/null || true
  done
  rm -rf data/raw                                   # old filings were saved under off-by-one fiscal-year names

  echo "== 1/5 Download 10-K filings (fiscal year matched on reportDate)"; run_nb 01_collect_filings.ipynb
  echo "== 2/5 Chunk + embed (recreates the Qdrant collection)";        run_nb 02_chunk_and_embed.ipynb
  echo "== 3/5 Build reference answers";                                  python -m src.reference_answers
  echo
  echo ">>> Review data/processed/reference_answers.csv now (numbers especially), then run:"
  echo ">>>     ./run_pipeline.sh --after-review"
  exit 0
fi

python -c "from src.reference_answers import attach_reference_answers; attach_reference_answers()"
echo "== 4/5 Retrieval experiments";            run_nb 03_run_retrieval_experiments.ipynb
echo "== 5/5 Generation + evaluation";          run_nb 04_evaluate_answers.ipynb
echo "== Figures + summary";                    run_nb 05_error_analysis_and_figures.ipynb
python reports/build_dashboard.py
echo "Done. Update the Results table in README.md from data/processed/summary_results.csv."
