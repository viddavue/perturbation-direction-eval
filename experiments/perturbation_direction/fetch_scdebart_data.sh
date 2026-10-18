#!/usr/bin/env bash
# Fetch the scDEBART files needed for the fine-tune. Works without the `hf`/`huggingface-cli` binary.
#
# Usage:
#   bash fetch_scdebart_data.sh            # -> ./scDEBART_data
#   DEST=/path/to/scDEBART_data bash fetch_scdebart_data.sh
#
# Total ~11 GB (the Nadig HepG2 DE file is ~10 GB). Resumable: re-run after a failure and
# finished files are skipped.
set -euo pipefail

REPO="Jieun-S/scDEBART"          # HF *dataset* repo (note: the GitHub org is Jieun-Sung)
DEST="${DEST:-$PWD/scDEBART_data}"

FILES=(
  "cellxgene/pretrain_output/best_pretrained_scDEBART_model_weights.pt"
  "data/BioBERT_gene_embedding_dict.pkl"
  "data/c2_all_2024_1_membership_ens2pathway.tsv"
  "data/ens2geneid.pkl"
  "perturb_seq/Nadig_HepG2/DE_results/de_masked_high_zero_logfc_to_zero.h5"
)

mkdir -p "$DEST"

# --- pick a downloader -------------------------------------------------------
# 1) huggingface_hub's CLI (new name `hf`, older name `huggingface-cli`)
CLI=""
for cand in hf huggingface-cli; do
  if command -v "$cand" >/dev/null 2>&1; then CLI="$cand"; break; fi
done

# 2) the huggingface_hub Python package without the CLI, or a plain python
PY=""
for cand in python python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done

have_hub() { [ -n "$PY" ] && "$PY" -c "import huggingface_hub" >/dev/null 2>&1; }

# 3) last resort: direct HTTP with whatever is installed
http_get() {  # http_get <repo-relative-path>
  local rel="$1" url="https://huggingface.co/datasets/$REPO/resolve/main/$1" out="$DEST/$1"
  mkdir -p "$(dirname "$out")"
  echo ">> curl $rel"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c --tries=5 -O "$out" "$url"
  elif [ -n "$PY" ]; then
    "$PY" - "$url" "$out" <<'PYEOF'
import sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as r, open(out, "wb") as f:
    while chunk := r.read(1 << 20):
        f.write(chunk)
PYEOF
  else
    echo "ERROR: need curl, wget, or python to download" >&2
    exit 1
  fi
}

echo "Destination: $DEST"
if [ -n "$CLI" ]; then
  echo "Downloader : $CLI ($(command -v "$CLI"))"
elif have_hub; then
  echo "Downloader : python -m huggingface_hub ($PY, no CLI)"
else
  echo "Downloader : plain HTTP (curl/wget/python); huggingface_hub not installed"
fi

# --- prefer the official downloader, fall back per file ----------------------
for rel in "${FILES[@]}"; do
  if [ -f "$DEST/$rel" ]; then
    echo "== skip (exists): $rel"
    continue
  fi
  if [ -n "$CLI" ]; then
    "$CLI" download "$REPO" "$rel" --repo-type dataset --local-dir "$DEST" || http_get "$rel"
  elif have_hub; then
    "$PY" -m huggingface_hub download "$REPO" "$rel" --repo-type dataset --local-dir "$DEST" || http_get "$rel"
  else
    http_get "$rel"
  fi
done

echo
echo "Done. Files:"
du -sh "$DEST" 2>/dev/null || true
find "$DEST" -type f -not -name '*.incomplete' -printf '  %10s  %p\n' 2>/dev/null | sort -k2 || find "$DEST" -type f

cat <<EOF

Next steps (in the scDEBART git clone, github.com/Jieun-Sung/scDEBART):
  # local-dir keeps the repo prefix, so the files are under \$DEST/data and \$DEST/cellxgene
  ln -sfn "$DEST/data" ./data                        # their script hardcodes datadir = "./data"
  mkdir -p perturb_seq/Nadig_HepG2/DE_results cellxgene/pretrain_output
  cp "$DEST"/perturb_seq/Nadig_HepG2/DE_results/*.h5 perturb_seq/Nadig_HepG2/DE_results/
  cp "$DEST"/cellxgene/pretrain_output/*.pt          cellxgene/pretrain_output/
  cp hvg_geneids_1784.pkl perturb_seq/Nadig_HepG2/DE_results/

NOTE: hvg_geneids_5000.pkl is not published for Nadig_HepG2, so their finetune script cannot
run without the gene list we generate (hvg_geneids_1784.pkl, passed as --num_HVG 1784).
EOF
