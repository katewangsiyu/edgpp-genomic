#!/usr/bin/env bash
# Download script. Run from project root.
#   1. hg38 reference
#   2. Flashzoi weights (HuggingFace)
#   3. Borzoi QTL VCFs (gs://borzoi-paper/, Requester Pays — needs gcloud + GCP project)
set -euo pipefail

mkdir -p data/raw data/raw/qtl_vcf data/raw/flashzoi

# ---------- 1. hg38 ----------
if [[ ! -f data/raw/hg38.fa ]]; then
    echo "[1/3] Downloading hg38..."
    wget -O data/raw/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    gunzip data/raw/hg38.fa.gz
    echo "[1/3] Building .fai index (pyfaidx will do this on first use, safe to skip)..."
fi

# ---------- 2. Flashzoi weights (HuggingFace) ----------
if [[ ! -d data/raw/flashzoi/replicate_0 ]]; then
    echo "[2/3] Downloading Flashzoi replicates 0-3 from HuggingFace..."
    python - <<'PY'
from huggingface_hub import snapshot_download
for r in range(4):
    snapshot_download(
        repo_id=f"johahi/borzoi-replicate-{r}",
        local_dir=f"data/raw/flashzoi/replicate_{r}",
        local_dir_use_symlinks=False,
    )
print("Flashzoi replicates downloaded.")
PY
fi

# ---------- 3. Borzoi QTL VCFs ----------
# Requires gcloud CLI + billing-enabled GCP project.
if ! command -v gsutil >/dev/null 2>&1; then
    echo "[3/3] SKIP: gsutil not installed."
    echo "  Install:   curl https://sdk.cloud.google.com | bash && exec -l \$SHELL && gcloud init"
    echo "  Then re-run:  GCP_PROJECT=<your-project-id> bash scripts/download_data.sh"
    exit 0
fi
if [[ -z "${GCP_PROJECT:-}" ]]; then
    echo "[3/3] SKIP: GCP_PROJECT env var not set."
    echo "  Re-run with:  GCP_PROJECT=<your-project-id> bash scripts/download_data.sh"
    exit 0
fi

echo "[3/3] Downloading Borzoi QTL VCFs (Whole_Blood, eQTL)..."
gsutil -u "$GCP_PROJECT" cp \
    gs://borzoi-paper/qtl/eqtl/gtex_fine/tissues_susie/Whole_Blood_pos.vcf \
    gs://borzoi-paper/qtl/eqtl/gtex_fine/tissues_susie/Whole_Blood_neg.vcf \
    data/raw/qtl_vcf/

echo ""
echo "All downloads finished."
