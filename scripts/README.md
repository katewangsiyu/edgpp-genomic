# Scripts

| # | Script | Phase | Purpose |
|---|---|---|---|
| env | `env_t4.sh` / `env_5090.sh` | setup | conda env creation |
| 01 | `download_data.sh` | setup | hg38 + Flashzoi weights + Borzoi QTL VCFs |
| 02 | `02_prepare_debug_snp.py` | setup | build 100-SNP debug parquet |
| 03 | `03_smoke.py` | Phase 0 / 1 | smoke test (config-driven, same script for T4 and 5090) |
| 05 | `05_teacher_precompute.py` | Phase 2a | *(TODO)* precompute Flashzoi SED to parquet cache |
| 06 | `06_train.py` | Phase 2b | *(TODO)* train 3-way ablation {baseline, degu, edgpp} |
| 07 | `07_evaluate.py` | Phase 2c | *(TODO)* QTL metrics + calibration + selective-gain |

Phase 2 scripts (05/06/07) are created once Phase 0/1 are green — they share
the same `config.yaml` and reuse the modules in `src/edgpp_genomic/`.
