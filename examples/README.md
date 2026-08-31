# Example inputs and expected output for `predict.py`

These tiny files let you confirm Melody inference works end-to-end before running
on your own data. See the **Inference** section of the top-level `README.md`.

## Files

| File | What it is | Used by |
|------|-----------|---------|
| `example_regions.bed` | **User input** for Mode A: 5 GRCh38 intervals (`chrom start end`). | `--regions` |
| `example_sequences.fa` | **User input** for Mode B: 2 synthetic DNA sequences (one CpG-rich, one CpG-poor). | `--input-fasta` |
| `example_predictions_regions.csv` | **Expected output** for the Mode A command below, using the 39-track Melody-MT checkpoint and three cell types. | reference |

The GRCh38 FASTA and the model checkpoint are **not** in this folder — they are
downloaded from the Google Drive link in the main README (they are part of the
software, not user input).

## Reproduce the expected output (Mode A)

```bash
python predict.py \
  --checkpoint data/checkpoints/Melody-MT-39.pth \
  --regions    examples/example_regions.bed \
  --cell-types Blood-B,Adipocytes,Cortex-Neuron \
  --output     examples/example_predictions_regions.csv
```

This writes one row per (CpG × cell type), e.g.:

```
chrom,cpg_pos,cell_type,methylation
chr1,898022,GSM5652317_Blood-B-Z000000UB,0.3327
chr1,898022,GSM5652176_Adipocytes-Z000000T7,0.3726
chr1,898022,GSM5652223_Cortex-Neuron-Z000000TF,0.5349
...
```

Exact methylation values depend on the checkpoint you download; with the released
Melody-MT-39 checkpoint they reproduce `example_predictions_regions.csv`
(552 prediction rows over the 5 example regions).

## Quick Mode B check

```bash
python predict.py \
  --checkpoint data/checkpoints/Melody-MT-39.pth \
  --input-fasta examples/example_sequences.fa \
  --cell-types Blood-B \
  --output /tmp/seq_preds.csv
```

The CpG-rich sequence yields many CpG predictions; the CpG-poor sequence yields
none (it contains no `CG` dinucleotides) — a quick sanity check that CpG
detection is working.
