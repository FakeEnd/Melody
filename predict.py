#!/usr/bin/env python
"""
predict.py -- Run Melody inference: predict single-CpG DNA methylation from sequence.

This is the entry point for *using* a trained Melody model on your own data
(as opposed to ``run.py``, which *trains* a model).

================================================================================
WHAT YOU (THE USER) PROVIDE  vs  WHAT THE SOFTWARE ALREADY SHIPS
================================================================================

USER-PROVIDED INPUT (pick exactly ONE of the two modes):

  Mode A  --regions  REGIONS.bed
          A plain-text BED/TSV/CSV file of genomic intervals on the GRCh38
          assembly that you want methylation predicted for. One interval per
          line, with at least three columns:
                chrom    start    end
          e.g.  chr1     900000   901000
          (Header line optional; extra columns are ignored. Coordinates are
           0-based, half-open, like a standard BED file.)
          The DNA sequence is looked up *for you* from the bundled GRCh38 FASTA,
          so in this mode you do NOT supply any sequence yourself.

  Mode B  --input-fasta  SEQS.fa
          A FASTA file of raw DNA sequences you supply yourself (they do NOT
          need to come from the human genome). Each record is one sequence of
          A/C/G/T/N. Sequences are center-cropped or center-padded to the model
          window (default 10,000 bp). Use this mode for designed / synthetic /
          variant sequences.

SOFTWARE-PROVIDED FILES (downloaded once from our Google Drive into ./data/,
NOT something you create -- see README "Data Preparation"):

  --genome       GRCh38 primary-assembly FASTA. Required for Mode A only.
                 Default: ./data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa
  --checkpoint   A trained Melody checkpoint (.pth). This defines which cell
                 types the model can predict and how many output tracks it has.
                 A 39-track multi-tissue (Melody-MT) checkpoint predicts all 39
                 cell types at once; a single-track (Melody-ST) checkpoint
                 predicts one cell type.

OUTPUT:

  A CSV with one row per (CpG site x requested cell type):
      Mode A columns:  chrom, cpg_pos, cell_type, methylation
      Mode B columns:  seq_id, cpg_pos, cell_type, methylation
  ``cpg_pos`` is the 0-based position of the cytosine of each forward-strand
  CpG (a "CG" dinucleotide); ``methylation`` is the predicted level in [0, 1].

EXAMPLE
-------
    # Mode A: predict for a few GRCh38 regions, all 39 cell types
    python predict.py \
        --checkpoint data/checkpoints/Melody-MT-39.pth \
        --regions examples/example_regions.bed \
        --output  example_predictions.csv

    # Mode A: only two cell types (substring match on the track name)
    python predict.py --checkpoint data/checkpoints/Melody-MT-39.pth \
        --regions examples/example_regions.bed \
        --cell-types Blood-B,Adipocytes --output two_types.csv

    # Mode B: your own sequences, single-track model
    python predict.py --checkpoint path/to/Melody-ST.pth --n-track 1 \
        --input-fasta examples/example_sequences.fa --output seq_predictions.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.append(str(Path(__file__).parent))

from global_constants import track_39_names
from models import Melody
from stateless import load_ckpt, sigmoid_first

# One-hot channel order used throughout Melody / selene_sdk: A, C, G, T.
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def encode_sequence(seq: str) -> np.ndarray:
    """Encode an A/C/G/T/N string into a (L, 4) one-hot array (A,C,G,T order).

    Unknown bases (N or anything else) become a flat 0.25 vector, matching the
    convention selene_sdk uses for unspecified positions.
    """
    seq = seq.upper()
    arr = np.full((len(seq), 4), 0.25, dtype=np.float32)
    for i, base in enumerate(seq):
        idx = BASE_TO_IDX.get(base)
        if idx is not None:
            arr[i] = 0.0
            arr[i, idx] = 1.0
    return arr


def fit_to_window(onehot: np.ndarray, window: int):
    """Center-crop or center-pad a (L, 4) one-hot array to exactly ``window`` rows.

    Returns ``(fitted, offset)`` where a position ``w`` in ``fitted`` corresponds
    to position ``w + offset`` in the original ``onehot`` (so callers can map
    predictions back to the user's input coordinates).
    """
    length = onehot.shape[0]
    if length == window:
        return onehot, 0
    if length > window:
        start = (length - window) // 2
        return onehot[start:start + window], start
    # pad symmetrically with 0.25 (unknown)
    pad_left = (window - length) // 2
    out = np.full((window, 4), 0.25, dtype=np.float32)
    out[pad_left:pad_left + length] = onehot
    return out, -pad_left


def cpg_c_positions(onehot: np.ndarray) -> np.ndarray:
    """Return indices i where position i is 'C' and i+1 is 'G' (forward-strand CpG).

    The returned index is the cytosine position, the standard CpG coordinate.
    """
    c = onehot[:, 1] > 0.5
    g = onehot[:, 2] > 0.5
    is_cpg_c = np.zeros(onehot.shape[0], dtype=bool)
    is_cpg_c[:-1] = c[:-1] & g[1:]
    return np.nonzero(is_cpg_c)[0]


def read_fasta(path: str):
    """Minimal FASTA reader -> list of (id, sequence). Avoids a Biopython dep."""
    records = []
    seq_id, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(chunks)))
                seq_id = line[1:].split()[0] or f"seq{len(records)}"
                chunks = []
            else:
                chunks.append(line.strip())
    if seq_id is not None:
        records.append((seq_id, "".join(chunks)))
    return records


def read_regions(path: str):
    """Read a BED/TSV/CSV of (chrom, start, end). Skips an optional header line."""
    regions = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], parts[1], parts[2]
            try:
                start, end = int(start), int(end)
            except ValueError:
                # header row such as "chrom start end"
                continue
            if end <= start:
                logger.warning(f"Skipping region with end <= start: {line}")
                continue
            regions.append((chrom, start, end))
    return regions


def tile_window_starts(start: int, end: int, window: int):
    """Window start positions tiling [start, end).

    For regions shorter than ``window`` the single window is centered on the
    region; longer regions are tiled non-overlapping with the final window
    anchored at the region end so the whole interval is covered.
    """
    span = end - start
    if span <= window:
        center = (start + end) // 2
        return [max(0, center - window // 2)]
    starts = list(range(start, end, window))
    last = end - window
    if starts[-1] != last:
        starts.append(last)
    return starts


# --------------------------------------------------------------------------- #
# Cell-type / track selection
# --------------------------------------------------------------------------- #
def resolve_cell_types(requested, track_names):
    """Map user-requested cell types (exact or substring) to (index, name) pairs.

    With no request, returns all tracks in order.
    """
    if not requested:
        return list(enumerate(track_names))
    selected = []
    for query in requested:
        query = query.strip()
        if query in track_names:
            selected.append((track_names.index(query), query))
            continue
        hits = [(i, n) for i, n in enumerate(track_names) if query in n]
        if len(hits) == 1:
            selected.append(hits[0])
        elif len(hits) == 0:
            raise SystemExit(f"[predict] cell type '{query}' not found in checkpoint track names.")
        else:
            names = ", ".join(n for _, n in hits)
            raise SystemExit(f"[predict] cell type '{query}' is ambiguous, matches: {names}")
    return selected


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_model(checkpoint: str, n_track: int, cpg_cls_n: int, device: str) -> Melody:
    args = argparse.Namespace(cpg_cls_n=cpg_cls_n)
    model = Melody(args, n_track=n_track)
    load_ckpt(model, checkpoint, mute=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_batch(model: Melody, onehots, device: str) -> np.ndarray:
    """onehots: list/array of (window, 4) -> methylation array (B, n_track, window)."""
    x = torch.from_numpy(np.stack(onehots)).to(device=device, dtype=torch.float32)
    x = x.transpose(1, 2).contiguous()  # (B, 4, window)
    out_final = model(x)[0]             # (B, n_track, window) raw logits
    return sigmoid_first(out_final).cpu().numpy()


def run_regions(model, genome, regions, window, cell_types, batch_size, device, writer):
    """Mode A: predict for GRCh38 intervals; emit per-CpG rows inside each interval."""
    # Build the full work list of (region_idx, chrom, win_start) windows first.
    jobs = []
    for chrom, start, end in regions:
        for w_start in tile_window_starts(start, end, window):
            jobs.append((chrom, start, end, w_start))

    n_written = 0
    for b in range(0, len(jobs), batch_size):
        batch = jobs[b:b + batch_size]
        onehots, metas = [], []
        for chrom, r_start, r_end, w_start in batch:
            try:
                enc = np.asarray(genome.get_encoding_from_coords(chrom, w_start, w_start + window),
                                 dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - report and skip bad coords
                logger.warning(f"Could not fetch {chrom}:{w_start}-{w_start + window}: {exc}")
                continue
            if enc.shape[0] != window:
                logger.warning(f"{chrom}:{w_start}-{w_start + window} returned {enc.shape[0]} bp "
                               f"(near a chromosome boundary?); skipping this window.")
                continue
            onehots.append(enc)
            metas.append((chrom, r_start, r_end, w_start, enc))
        if not onehots:
            continue
        preds = predict_batch(model, onehots, device)  # (B, n_track, window)
        for k, (chrom, r_start, r_end, w_start, enc) in enumerate(metas):
            for local_c in cpg_c_positions(enc):
                genomic_pos = w_start + int(local_c)
                if not (r_start <= genomic_pos < r_end):
                    continue  # only report CpGs inside the requested interval
                for track_idx, name in cell_types:
                    writer.writerow([chrom, genomic_pos, name,
                                     f"{preds[k, track_idx, local_c]:.4f}"])
                    n_written += 1
    return n_written


def run_fasta(model, records, window, cell_types, batch_size, device, writer):
    """Mode B: predict for user sequences; emit per-CpG rows along each sequence."""
    n_written = 0
    for b in range(0, len(records), batch_size):
        batch = records[b:b + batch_size]
        onehots, metas = [], []
        for seq_id, seq in batch:
            enc_full = encode_sequence(seq)
            if enc_full.shape[0] > window:
                logger.warning(f"Sequence '{seq_id}' is {enc_full.shape[0]} bp > window {window}; "
                               f"only the central {window} bp are scored.")
            enc, offset = fit_to_window(enc_full, window)
            onehots.append(enc)
            metas.append((seq_id, enc, offset))
        preds = predict_batch(model, onehots, device)
        for k, (seq_id, enc, offset) in enumerate(metas):
            for local_c in cpg_c_positions(enc):
                orig_pos = int(local_c) + offset  # position in the user's original sequence
                for track_idx, name in cell_types:
                    writer.writerow([seq_id, orig_pos, name,
                                     f"{preds[k, track_idx, local_c]:.4f}"])
                    n_written += 1
    return n_written


def main():
    p = argparse.ArgumentParser(
        description="Melody inference: predict single-CpG DNA methylation from sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--regions", help="[USER INPUT, Mode A] BED/TSV/CSV of GRCh38 intervals (chrom start end).")
    src.add_argument("--input-fasta", dest="input_fasta",
                     help="[USER INPUT, Mode B] FASTA of your own DNA sequences.")

    p.add_argument("--checkpoint", required=True,
                   help="[SOFTWARE FILE] Trained Melody checkpoint (.pth).")
    p.add_argument("--genome",
                   default="./data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                   help="[SOFTWARE FILE] GRCh38 FASTA (Mode A only).")
    p.add_argument("--output", required=True, help="Output CSV path.")

    p.add_argument("--n-track", dest="n_track", type=int, default=39,
                   help="Number of output tracks the checkpoint was trained with (39 for Melody-MT, 1 for Melody-ST). Default: 39.")
    p.add_argument("--track-names", dest="track_names", default=None,
                   help="Optional comma-separated track names overriding the built-in 39-track names (order must match the checkpoint).")
    p.add_argument("--cell-types", dest="cell_types", default=None,
                   help="Comma-separated subset of cell types to output (exact or substring match). Default: all tracks.")
    p.add_argument("--window-size", dest="window_size", type=int, default=10000,
                   help="Model input length. Default: 10000.")
    p.add_argument("--cpg-cls-n", dest="cpg_cls_n", type=int, default=7,
                   help="CpG-count classification head size (must match training). Default: 7.")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda or cpu. Default: cuda if available.")
    args = p.parse_args()

    # Resolve track names for this checkpoint.
    if args.track_names:
        track_names = [t.strip() for t in args.track_names.split(",") if t.strip()]
    elif args.n_track == 39:
        track_names = list(track_39_names)
    else:
        track_names = [f"track_{i}" for i in range(args.n_track)]
    if len(track_names) != args.n_track:
        raise SystemExit(f"[predict] --n-track={args.n_track} but {len(track_names)} track names were resolved.")

    requested = args.cell_types.split(",") if args.cell_types else None
    cell_types = resolve_cell_types(requested, track_names)

    logger.info(f"Device: {args.device} | checkpoint: {args.checkpoint} | n_track: {args.n_track}")
    logger.info(f"Reporting {len(cell_types)} cell type(s): {', '.join(n for _, n in cell_types)}")
    model = build_model(args.checkpoint, args.n_track, args.cpg_cls_n, args.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        if args.regions:
            from selene_sdk.sequences import Genome
            logger.info(f"Loading GRCh38 genome from {args.genome}")
            genome = Genome(input_path=args.genome, blacklist_regions="hg38")
            regions = read_regions(args.regions)
            logger.info(f"Mode A: {len(regions)} region(s)")
            writer.writerow(["chrom", "cpg_pos", "cell_type", "methylation"])
            n = run_regions(model, genome, regions, args.window_size, cell_types,
                            args.batch_size, args.device, writer)
        else:
            records = read_fasta(args.input_fasta)
            logger.info(f"Mode B: {len(records)} sequence(s)")
            writer.writerow(["seq_id", "cpg_pos", "cell_type", "methylation"])
            n = run_fasta(model, records, args.window_size, cell_types,
                          args.batch_size, args.device, writer)
    logger.info(f"Wrote {n} prediction rows to {out_path}")


if __name__ == "__main__":
    main()
