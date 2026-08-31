import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from stateless import mask2value
class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.GroupNorm(1, oup),
        )

    def forward(self, x):
        return x + self.conv(x)

# noinspection PyShadowingNames,SpellCheckingInspection,GrazieInspection,PyTypeChecker
def calc_cg_and_avg_loss_variable_len(
        seq,  # (bsz, seq_len, 4)
        methy,  # (bsz, n_track, seq_len)
        pred_100_cg_count,  # (bsz, 7, num_segs)
        pred_100_methy_avg,  # (bsz, n_track, num_segs)
        cpg_cls_n,
        cpg_cls_bin_width,
):
    batch_size, n_track, original_seq_length = methy.shape
    seg_length = 100
    device = seq.device


    num_original_segs = (original_seq_length + seg_length - 1) // seg_length  # Ceil division
    padded_length = num_original_segs * seg_length
    padding_needed = padded_length - original_seq_length

    if padding_needed > 0:
        # Pad seq with [0,0,0,0] (N base)
        seq_padded = F.pad(seq, (0, 0, 0, padding_needed), 'constant', 0)
        # Pad methy with a negative value to easily ignore it later
        methy_padded = F.pad(methy, (0, padding_needed), 'constant', -1.0)
    else:
        seq_padded = seq
        methy_padded = methy

    num_segs = padded_length // seg_length

    is_c = seq_padded[:, :-1, 1] == 1
    is_g_next = seq_padded[:, 1:, 2] == 1
    is_cg_start = is_c & is_g_next

    cg_sites = torch.zeros((batch_size, padded_length), dtype=torch.bool, device=device)
    cg_sites[:, :-1][is_cg_start] = True
    cg_sites[:, 1:][is_cg_start] = True

    # --- Task 1: CG Count Loss ---
    cg_sites_reshaped_for_count = cg_sites.view(batch_size, num_segs, seg_length)
    cg_counts = cg_sites_reshaped_for_count.sum(dim=2)
    target_cg_class = parse_cg_class_n_cuda(cg_counts, cpg_cls_bin_width=cpg_cls_bin_width, cpg_cls_n=cpg_cls_n)
    target_cg_class = target_cg_class.to(dtype=torch.long)

    assert pred_100_cg_count.shape[2] == num_segs, \
        f"Prediction segment count ({pred_100_cg_count.shape[2]}) does not match target segment count ({num_segs})"

    loss_cg = F.cross_entropy(pred_100_cg_count, target_cg_class)

    # --- Task 2: Average Methylation Loss ---
    methy_reshaped = methy_padded.view(batch_size, n_track, num_segs, seg_length)
    cg_sites_expanded = cg_sites.unsqueeze(1).expand(-1, n_track, -1)  # Shape: (batch_size, n_track, padded_length)
    cg_sites_reshaped = cg_sites_expanded.view(batch_size, n_track, num_segs,
                                               seg_length)  # Shape: (batch_size, n_track, 100, 100)

    valid_cg_mask = cg_sites_reshaped & (methy_reshaped >= 0)
    sum_methy = torch.where(valid_cg_mask, methy_reshaped, torch.zeros_like(methy_reshaped)).sum(dim=-1)
    count_valid_cg = valid_cg_mask.sum(dim=-1, dtype=torch.float)
    avg_methy = sum_methy / (count_valid_cg + 1e-8)

    # --- Region Mask Calculation ---
    total_cg = cg_counts.unsqueeze(1).float().expand(-1, n_track, -1)

    region_valid_ratio = count_valid_cg / (total_cg + 1e-8)
    region_mask = (region_valid_ratio >= 0.8).float()

    segment_mask = torch.arange(num_segs, device=device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_segs)
    segment_mask = (segment_mask < num_original_segs).float()  # Shape: (1, 1, num_segs)

    # --- Calculate MSE Loss ---
    squared_error = (pred_100_methy_avg - avg_methy) ** 2
    final_mask = region_mask * segment_mask
    masked_squared_error = squared_error * final_mask
    loss_avg = masked_squared_error.sum() / (final_mask.sum() + 1e-8)

    return loss_cg, loss_avg



def make_cpg_and_valid_and_low_methy_mask_multi_track(seq_tensor, methy_tensor):
    """
        seq_tensor: (bsz, 4, seq_len) (one-hot). [1, 0, 0, 0] (representing 'A'), [0, 1, 0, 0] ('C'),[0, 0, 1, 0] ('G'), and [0, 0, 0, 1] ('T').
        methy_tensor: (bsz, n_track, seq_len)
        """
    n_track = methy_tensor.shape[1]
    # Create CpG flag (C followed by G)
    bsz, _, seq_len = seq_tensor.shape
    c_sites = seq_tensor[:, 1:2, :]  # Get C positions
    g_sites = seq_tensor[:, 2:3, :]  # Get G positions

    # Initialize CpG flag with zeros
    cpg_flag = torch.zeros((bsz, n_track, seq_len), dtype=torch.float32, device=seq_tensor.device)  # shape [bsz, n_track, seq_len]

    # Mark C positions in CpG
    cpg_flag[:, :, :-1] = (c_sites[:, :, :-1] * g_sites[:, :, 1:]) > 0

    # Mark G positions in CpG (shift the mask to mark G positions that follow C)
    g_in_cpg = torch.zeros_like(cpg_flag)
    g_in_cpg[:, :, 1:] = cpg_flag[:, :, :-1]

    # Combine both C and G positions in CpG
    combined_cpg = torch.max(cpg_flag, g_in_cpg)

    cpg_mask = combined_cpg.to(dtype=torch.bool)  # 1 for CpG sites (both C and G), 0 for non-CpG sites
    valid_mask = (methy_tensor >= 0)  # shape: [bsz, n_track, seq_len]
    low_methy_mask = (methy_tensor < 0.5)  # shape: [bsz, n_track, seq_len]
    # Combine masks
    combined_mask = cpg_mask * valid_mask * low_methy_mask  # shape: [bsz, n_track, seq_len]

    return cpg_mask, valid_mask, low_methy_mask, combined_mask

def mask_cell_region(cell_region_mask):
    cell_region_mask = cell_region_mask.bool()
    # reshape mask
    cell_region_mask = cell_region_mask.unsqueeze(1)  # shape: [8, 1, 10000]
    return cell_region_mask

def mask2alpha_multitrack(
        methy_tensor,  # (bsz, n_track, seq_len)
        cpg_mask,  # (bsz, seq_len)
        central_value=0.5
):

    alpha = torch.where(cpg_mask, torch.abs(methy_tensor - central_value) * 10,
                        torch.tensor(1.0, device=methy_tensor.device))
    # print(alpha)
    return alpha

def get_alpha_tensor(methy_tensor, args, valid_cpg_mask, combined_mask, batch_idx_valid, track_idx_valid, pos_idx_valid):
    if args.use_advanced_focal_loss:
        alpha = mask2alpha_multitrack(methy_tensor, valid_cpg_mask)[batch_idx_valid, track_idx_valid, pos_idx_valid]
    elif args.use_fixed_focal_cpg_loss:

        alpha_cpg_low_methy = mask2value(combined_mask,
                                         {True: args.low_methy_cpg_focal_weight - args.any_cpg_focal_weight,
                                          False: 1.0})
        # filtered_valid_cpg_mask = valid_cpg_mask[batch_idx_valid, track_idx_valid, pos_idx_valid].unsqueeze(-1)
        alpha_valid_cpg_mask = mask2value(valid_cpg_mask,
                                          {True: args.any_cpg_focal_weight, False: 0.0})
        alpha = alpha_cpg_low_methy + alpha_valid_cpg_mask
    else:
        alpha = torch.ones_like(methy_tensor[batch_idx_valid, track_idx_valid, pos_idx_valid], device=methy_tensor.device)

    return alpha

def get_alpha_tensor_G2(methy_tensor, args, valid_cpg_mask, combined_mask, cell_region_mask, batch_idx_valid, track_idx_valid, pos_idx_valid):
    if args.use_advanced_focal_loss:
        alpha = mask2alpha_multitrack(methy_tensor, valid_cpg_mask)[batch_idx_valid, track_idx_valid, pos_idx_valid]
    elif args.use_fixed_focal_cpg_loss:
        # filtered_combined_mask = combined_mask[batch_idx_valid, track_idx_valid, pos_idx_valid].unsqueeze(-1)
        # alpha = mask2value(filtered_combined_mask, {True: args.cpg_focal_weight, False: 1.0})

        alpha_cpg_low_methy = mask2value(combined_mask,
                                         {True: args.low_methy_cpg_focal_weight - args.any_cpg_focal_weight,
                                          False: 1.0})
        # filtered_valid_cpg_mask = valid_cpg_mask[batch_idx_valid, track_idx_valid, pos_idx_valid].unsqueeze(-1)
        alpha_valid_cpg_mask = mask2value(valid_cpg_mask,
                                          {True: args.any_cpg_focal_weight, False: 0.0})
        alpha = alpha_cpg_low_methy + alpha_valid_cpg_mask
    else:
        alpha = torch.ones_like(methy_tensor[batch_idx_valid, track_idx_valid, pos_idx_valid], device=methy_tensor.device)

    alpha_cell_region_mask = mask2value(cell_region_mask, {True: 20.0, False: 0.0})

    alpha = alpha + alpha_cell_region_mask

    # alpha_cell_region_cpg_mask = valid_cpg_mask * cell_region_mask
    # alpha_cell_region_cpg_mask = mask2value(alpha_cell_region_cpg_mask, {True: 32.0, False: 1.0})
    #
    # alpha = alpha_cell_region_cpg_mask

    return alpha

def get_alpha_tensor_G1(methy_tensor, args, valid_cpg_mask, combined_mask, batch_idx_valid, track_idx_valid, pos_idx_valid):
    if args.use_advanced_focal_loss:
        alpha = mask2alpha_multitrack(methy_tensor, valid_cpg_mask)[batch_idx_valid, track_idx_valid, pos_idx_valid]
    elif args.use_fixed_focal_cpg_loss:
        # filtered_combined_mask = combined_mask[batch_idx_valid, track_idx_valid, pos_idx_valid].unsqueeze(-1)
        # alpha = mask2value(filtered_combined_mask, {True: args.cpg_focal_weight, False: 1.0})

        alpha_cpg_low_methy = mask2value(combined_mask,
                                         {True: args.low_methy_cpg_focal_weight - args.any_cpg_focal_weight,
                                          False: 1.0})
        # filtered_valid_cpg_mask = valid_cpg_mask[batch_idx_valid, track_idx_valid, pos_idx_valid].unsqueeze(-1)
        alpha_valid_cpg_mask = mask2value(valid_cpg_mask,
                                          {True: args.any_cpg_focal_weight, False: 0.0})
        alpha = alpha_cpg_low_methy + alpha_valid_cpg_mask
    else:
        alpha = torch.ones_like(methy_tensor[batch_idx_valid, track_idx_valid, pos_idx_valid], device=methy_tensor.device)

    return alpha


def get_real_pred_and_cg_avg_loss_from_raw_model_output(model_output, seq_tensor, methy_tensor, args):
    if isinstance(model_output, tuple):
        pred, pred_cg, pred_methy_avg = model_output
        loss_cg, loss_avg = calc_cg_and_avg_loss_variable_len(
            seq=seq_tensor.transpose(1, 2),
            methy=methy_tensor,
            pred_100_cg_count=pred_cg,
            pred_100_methy_avg=pred_methy_avg,
            cpg_cls_n=args.cpg_cls_n,
            cpg_cls_bin_width=args.cpg_cls_bin_width,
        )
    else:
        pred, loss_cg, loss_avg = model_output, 0, 0
    loss_cg, loss_avg = loss_cg * args.wcg, loss_avg * args.wavg
    return pred, loss_cg, loss_avg

def collect_track_losses(
        bigwigs,
        batch_idx_valid: Tensor,
        track_idx_valid: Tensor,
        pos_idx_valid: Tensor,
        model,
        pred,
        bin_methy,
        i,
        alpha: Tensor,
        flag_use_focal
):
    loss_di = {}
    for current_loop_track_idx, track_name in enumerate(bigwigs):
        track_name = track_name.replace('.hg38.bigwig', '')
        is_current_track = (track_idx_valid == current_loop_track_idx)
        b_idx_specific_track = batch_idx_valid[is_current_track]
        p_idx_specific_track = pos_idx_valid[is_current_track]
        track_filtered_predictions = pred[b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]
        track_filtered_ground_truth = bin_methy[b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]
        alpha_for_track = None if not flag_use_focal else alpha[
            b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]
        track_loss = model.loss_fn(track_filtered_predictions, track_filtered_ground_truth, alpha=alpha_for_track)

        if torch.isnan(track_loss) or torch.isinf(track_loss) or track_loss > 1e7:
            msg = f"NaN loss detected for track {track_name} at step {i}. is Nan: {torch.isnan(track_loss)}, is Inf: {torch.isinf(track_loss)}. Value: {track_loss}"
            logger.error(msg)
            try:
                loguru.logger.error(f"ERR loss detected at step {i}, {msg}")
            except Exception as e:
                logger.error(e)
            continue
        loss_di[track_name] = track_loss
    loss_tracks = torch.mean(torch.stack(list(loss_di.values())))
    return loss_tracks, loss_di

def collect_cell_losses(
        bigwigs,
        batch_idx_valid: Tensor,
        track_idx_valid: Tensor,
        pos_idx_valid: Tensor,
        model,
        pred,
        bin_methy,
        i,
        alpha: Tensor,
        flag_use_focal
):
    loss_di = {}
    # for current_loop_track_idx, track_name in enumerate(bigwigs[0]):
    track_name = 'cell'
    current_loop_track_idx = 0
    is_current_track = (track_idx_valid == current_loop_track_idx)
    b_idx_specific_track = batch_idx_valid[is_current_track]
    p_idx_specific_track = pos_idx_valid[is_current_track]
    track_filtered_predictions = pred[b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]
    track_filtered_ground_truth = bin_methy[b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]

    alpha_for_track = None if not flag_use_focal else alpha[
        b_idx_specific_track, current_loop_track_idx, p_idx_specific_track]
    track_loss = model.loss_fn(track_filtered_predictions, track_filtered_ground_truth, alpha=alpha_for_track)

    if torch.isnan(track_loss) or torch.isinf(track_loss) or track_loss > 1e7:
        msg = f"NaN loss detected for track {track_name} at step {i}. is Nan: {torch.isnan(track_loss)}, is Inf: {torch.isinf(track_loss)}. Value: {track_loss}"
        logger.error(msg)
        try:
            loguru.logger.error(f"ERR loss detected at step {i}, {msg}")
        except Exception as e:
            logger.error(e)
    loss_di[track_name] = track_loss
    loss_tracks = torch.mean(torch.stack(list(loss_di.values())))
    return loss_tracks, loss_di

def parse_cg_class_n_cuda(cg_counts, cpg_cls_bin_width: int, cpg_cls_n: int):
    """
    Optimized version to classify CpG counts into n classes using vectorized operations.

    Class 0 is always assigned to segments with 0 CpG counts.
    For segments with counts > 0:
    - Class 1: 1 to cpg_cls_bin_width counts
    - Class 2: cpg_cls_bin_width + 1 to 2 * cpg_cls_bin_width counts
    - ...
    - Class (cpg_cls_n - 2): (cpg_cls_n - 3) * cpg_cls_bin_width + 1 to (cpg_cls_n - 2) * cpg_cls_bin_width counts
    - Class (cpg_cls_n - 1): > (cpg_cls_n - 2) * cpg_cls_bin_width counts

    Parameters
    ----------
    cg_counts : torch.Tensor
        Tensor containing CpG counts for each segment. Expected integer type.
    cpg_cls_bin_width : int
        The width of each bin for positive counts (exclusive of class 0 and the max class).
    cpg_cls_n : int
        The total number of classes (must be >= 2).

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as cg_counts, containing class indices (dtype=torch.long).
    """
    if cpg_cls_bin_width <= 0:
        raise ValueError("cpg_cls_bin_width must be a positive integer.")
    if cpg_cls_n < 2:
        raise ValueError("cpg_cls_n must be at least 2 (for class 0 and at least one positive class).")

    # Ensure calculations are done with sufficient precision if counts can be large,
    # but intermediate calculations here should be fine with standard int types.
    # The original code used integer division, let's stick to that logic.

    # Calculate the 0-based bin index for counts >= 1
    # (counts - 1) maps counts [1, width] to [0, width-1], counts [width+1, 2*width] to [width, 2*width-1], etc.
    # Integer division by width maps these ranges to 0, 1, etc.
    # Use torch.div for explicit floor rounding, safer with potential negative results if counts could be >= 0 (though unlikely here)
    # or just use // which is floor division in Python/PyTorch.
    # Let's use // for simplicity as cg_counts are expected >= 0.
    # For counts=0, (0-1)//width = -1.
    # For counts=1..width, (1-1)//width to (width-1)//width = 0.
    # For counts=width+1..2*width, (width)//width to (2*width-1)//width = 1.
    zero_based_bin_index = (cg_counts - 1) // cpg_cls_bin_width

    # Add 1 to get the 1-based class index (maps bin 0 to class 1, bin 1 to class 2, etc.)
    # For counts=0, this results in class 0 (-1 + 1).
    calculated_class = zero_based_bin_index + 1

    # Determine the maximum valid class index
    max_class_index = cpg_cls_n - 1

    # Clamp the results:
    # - Ensure minimum class is 0 (handles the count=0 case correctly).
    # - Ensure maximum class is max_class_index (handles counts falling into the last bin).
    final_class = torch.clamp(calculated_class, min=0, max=max_class_index)

    # Ensure output is long type, suitable for cross_entropy targets
    return final_class.long()
