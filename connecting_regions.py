import cooler
import pandas as pd
import numpy as np
import multiprocess as mp
from pileup import _fill_zero_by_diag, transform_coor_to_bin_id, _norm_matrix
from typing import List


def connecting_strength_over_background(
    clr: cooler.Cooler,
    loop_bedpe_chunker,
    diag_expected,
    peak_flank=2,
    initial_search_flank=7,
    max_search_flank=20,
    na_cutoff=3,
    ignore_diag=2,
    fold_over_background=1.5,
    parallel=1,
    mad_max_mask_bins=None,
    norm_method="obexp",
    local_norm=False,
    fill_zero=True,
    balanced=True,
    sampling_ratio=1,
):
    # if diag_expected is None:
    #     diag_expected = expected_by_diag_avg(clr)

    with mp.Pool(parallel) as pool:
        results = pool.map(
            lambda loop_chunk: _connecting_strength_over_background(
                clr,
                loop_chunk,
                diag_expected,
                peak_flank,
                initial_search_flank,
                max_search_flank,
                na_cutoff,
                ignore_diag,
                fold_over_background,
                mad_max_mask_bins,
                norm_method,
                local_norm,
                fill_zero,
                balanced,
                sampling_ratio,
            ),
            loop_bedpe_chunker,
        )

    filtered_loops = pd.DataFrame()
    for loops in results:
        filtered_loops = pd.concat([filtered_loops, loops], ignore_index=True)

    return filtered_loops


def _connecting_strength_over_background(
    clr: cooler.Cooler,
    loop_bedpe,
    diag_expected,
    peak_flank=2,
    initial_search_flank=7,
    max_search_flank=20,
    na_cutoff=3,
    ignore_diag=2,
    fold_over_background=1.5,
    mad_max_mask_bins=None,
    norm_method="obexp",
    local_norm=False,
    fill_zero=True,
    balanced=True,
    sampling_ratio=1,
):
    loop_with_cs = pd.DataFrame()
    if sampling_ratio < 1:
        loop_bedpe = loop_bedpe.sample(
            int(np.ceil(len(loop_bedpe) * sampling_ratio))
        )
    for chro, chro_loops in loop_bedpe.groupby("chr1"):
        if chro not in clr.chromnames:
            continue
        chro_bins, chro_offset = clr.bins().fetch(chro), clr.offset(chro)
        chro_bins["ID"] = chro_bins.index
        mat = (
            clr.matrix(balance=balanced, sparse=True).fetch(chro).astype(float)
        )
        n = mat.shape[0]

        # since sparse matrix change NaN to 0, we keep track of which bins should be mask as NaN
        if mad_max_mask_bins is not None:
            mad_max_mask_vec = chro_bins.index.isin(mad_max_mask_bins.index)
        else:
            mad_max_mask_vec = np.zeros(len(chro_bins)).astype(bool)
        chro_nan_bin_index = np.arange(n)[
            chro_bins.weight.isnull().values | mad_max_mask_vec
        ]

        diag_pseudo_zero = _norm_matrix(
            mat, norm_method, chro, diag_expected, fill_zero
        )
        quant_mat = mat.tocsr()

        # fine loops' bid coordinates
        left_bid_coors = transform_coor_to_bin_id(
            chro_bins[["chrom", "start", "end", "ID"]],
            chro_loops[["chr1", "x1", "y1"]],
            chro_offset,
        )
        right_bid_coors = transform_coor_to_bin_id(
            chro_bins[["chrom", "start", "end", "ID"]],
            chro_loops[["chr2", "x2", "y2"]],
            chro_offset,
        )
        central_xs = (
            left_bid_coors.start_bid.values + left_bid_coors.end_bid.values
        ) // 2
        central_ys = (
            right_bid_coors.start_bid.values + right_bid_coors.end_bid.values
        ) // 2
        loop_entries = [
            (x, y)
            for x, y in zip(central_xs.astype(int), central_ys.astype(int))
        ]

        (
            signal_to_gbg,
            local_avg,
            local_var,
            left_insul,
            right_insul,
        ) = _search_na_loop_neighborhood(
            quant_mat,
            loop_entries,
            peak_flank,
            initial_search_flank,
            max_search_flank,
            ignore_diag,
            na_cutoff,
            chro_nan_bin_index,
            diag_pseudo_zero,
        )

        # chro_loops["CS"] = (
        #     weight[0] * signal_to_gbg + weight[1] * signal_to_gbg / local_avg
        # )
        if local_norm:
            if norm_method == "obexp":
                chro_loops["CS"] = signal_to_gbg / local_avg
            elif norm_method == "znorm":
                chro_loops["CS"] = (signal_to_gbg - local_avg) / local_var
        else:
            chro_loops["CS"] = signal_to_gbg

        chro_loops["LI"] = left_insul
        chro_loops["RI"] = right_insul
        # What type of pair will be dropout:
        # 1. chromosome end cases including small chromosome (very rare)
        # 2. sparese area determined by nz_cutoff (also suggest locus is sparse and cannot be reliably quantified)
        # 3. dot center NaN
        loop_with_cs = pd.concat(
            [loop_with_cs, chro_loops[chro_loops.CS >= fold_over_background]],
            ignore_index=True,
        )

    return loop_with_cs


def _search_na_loop_neighborhood(
    quant_mat,
    loop_entries,
    peak_flank,
    initial_search_flank,
    max_search_flank,
    ignore_diag,
    na_cutoff,
    nan_bin_index,
    diag_pseudo_zero,
):
    """Compute the center and local background signal of every loop entry."""
    signal_to_gbg, local_avg_bg, local_var_bg, left_insul, right_insul = (
        np.zeros(len(loop_entries)) * np.nan,
        np.zeros(len(loop_entries)) * np.nan,
        np.zeros(len(loop_entries)) * np.nan,
        np.zeros(len(loop_entries)) * np.nan,
        np.zeros(len(loop_entries)) * np.nan,
    )
    for i, (x, y) in enumerate(loop_entries):
        max_sub_mat = _take_submat(
            quant_mat,
            nan_bin_index,
            x,
            y,
            max_search_flank,
            ignore_diag,
            True,
            diag_pseudo_zero,
        )
        if max_sub_mat is None:
            continue
        # center signal. possible values: NaN for masked bins, 0 no observed but not mask, numbers
        signal_to_gbg[i] = max_sub_mat[
            max_search_flank,
            max_search_flank,
        ]
        for j in range(initial_search_flank, max_search_flank + 1):
            # growing size of lower left matrix (local background) until it passes sparsity threshold
            lower_left_mat = max_sub_mat[
                max_search_flank + peak_flank + 1 : max_search_flank + j + 1,
                max_search_flank - j : max_search_flank - peak_flank,
            ]
            local_left_insul = max_sub_mat[
                max_search_flank + 1 : max_search_flank + 6,
                : max_search_flank - peak_flank,
            ].mean(axis=0)
            local_right_insul = max_sub_mat[
                max_search_flank + 1 : max_search_flank + 6,
                max_search_flank + peak_flank + 1 :,
            ].mean(axis=0)
            # if np.nansum(lower_left_mat > 0) >= nz_cutoff:
            if np.sum(np.logical_not(np.isnan(lower_left_mat))) >= na_cutoff:
                std = np.nanstd(lower_left_mat)
                if std != 0:
                    local_avg_bg[i] = np.nanmean(lower_left_mat)
                    local_var_bg[i] = std
                    left_insul[i] = np.nanmax(local_left_insul)
                    right_insul[i] = np.nanmin(local_right_insul)
                    break

    return signal_to_gbg, local_avg_bg, local_var_bg, left_insul, right_insul


def chro_pairs_generator(loci_1, loci_2, low, high, one_side=False):
    chromosomes = loci_1.Chromosome.unique()
    for chro in chromosomes:
        chro_loci_1 = loci_1[loci_1.Chromosome == chro].copy()
        chro_loci_2 = loci_2[loci_2.Chromosome == chro].copy()
        chro_loci_1["key"] = 1
        chro_loci_2["key"] = 1
        chro_pairs = pd.merge(
            chro_loci_1, chro_loci_2, on="key", suffixes=("1", "2")
        )
        chro_pairs = chro_pairs.rename(
            columns={
                "Chromosome1": "chr1",
                "Start1": "x1",
                "End1": "y1",
                "Chromosome2": "chr2",
                "Start2": "x2",
                "End2": "y2",
            }
        )
        chro_pairs = chro_pairs.drop(columns="key")

        # filter distance
        dist = chro_pairs.x2 - chro_pairs.x1
        if one_side:
            # if loci_1 == loci_2, use this to avoid counting in duplicating interactions
            yield chro_pairs[(dist >= low) & (dist <= high)]
        else:
            yield chro_pairs[(np.abs(dist) >= low) & (np.abs(dist) <= high)]


def comparative_connecting_strength(
    clrs: List[cooler.Cooler],
    loop_bedpe_chunker,
    parallel=1,
    diag_expected=None,
    mad_max_mask_bins=None,
    ignore_diag=2,
    preserve_anchor_order=True,
    norm_method="znorm",
    peak_flank=2,
    initial_search_flank=7,
    max_search_flank=20,
    non_nan_cutoff=3,
):
    if norm_method != "znorm":
        raise NotImplementedError()

    with mp.Pool(parallel) as pool:
        results = pool.map(
            lambda loop_chunk: _comparative_cs_over_background(
                clrs[0],
                clrs[1],
                loop_chunk,
                ignore_diag,
                diag_expected,
                mad_max_mask_bins,
                preserve_anchor_order,
                norm_method,
                peak_flank,
                initial_search_flank,
                max_search_flank,
                non_nan_cutoff,
            ),
            loop_bedpe_chunker,
        )

    filtered_loops = pd.DataFrame()
    for loops in results:
        filtered_loops = pd.concat([filtered_loops, loops], ignore_index=True)

    return filtered_loops


def _comparative_cs_over_background(
    clr1: cooler.Cooler,
    clr2: cooler.Cooler,
    loop_bedpe,
    ignore_diag=2,
    diag_expected=None,
    mad_max_mask_bins=None,
    preserve_anchor_order=True,
    norm_method="znorm",
    peak_flank=2,
    initial_search_flank=7,
    max_search_flank=20,
    non_nan_cutoff=3,
):
    """ """
    quant_mat, chro_nan_bin_index, diag_pseudo_zero = [], [], []
    for chro, chro_loops in loop_bedpe.groupby("chr1"):
        # compute quant_mat and auxiliary data
        if (chro not in clr1.chromnames) or (chro not in clr2.chromnames):
            continue
        for i, clr in enumerate([clr1, clr2]):
            chro_bins, chro_offset = clr.bins().fetch(chro), clr.offset(chro)
            chro_bins["ID"] = chro_bins.index
            mat = clr.matrix(balance=True, sparse=True).fetch(chro)
            n = mat.shape[0]
            if mad_max_mask_bins is not None:
                mad_max_mask_vec = chro_bins.index.isin(
                    mad_max_mask_bins[i].index
                )
            else:
                mad_max_mask_vec = np.zeros(len(chro_bins)).astype(bool)
            chro_nan_bin_index_ = np.arange(n)[
                chro_bins.weight.isnull().values | mad_max_mask_vec
            ]

            diag_pseudo_zero_ = _norm_matrix(
                mat, norm_method, chro, diag_expected[i], True
            )
            diag_pseudo_zero.append(diag_pseudo_zero_)

            quant_mat.append(mat.tocsr())
            chro_nan_bin_index.append(chro_nan_bin_index_)

        # compute diff sub mat
        (
            signal_to_gbg,
            local_avg,
            local_std,
        ) = _chro_loop_comparative_neighborhood(
            quant_mat,
            chro_loops,
            chro_nan_bin_index,
            diag_pseudo_zero,
            chro_offset,
            chro_bins,
            ignore_diag,
            preserve_anchor_order,
            peak_flank,
            initial_search_flank,
            max_search_flank,
            non_nan_cutoff,
        )

        chro_loops["Diff"] = signal_to_gbg
        chro_loops["BG_Avg"] = local_avg
        chro_loops["BG_Var"] = local_std

    return chro_loops


def _chro_loop_comparative_neighborhood(
    quant_mat,
    chro_loops: pd.DataFrame,
    chro_nan_bin_index,
    diag_pseudo_zero,
    chro_offset,
    chro_bins,
    ignore_diag,
    preserve_anchor_order,
    peak_flank,
    initial_search_flank,
    max_search_flank,
    non_nan_cutoff,
):

    signal_to_gbg, local_avg_bg, local_std_bg = (
        np.zeros(len(chro_loops)) * np.nan,
        np.zeros(len(chro_loops)) * np.nan,
        np.zeros(len(chro_loops)) * np.nan,
    )
    # fine loops' bid coordinates
    left_bid_coors = transform_coor_to_bin_id(
        chro_bins[["chrom", "start", "end", "ID"]],
        chro_loops[["chr1", "x1", "y1"]],
        chro_offset,
    )
    right_bid_coors = transform_coor_to_bin_id(
        chro_bins[["chrom", "start", "end", "ID"]],
        chro_loops[["chr2", "x2", "y2"]],
        chro_offset,
    )

    central_xs = (
        left_bid_coors.start_bid.values + left_bid_coors.end_bid.values
    ) // 2
    central_ys = (
        right_bid_coors.start_bid.values + right_bid_coors.end_bid.values
    ) // 2
    central_xs, central_ys = central_xs.astype(int), central_ys.astype(int)
    for i, (central_x_, central_y_) in enumerate(zip(central_xs, central_ys)):
        # only extract upper part
        sub_mat1 = _take_submat(
            quant_mat[0],
            chro_nan_bin_index[0],
            central_x_,
            central_y_,
            max_search_flank,
            ignore_diag,
            preserve_anchor_order,
            diag_pseudo_zero[0],
        )
        sub_mat2 = _take_submat(
            quant_mat[1],
            chro_nan_bin_index[1],
            central_x_,
            central_y_,
            max_search_flank,
            ignore_diag,
            preserve_anchor_order,
            diag_pseudo_zero[1],
        )

        # nz detect
        # center signal. possible values: NaN for masked bins, 0 no observed but not mask, numbers
        if sub_mat1 is None or sub_mat2 is None:
            continue
        diff_mat = sub_mat1 - sub_mat2
        # neigbor_size = 2 * max_search_flank + 1
        # if (
        #     diff_mat.shape[0] != neigbor_size
        #     or diff_mat.shape[1] != neigbor_size
        # ):
        #     continue

        signal_to_gbg[i] = diff_mat[
            max_search_flank,
            max_search_flank,
        ]

        for j in range(initial_search_flank, max_search_flank + 1):
            # growing size of lower left matrix (local background) until it passes sparsity threshold
            lower_left_mat = diff_mat[
                max_search_flank + peak_flank + 1 : max_search_flank + j + 1,
                max_search_flank - j : max_search_flank - peak_flank,
            ]
            if (
                np.sum(np.logical_not(np.isnan(lower_left_mat)))
                >= non_nan_cutoff
            ):
                local_avg_bg[i] = np.nanmean(lower_left_mat)
                local_std_bg[i] = np.nanstd(lower_left_mat)
                break

    return signal_to_gbg, local_avg_bg, local_std_bg


def _take_submat(
    quant_mat,
    chro_nan_bin_index,
    central_x_,
    central_y_,
    flank_bins,
    ignore_diag,
    preserve_anchor_order,
    diag_pseudo_zero,
):
    # only extract upper part
    if central_x_ > central_y_:
        central_x, central_y = central_y_, central_x_
    else:
        central_x, central_y = central_x_, central_y_
    sub_mat = quant_mat[
        central_x - flank_bins : central_x + flank_bins + 1,
        central_y - flank_bins : central_y + flank_bins + 1,
    ].toarray()

    # mask diag
    lowest_diag_index = central_y - flank_bins - (central_x + flank_bins)
    m = sub_mat.shape[0]
    if lowest_diag_index <= ignore_diag:
        ori_nan_num = np.sum(np.isnan(sub_mat))
        n_diag_to_mask = ignore_diag - lowest_diag_index + 1
        view_mask_sub_mat = sub_mat[m - n_diag_to_mask : m, :n_diag_to_mask]
        # this assignment will take effect on the original sub_mat
        view_mask_sub_mat[np.tril_indices_from(view_mask_sub_mat)] = np.nan
        # if ori_nan_num >= np.sum(np.isnan(sub_mat)):
        #     raise ValueError(f"Fill NaN fail, {lowest_diag_index}, {ignore_diag}")

    # fill zeros with diagonal lowest signal
    if diag_pseudo_zero is not None:
        main_diag = central_y - central_x
        sub_mat = _fill_zero_by_diag(sub_mat, main_diag, diag_pseudo_zero)

    neigbor_size = 2 * flank_bins + 1
    if sub_mat.shape[0] != neigbor_size or sub_mat.shape[1] != neigbor_size:
        return None

    # handle NaN value. Using sparse matrix may change NaN entry to 0
    row_range = np.arange(central_x - flank_bins, central_x + flank_bins + 1)
    col_range = np.arange(central_y - flank_bins, central_y + flank_bins + 1)
    sub_mat[
        np.isin(row_range, chro_nan_bin_index, assume_unique=True),
        :,
    ] = np.nan
    sub_mat[
        :,
        np.isin(col_range, chro_nan_bin_index, assume_unique=True),
    ] = np.nan

    if central_x_ > central_y_ and preserve_anchor_order:
        # flip the matrix to preserve left anchor on the row, right anchor on the col
        sub_mat = np.fliplr(np.transpose(np.fliplr(sub_mat)))

    return sub_mat


def virtual_4C(clr, bait, prey):
    """
    Generate 4C profile
    bait: ("chr1", 4000000, 500000)
    prey: (1000000, 2000000), indicate to extend 1M to the left and 2M to the right
    """
    chro, start, end = bait
    chro_bins, chro_offset = clr.bins().fetch(chro), clr.offset(chro)
    chro_bins["ID"] = chro_bins.index
    mat = clr.matrix(balance=True, sparse=True).fetch(chro).tocsr()

    bait_bid_coors = transform_coor_to_bin_id(
        chro_bins[["chrom", "start", "end", "ID"]],
        pd.DataFrame({"chrom": [chro], "start": start, "end": end}),
        chro_offset,
    )
    prey_bid_coors = transform_coor_to_bin_id(
        chro_bins[["chrom", "start", "end", "ID"]],
        pd.DataFrame(
            {"chrom": [chro], "start": start - prey[0], "end": end + prey[1]}
        ),
        chro_offset,
    )
    row = bait_bid_coors.start_bid[0], bait_bid_coors.end_bid[0]
    col = prey_bid_coors.start_bid[0], prey_bid_coors.end_bid[0]
    # symetric matrix
    track_4c = (
        mat[row[0] : row[1] + 1, col[0] : col[1] + 1]
        + mat[col[0] : col[1] + 1, row[0] : row[1] + 1].T
    )
    track_4c = track_4c.toarray()
    n, m = track_4c.shape
    # scale back diagonal
    offset = (m - n) // 2
    for i, j in enumerate(range(offset, offset + n)):
        track_4c[i, j] /= 2
    # bin coordinate
    track_4c_coor = chro_bins.start.values[col[0] : col[1] + 1]
    return track_4c, track_4c_coor
