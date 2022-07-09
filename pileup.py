from typing import List
import cooler
from numpy.lib import math
import pandas as pd
import numpy as np
import cooltools.expected
import multiprocess as mp
import pyranges as pr
from numba import njit


def pileup_at_loops(
    clr: cooler.Cooler,
    loop_bedpe_chunker,
    flank_bins=10,
    parallel=1,
    quantify_method="obexp",
    diag_expected=None,
    mad_max_mask_bins=None,
    ignore_diag=2,
    preserve_anchor_order=True,
    fill_zero=False,
    balanced=True,
):
    # if quantify_method == "expected" and diag_expected is None:
    #     diag_expected = expected_by_diag_avg(clr)

    with mp.Pool(parallel) as pool:
        results = pool.map(
            lambda loop_chunk: _pileup_at_loops(
                clr,
                loop_chunk,
                flank_bins,
                quantify_method,
                ignore_diag,
                diag_expected,
                mad_max_mask_bins,
                preserve_anchor_order,
                fill_zero,
                balanced,
            ),
            loop_bedpe_chunker,
        )

    n = flank_bins * 2 + 1
    count_mat, sum_mat = np.zeros((n, n)), np.zeros((n, n))
    for c, s in results:
        count_mat += c
        sum_mat += s

    return count_mat, sum_mat


def _pileup_at_loops(
    clr: cooler.Cooler,
    loop_bedpe,
    flank_bins=10,
    quantify_method="obexp",
    ignore_diag=2,
    diag_expected=None,
    mad_max_mask_bins=None,
    preserve_anchor_order=True,
    fill_zero=False,
    balanced=True,
):
    sub_mat_gen = _loop_neighborhood_generator(
        clr,
        loop_bedpe,
        flank_bins,
        quantify_method,
        ignore_diag,
        diag_expected,
        mad_max_mask_bins,
        preserve_anchor_order,
        fill_zero,
        balanced,
    )
    n = 2 * flank_bins + 1
    count_mat, sum_mat = np.zeros((n, n)), np.zeros((n, n))
    for loop_submat in sub_mat_gen:
        valid_index = np.logical_not(np.isnan(loop_submat))
        count_mat += valid_index
        sum_mat[valid_index] += loop_submat[valid_index]

    return count_mat, sum_mat


def _loop_neighborhood_generator(
    clr: cooler.Cooler,
    loop_bedpe: pd.DataFrame,
    flank_bins,
    quantify_method,
    ignore_diag,
    diag_expected=None,
    mad_max_mask_bins=None,
    preserve_anchor_order=True,
    fill_zero=False,
    balanced=True,
):
    """Generate neighborhood sub-matrix for each loop entry"""
    for chro, chro_loops in loop_bedpe.groupby("chr1"):
        if chro not in clr.chromnames:
            continue
        chro_bins, chro_offset = clr.bins().fetch(chro), clr.offset(chro)
        chro_bins["ID"] = chro_bins.index
        mat = (
            clr.matrix(balance=balanced, sparse=True).fetch(chro).astype(float)
        )
        n = mat.shape[0]
        if mad_max_mask_bins is not None:
            mad_max_mask_vec = chro_bins.index.isin(mad_max_mask_bins.index)
        else:
            mad_max_mask_vec = np.zeros(len(chro_bins)).astype(bool)
        chro_nan_bin_index = np.arange(n)[
            chro_bins.weight.isnull().values | mad_max_mask_vec
        ]

        diag_pseudo_zero = _norm_matrix(
            mat, quantify_method, chro, diag_expected, fill_zero
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
        central_xs, central_ys = central_xs.astype(int), central_ys.astype(int)
        for central_x_, central_y_ in zip(
            central_xs,
            central_ys,
        ):
            sub_mat = _take_submat(
                quant_mat,
                chro_nan_bin_index,
                central_x_,
                central_y_,
                flank_bins,
                ignore_diag,
                preserve_anchor_order,
                diag_pseudo_zero,
            )

            yield sub_mat


def expected_by_diag_avg(clr, balanced=True):
    chromsizes, chromosomes = clr.chromsizes, clr.chromnames
    supports = [(chrom, 0, chromsizes[chrom]) for chrom in chromosomes]
    # Calculate expected interactions for chromosome
    with mp.Pool(8) as pool:
        if balanced:
            expected = cooltools.expected.diagsum(
                clr,
                regions=supports,
                transforms={
                    "balanced": lambda p: p["count"]
                    * p["weight1"]
                    * p["weight2"]
                },
                map=pool.map,
            )
        else:
            expected = cooltools.expected.diagsum(
                clr,
                regions=supports,
                transforms={"balanced": lambda p: p["count"]},
                map=pool.map,
            )

    # Calculate average number of interactions per diagonal
    expected["balanced.avg"] = expected["balanced.sum"] / expected["n_valid"]
    expected["region"] = [
        region.split(":")[0] for region in expected.region.values
    ]
    return expected


def transform_coor_to_bin_id(bins: pd.DataFrame, bed: pd.DataFrame, offset):
    pr_bins = pr.PyRanges(
        bins.rename(
            columns={
                old: new
                for new, old in zip(
                    ["Chromosome", "Start", "End", "ID"], bins.columns[:3]
                )
            }
        )
    )
    pr_bed = pr.PyRanges(
        bed.rename(
            columns={
                old: new
                for new, old in zip(
                    ["Chromosome", "Start", "End"], bed.columns[:3]
                )
            }
        )
    )

    assigned = pr_bed.join(pr_bins, how="left").df[
        ["Chromosome", "Start", "End", "ID"]
    ]
    bed_bid_coors = assigned.groupby(by=["Chromosome", "Start", "End"]).apply(
        lambda rec: pd.Series(
            {
                "chrom": rec.Chromosome.iloc[0],
                "start": rec.Start.iloc[0],
                "end": rec.End.iloc[0],
                "start_bid": rec.ID.min() - offset,
                "end_bid": rec.ID.max() - offset,
            }
        )
    )
    # preserve order from input
    return bed.merge(
        bed_bid_coors,
        how="left",
        left_on=list(bed.columns[:3]),
        right_on=["chrom", "start", "end"],
    )[["chrom", "start_bid", "end_bid"]]


def comparative_pileup_at_loops(
    clrs: List[cooler.Cooler],
    loop_bedpe_chunker,
    flank_bins=10,
    parallel=1,
    diag_expected=None,
    mad_max_mask_bins=None,
    ignore_diag=2,
    preserve_anchor_order=True,
    norm_method="znorm",
    balanced=True,
):
    with mp.Pool(parallel) as pool:
        results = pool.map(
            lambda loop_chunk: _comparative_pileup_at_loops(
                clrs[0],
                clrs[1],
                loop_chunk,
                flank_bins,
                ignore_diag,
                diag_expected,
                mad_max_mask_bins,
                preserve_anchor_order,
                norm_method,
                balanced,
            ),
            loop_bedpe_chunker,
        )

    n = flank_bins * 2 + 1
    count_mat, sum_mat = np.zeros((n, n)), np.zeros((n, n))
    for c, s in results:
        count_mat += c
        sum_mat += s

    return count_mat, sum_mat


def _comparative_pileup_at_loops(
    clr1: cooler.Cooler,
    clr2: cooler.Cooler,
    loop_bedpe,
    flank_bins=10,
    ignore_diag=2,
    diag_expected=None,
    mad_max_mask_bins=None,
    preserve_anchor_order=True,
    norm_method="znorm",
    balanced=True,
):
    """ """
    n = 2 * flank_bins + 1
    count_mat, sum_mat = np.zeros((n, n)), np.zeros((n, n))
    quant_mat, chro_nan_bin_index, diag_pseudo_zero = [], [], []
    for chro, chro_loops in loop_bedpe.groupby("chr1"):
        if (chro not in clr1.chromnames) or (chro not in clr2.chromnames):
            continue
        for i, clr in enumerate([clr1, clr2]):
            chro_bins, chro_offset = clr.bins().fetch(chro), clr.offset(chro)
            chro_bins["ID"] = chro_bins.index
            mat = clr.matrix(balance=balanced, sparse=True).fetch(chro)
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

        sub_mat_gen = _chro_loop_comparative_neighborhood_generator(
            quant_mat,
            chro_loops,
            chro_nan_bin_index,
            diag_pseudo_zero,
            chro_offset,
            chro_bins,
            flank_bins,
            ignore_diag,
            preserve_anchor_order,
        )
        center_vals = []
        for loop_submat in sub_mat_gen:
            valid_index = np.logical_not(np.isnan(loop_submat))
            count_mat += valid_index
            sum_mat[valid_index] += loop_submat[valid_index]
            center_vals.append(loop_submat[flank_bins, flank_bins])

    return count_mat, sum_mat


def _chro_loop_comparative_neighborhood_generator(
    quant_mat,
    chro_loops: pd.DataFrame,
    chro_nan_bin_index,
    diag_pseudo_zero,
    chro_offset,
    chro_bins,
    flank_bins,
    ignore_diag,
    preserve_anchor_order,
):

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
    for central_x_, central_y_ in zip(central_xs, central_ys):
        # only extract upper part
        sub_mat1 = _take_submat(
            quant_mat[0],
            chro_nan_bin_index[0],
            central_x_,
            central_y_,
            flank_bins,
            ignore_diag,
            preserve_anchor_order,
            diag_pseudo_zero[0],
        )
        sub_mat2 = _take_submat(
            quant_mat[1],
            chro_nan_bin_index[1],
            central_x_,
            central_y_,
            flank_bins,
            ignore_diag,
            preserve_anchor_order,
            diag_pseudo_zero[1],
        )

        yield (sub_mat1 - sub_mat2)


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
        if ori_nan_num >= np.sum(np.isnan(sub_mat)):
            raise ValueError("Fill NaN fail")

    # fill zeros with diagonal lowest signal
    if diag_pseudo_zero is not None:
        main_diag = central_y - central_x
        sub_mat = _fill_zero_by_diag(sub_mat, main_diag, diag_pseudo_zero)

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


@njit
def _fill_zero_by_diag(mat, diag_index_offset, diag_val_to_fill):
    zero_x, zero_y = np.where(mat == 0)
    # mat[zero_x, zero_y] = diag_val_to_fill[diag_index_offset + zero_y - zero_x]
    for x, y in zip(zero_x, zero_y):
        mat[x, y] = diag_val_to_fill[diag_index_offset + y - x]
    return mat


def _get_diags_df(tri_df, n_diags=None, scaled=True):
    if n_diags is None:
        n_diags = tri_df[["I", "J"]].max().max() + 1

    def _diag_mean_var(diag):
        diag = diag[diag != 0]
        if len(diag) > 0:
            mean = np.mean(diag)
            if scaled:
                std = (
                    np.std(diag) / np.sqrt(len(diag))
                    if np.std(diag) != 0
                    else 1
                )
            else:
                std = np.std(diag) if np.std(diag) != 0 else 1

            if math.isnan(mean):
                mean = 0
            if math.isnan(std):
                std = 1
        else:
            mean = 0
            std = 1
        return mean, std

    agg_diag = tri_df.groupby(tri_df.J - tri_df.I).V.agg(
        mean=lambda x: _diag_mean_var(x)[0],
        std=lambda x: _diag_mean_var(x)[1],
    )

    means, stds = np.zeros(n_diags), np.ones(n_diags)
    for idx, row in agg_diag.iterrows():
        means[idx] = row["mean"]
        stds[idx] = row["std"]

    return means, stds


def _per_diag_nz_min(tri_df, n_diags=None):
    if n_diags is None:
        n_diags = tri_df[["I", "J"]].max().max() + 1

    def _diag_min(diag):
        diag = diag[diag != 0]
        if len(diag) > 0:
            min_val = np.nanmin(diag)
        else:
            min_val = np.nan

        return min_val

    agg_diag = tri_df.groupby(tri_df.J - tri_df.I).V.agg(
        min=lambda x: _diag_min(x),
    )

    mins = np.zeros(n_diags) * np.nan
    for idx, row in agg_diag.iterrows():
        mins[idx] = row["min"]

    return mins


def _norm_matrix(mat, norm_method, chro, diag_expected, fill_zero):
    if norm_method == "obexp":
        # ob / exp
        chro_diag_values = diag_expected[diag_expected.region == chro][
            "balanced.avg"
        ].values.copy()
        chro_diag_values[chro_diag_values == 0] = np.nan
        mat.data /= chro_diag_values[np.abs(mat.row - mat.col)]
        if fill_zero:
            chro_diag_mins = diag_expected[diag_expected.region == chro][
                "diag.min"
            ].values
            diag_pseudo_zero = chro_diag_mins / chro_diag_values
        else:
            diag_pseudo_zero = None
    elif norm_method == "znorm":
        # diag z norm
        means = diag_expected[diag_expected.region == chro]["znorm.mean"].values
        stds = diag_expected[diag_expected.region == chro]["znorm.std"].values
        mat.data -= means[np.abs(mat.row - mat.col)]
        mat.data /= stds[np.abs(mat.row - mat.col)]
        if fill_zero:
            diag_pseudo_zero = -means / stds
        else:
            diag_pseudo_zero = None

    return diag_pseudo_zero
