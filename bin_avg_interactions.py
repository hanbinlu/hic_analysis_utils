from tkinter import N
import numpy as np
import pyranges as pr
import cooltools
import cooltools.saddle, cooltools.expected
import multiprocess as mp
from pileup import expected_by_diag_avg


def compute_saddle(
    compartment,
    clr,
    valid_regions,
    min_diag_dist=2_000_000,
    n_groups=40,
    q_low=0.025,
    q_hi=0.975,
):
    # Digitize eigenvectors, i.e. group genomic bins into
    # equisized groups according to their eigenvector rank.
    N_GROUPS = (
        n_groups - 2
    )  # divide remaining 95% of the genome into 38 equisized groups, 2.5% each

    # equal binning separated by 0
    group_E1_bounds = np.append(
        np.percentile(
            compartment.E1[compartment.E1 <= 0],
            np.linspace(q_low, 1, N_GROUPS // 2 + 1) * 100,
        )[:-1],
        np.array([0]),
    )
    group_E1_bounds = np.append(
        group_E1_bounds,
        np.percentile(
            compartment.E1[compartment.E1 >= 0],
            np.linspace(0, q_hi, N_GROUPS // 2 + 1) * 100,
        )[1:],
    )

    # Assign the group to each genomic bin according to its E1, i.e. "digitize" E1.
    digitized, _ = cooltools.saddle.digitize_track(
        group_E1_bounds,
        track=(compartment, "E1"),
    )

    # Calculate expected interactions for chromosome
    with mp.Pool(10) as pool:
        expected = cooltools.expected.diagsum(
            clr,
            regions=valid_regions,
            transforms={
                "balanced": lambda p: p["count"] * p["weight1"] * p["weight2"]
            },
            map=pool.map,
        )
    # Calculate average number of interactions per diagonal
    expected["balanced.avg"] = expected["balanced.sum"] / expected["n_valid"]

    # Make a function that returns observed/expected dense matrix of an arbitrary
    # region of the Hi-C map.
    getmatrix = cooltools.saddle.make_cis_obsexp_fetcher(
        clr, (expected, "balanced.avg")
    )

    # Compute the saddle plot, i.e. the average observed/expected between genomic
    # ins as a function of their digitized E1.
    S, C = cooltools.saddle.make_saddle(
        getmatrix,
        group_E1_bounds,
        (digitized, "E1" + ".d"),
        min_diag=min_diag_dist // clr.binsize,
        regions=valid_regions,
        contact_type="cis",
    )

    return S, C, group_E1_bounds


def compute_saddle_generic(
    compartment,
    clr,
    min_diag_dist=2_000_000,
    n_groups=40,
    grouping_center=0,
    q_low=0.025,
    q_hi=0.975,
    parallel=8,
    n_bins_for_stats=5,
):
    # Digitize eigenvectors, i.e. group genomic bins into
    # equisized groups according to their eigenvector rank.
    N_GROUPS = (
        n_groups - 2
    )  # divide remaining 97.5% of the genome into 38 equisized groups, 2.5% each

    # bin eigenvector track
    if grouping_center is not None:
        # equal binning separated by grouping center
        group_E1_bounds = np.append(
            np.percentile(
                compartment.E1[compartment.E1 <= grouping_center],
                np.linspace(q_low, 1, N_GROUPS // 2 + 1) * 100,
            )[:-1],
            np.array([0]),
        )
        group_E1_bounds = np.append(
            group_E1_bounds,
            np.percentile(
                compartment.E1[compartment.E1 >= grouping_center],
                np.linspace(0, q_hi, N_GROUPS // 2 + 1) * 100,
            )[1:],
        )
    else:
        # bin on the whole range
        group_E1_bounds = (
            np.percentile(
                compartment.E1,
                np.linspace(q_low, q_hi, N_GROUPS) * 100,
            ),
        )

    # Calculate expected interactions for chromosome
    diag_expected = expected_by_diag_avg(clr)

    # Make a function that do one chromasome at a time
    def chrom_saddle(
        chro, clr, compartment, diag_expected, group_E1_bounds, min_diag_dist
    ):
        bins, offset = clr.bins().fetch(chro), clr.offset(chro)
        bins["ID"] = bins.index.values
        # bin of interest
        bins_pr = pr.PyRanges(
            bins.rename(
                columns={
                    old: new
                    for old, new in zip(
                        bins.columns[:3], ["Chromosome", "Start", "End"]
                    )
                }
            )
        )
        compartment_pr = pr.PyRanges(
            compartment.rename(
                columns={
                    old: new
                    for old, new in zip(
                        compartment.columns[:3], ["Chromosome", "Start", "End"]
                    )
                }
            )
        )
        boi = bins_pr.join(compartment_pr).drop_duplicate_positions()
        boi, _ = cooltools.saddle.digitize_track(
            group_E1_bounds, track=(boi.df, "E1")
        )
        boi_idx = boi.ID.values - offset

        # ob/exp matrix
        mat = (
            clr.matrix(balance=True, sparse=True)
            .fetch(chro)
            .astype(float)
            .tocoo()
        )
        # ob / exp
        chro_diag_values = diag_expected[diag_expected.region == chro][
            "balanced.avg"
        ].values.copy()
        chro_diag_values[chro_diag_values == 0] = np.nan
        mat.data /= chro_diag_values[np.abs(mat.row - mat.col)]
        mat = mat.toarray()

        # subtract matrix
        mat = mat[boi_idx, :][:, boi_idx]
        dist = np.repeat(boi.Start.values, len(boi)).reshape(
            (len(boi), len(boi))
        )
        dist = np.abs(dist - dist.T)
        mat[dist <= min_diag_dist] = np.nan
        # process to saddle data
        S, C = np.zeros((n_groups, n_groups)), np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            idx_i = boi["E1.d"].values == i
            mat_ = mat[idx_i, :]
            for j in range(n_groups):
                idx_j = boi["E1.d"].values == j
                valid_idx = np.logical_not(np.isnan(mat_[:, idx_j]))
                C[i, j] += np.sum(valid_idx)
                S[i, j] += np.nansum((mat_[:, idx_j]))

        return S, C

    with mp.Pool(parallel) as pool:
        S_C_results = pool.map(
            lambda chro: chrom_saddle(
                chro,
                clr,
                compartment,
                diag_expected,
                group_E1_bounds,
                min_diag_dist,
            ),
            compartment.chrom.unique(),
        )

    S, C = np.zeros((n_groups, n_groups)), np.zeros((n_groups, n_groups))
    for S_, C_ in S_C_results:
        S += S_
        C += C_

    stats = []
    for S_, C_ in S_C_results:
        saddle_data = S_ / C_  # [1:-1, 1:-1]
        a_b = np.nanmean(
            saddle_data[0:n_bins_for_stats, n_groups - n_bins_for_stats :]
        )
        a_a = np.nanmean(
            saddle_data[
                n_groups - n_bins_for_stats :, n_groups - n_bins_for_stats :
            ]
        )
        b_b = np.nanmean(saddle_data[0:n_bins_for_stats, 0:n_bins_for_stats])
        stats.append((a_a * b_b) / (a_b * a_b))

    return S, C, group_E1_bounds, np.array(stats)


def compartmental_segregation(
    clr,
    cluster_ref,
    use_expected=True,
    low_pc_percentile=20,
    high_pc_percentile=80,
    min_dist=2_000_000,
    max_dist=None,
    fill_min_signal_at="final",
):
    """
    For each bin, calculate its average interaction to each bin cluster
    """
    # make sure clr and bin match
    bin_cluster = clr.bins()[:].merge(
        cluster_ref, on=["chrom", "start", "end"], how="inner"
    )
    bin_size = clr.binsize

    if use_expected:
        # build expected
        chromsizes, chromosomes = clr.chromsizes, clr.chromnames
        supports = [(chrom, 0, chromsizes[chrom]) for chrom in chromosomes]
        # Calculate expected interactions for chromosome
        with mp.Pool(8) as pool:
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

        # Calculate average number of interactions per diagonal
        expected["balanced.avg"] = (
            expected["balanced.sum"] / expected["n_valid"]
        )
        expected["region"] = [
            region.split(":")[0] for region in expected.region.values
        ]

    chroms = clr.chromnames
    n_clusters = bin_cluster.Cluster.unique()
    cluster_inter = {
        i: np.ones(len(bin_cluster)) * np.nan
        for i in n_clusters
        if not np.isnan(i)
    }
    for chrom in chroms:
        mat = clr.matrix(balance=True).fetch(chrom)
        n = mat.shape[0]
        if use_expected:
            # ob / exp
            chrom_expected = expected[expected.region == chrom]
            diag_values = chrom_expected["balanced.avg"].values
        else:
            diag_values = np.ones(n)

        # mask low and high dist to nan
        diag_values[: min_dist // bin_size] = np.nan
        if max_dist is not None:
            diag_values[max_dist // bin_size :] = np.nan

        diag_mask = np.array([np.roll(diag_values, k) for k in range(n)])
        diag_mask = np.triu(diag_mask) + np.triu(diag_mask, 1).T
        obexp = mat / diag_mask
        obexp[np.isinf(obexp)] = np.nan
        if fill_min_signal_at == "expected":
            obexp[obexp == 0] = np.nanmin(obexp[obexp != 0])

        chro_bin_mask = (bin_cluster.chrom == chrom).values
        chro_bin_cluster = bin_cluster[chro_bin_mask]
        min_sig = np.inf
        for i in cluster_inter.keys():
            mask = chro_bin_cluster.Cluster == i
            if i == 1:
                mask = mask & (
                    chro_bin_cluster.E1
                    >= np.nanpercentile(chro_bin_cluster.E1, high_pc_percentile)
                )
            else:
                mask = mask & (
                    chro_bin_cluster.E1
                    <= np.nanpercentile(chro_bin_cluster.E1, low_pc_percentile)
                )
            col_masked_mat = obexp[:, mask.values]
            mean_sig = np.nanmean(col_masked_mat, axis=1)
            if fill_min_signal_at == "final":
                temp_min_sig = np.nanmin(mean_sig[mean_sig != 0])
                if temp_min_sig < min_sig:
                    min_sig = temp_min_sig
            cluster_inter[i][chro_bin_mask] = mean_sig

        if fill_min_signal_at == "final":
            # mask 0 with lowest signal to avoid inf
            for i in cluster_inter.keys():
                chrom_cluster_inter = cluster_inter[i][chro_bin_mask]
                chrom_cluster_inter[chrom_cluster_inter == 0] = min_sig
                cluster_inter[i][chro_bin_mask] = chrom_cluster_inter

    for cluster, inter in cluster_inter.items():
        bin_cluster[f"with_{cluster}"] = inter

    return bin_cluster


def bin_avg_interaction(
    clr,
    ignore_diag=2,
    max_dist=1_000_000,
    use_expected=True,
    fill_min_signal_at="final",
):
    """
    For each bin, calculate its average interaction to each bin cluster
    """
    # make sure clr and bin match
    bins = clr.bins()[:].copy()
    chroms = clr.chromnames
    bin_size = clr.binsize
    avg_interactions = np.ones(len(bins)) * np.nan

    if use_expected:
        # build expected
        chromsizes, chromosomes = clr.chromsizes, clr.chromnames
        supports = [(chrom, 0, chromsizes[chrom]) for chrom in chromosomes]
        # Calculate expected interactions for chromosome
        with mp.Pool(8) as pool:
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

        # Calculate average number of interactions per diagonal
        expected["balanced.avg"] = (
            expected["balanced.sum"] / expected["n_valid"]
        )
        expected["region"] = [
            region.split(":")[0] for region in expected.region.values
        ]

    for chrom in chroms:
        mat = clr.matrix(balance=True).fetch(chrom)
        n = mat.shape[0]
        if use_expected:
            # ob / exp
            chrom_expected = expected[expected.region == chrom]
            diag_values = chrom_expected["balanced.avg"].values
        else:
            diag_values = np.ones(n)

        # mask low and high dist to nan
        diag_values[:ignore_diag] = np.nan
        diag_values[max_dist // bin_size :] = np.nan

        diag_mask = np.array([np.roll(diag_values, k) for k in range(n)])
        diag_mask = np.triu(diag_mask) + np.triu(diag_mask, 1).T
        obexp = mat / diag_mask
        obexp[np.isinf(obexp)] = np.nan
        if fill_min_signal_at == "expected":
            obexp[obexp == 0] = np.nanmin(obexp[obexp != 0])

        chro_bin_mask = (bins.chrom == chrom).values
        mean_sig = np.nanmean(obexp, axis=1)
        if fill_min_signal_at == "final":
            mean_sig[mean_sig == 0] = np.nanmin(mean_sig[mean_sig != 0])

        avg_interactions[chro_bin_mask] = mean_sig

    bins["AvgInteraction"] = avg_interactions

    return bins


def local_vs_distal(
    clr,
    ignore_diag=2,
    local_dist=300_000,
):
    """
    For each bin, calculate its average interaction to each bin cluster
    """
    # make sure clr and bin match
    bins = clr.bins()[:].copy()
    chroms = clr.chromnames
    bin_size = clr.binsize
    ldr = np.ones(len(bins)) * np.nan

    for chrom in chroms:
        mat = clr.matrix(balance=True).fetch(chrom)
        n = mat.shape[0]
        diag_values = np.ones(n)

        diag_values[:ignore_diag] = np.nan
        diag_mask = np.array([np.roll(diag_values, k) for k in range(n)])
        sum_counts = np.nansum(mat * diag_mask, axis=1)

        diag_values[local_dist // bin_size :] = np.nan
        diag_mask = np.array([np.roll(diag_values, k) for k in range(n)])
        local_counts = np.nansum(mat * diag_mask, axis=1)

        chro_bin_mask = (bins.chrom == chrom).values
        ldr[chro_bin_mask] = local_counts / (sum_counts - local_counts)

    bins["LDR"] = ldr

    return bins
