import numpy as np
import pandas as pd
import pyranges as pr


def region_coverage(
    regions: pr.PyRanges,
    frag_file,
    chunk_size=1_000_000,
    scale_to=1_000_000,
    fill_zeros=True,
):
    region_coverage = np.zeros(len(regions))
    depth = 0
    for chunk in pd.read_csv(
        frag_file,
        sep="\t",
        header=None,
        chunksize=chunk_size,
        names=["Chromosome", "Start", "End"],
    ):
        depth += len(chunk)
        chunk_pr = pr.PyRanges(chunk)
        chunk_region_coverage = (
            regions.coverage(chunk_pr, overlap_col="Coverage")
            .drop_duplicate_positions()
            .df
        )
        # sum to open count
        region_coverage += regions.df.merge(
            chunk_region_coverage, on=["Chromosome", "Start", "End"], how="left"
        ).Coverage.values

    # normalized by depth
    # scale_factor = scale_to / depth
    if scale_to != 0:
        region_coverage *= scale_to / depth
    if fill_zeros:
        region_coverage[region_coverage == 0] = np.min(
            region_coverage[region_coverage != 0]
        )

    return region_coverage


def region_depth_diff_agg_from_peaks(
    regions: pd.DataFrame,
    peaks,
    frag_files,
    suffix=None,
    count_scale=1_000_000,
    depthcol_prefix="Depth",
    diffcol_prefix="Diff",
    log2_transform=False,
    agg_method="median",
):
    """
    Diff is quantified by fold change to first sample in the `frag_files`.
    `log2_transform` defines whether transform diff columns before aggregation to sum diff stat of a region.
    """
    if suffix is None:
        suffix = [i for i in range(frag_files)]

    # diff against first frag_file
    depthcols = [f"{depthcol_prefix}_{sfx}" for sfx in suffix]
    diffcols = [f"{diffcol_prefix}_{sfx}" for sfx in suffix]

    # split regions by peaks
    region_pr = pr.PyRanges(
        regions.rename(
            columns={
                old: new
                for old, new in zip(
                    regions.columns[:3], ["Chromosome", "Start", "End"]
                )
            }
        )
    )
    region_pr = region_pr.insert(
        pd.Series(data=range(len(region_pr)), name="ID")
    )
    # count coverage per peak region
    region_depth_by_peaks = region_pr.intersect(peaks)
    for frag_file, prefix_ff in zip(frag_files, depthcols):
        region_coverage_by_peaks = region_coverage(
            region_depth_by_peaks, frag_file, scale_to=count_scale
        )
        region_depth_by_peaks = region_depth_by_peaks.insert(
            pd.Series(data=region_coverage_by_peaks, name=prefix_ff)
        )

    region_depth_by_peaks = region_depth_by_peaks.df

    # diff agains first frag file
    diffcols = diffcols[1:]
    ctl_depth = depthcols[0]
    for target_diffcol, target_depth in zip(diffcols, depthcols[1:]):
        if log2_transform:
            region_depth_by_peaks[target_diffcol] = np.log2(
                region_depth_by_peaks[target_depth]
                / region_depth_by_peaks[ctl_depth]
            )
        else:
            region_depth_by_peaks[target_diffcol] = (
                region_depth_by_peaks[target_depth]
                / region_depth_by_peaks[ctl_depth]
            )

    # combine signal for each region
    # weighted average signal change
    def apply_func(
        df, agg_method=agg_method, depthcols=depthcols, diffcols=diffcols
    ):
        agg_depth = {
            depthcol: np.nansum(df[depthcol]) for depthcol in depthcols
        }
        if agg_method == "median":
            agg_diff = {
                diffcol: np.nanmedian(df[diffcol]) for diffcol in diffcols
            }
        elif agg_method == "mean":
            agg_diff = {
                diffcol: np.nanmean(df[diffcol]) for diffcol in diffcols
            }
        elif agg_method == "weighted":
            weight = df[depthcols[0]] / np.nansum(df[depthcols[0]])
            agg_diff = {
                diffcol: np.nansum(weight * df[diffcol]) for diffcol in diffcols
            }

        return pd.Series(
            {
                "N_Peaks": len(df),
                **agg_depth,
                **agg_diff,
            }
        )

    region_by_peaks_combined = (
        region_depth_by_peaks.groupby("ID")[depthcols + diffcols]
        .apply(
            apply_func,
        )
        .reset_index()
    )

    return region_pr.df.merge(region_by_peaks_combined, on="ID", how="left")
