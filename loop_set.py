import cooler
import pandas as pd
import numpy as np
import pyranges as pr
from scipy.spatial import distance_matrix
from itertools import combinations
from typing import Dict
from juice2cool import mad_max_bin_mask


class LoopSet:
    def __init__(self, loops_bedpe_in, name=None, padding_size=0):
        # if pad anchors, take the center of anchor, and pad half padding size at both side
        if padding_size != 0:
            half_pad = padding_size // 2
            middle_i = (loops_bedpe_in.x1 + loops_bedpe_in.y1) // 2
            middle_j = (loops_bedpe_in.x2 + loops_bedpe_in.y2) // 2
            self.loops_bedpe = pd.DataFrame(
                {
                    "chr1": loops_bedpe_in.chr1,
                    "x1": middle_i - half_pad,
                    "y1": middle_i + half_pad,
                    "chr2": loops_bedpe_in.chr2,
                    "x2": middle_j - half_pad,
                    "y2": middle_j + half_pad,
                }
            )
        else:
            self.loops_bedpe = loops_bedpe_in

        if name is not None:
            self.loops_bedpe["Name"] = name

        # non duplicated anchors
        self.anchor_i = pr.PyRanges(
            self.loops_bedpe[["chr1", "x1", "y1"]].rename(
                columns={"chr1": "Chromosome", "x1": "Start", "y1": "End"}
            )
        ).drop_duplicate_positions()
        self.anchor_j = pr.PyRanges(
            self.loops_bedpe[["chr2", "x2", "y2"]].rename(
                columns={"chr2": "Chromosome", "x2": "Start", "y2": "End"}
            )
        ).drop_duplicate_positions()
        self.all_anchors = pr.PyRanges(
            pd.concat([self.anchor_i.df, self.anchor_j.df], ignore_index=True)
        ).drop_duplicate_positions()

    @classmethod
    def from_bedpe(cls, bedpe, name=None, padding_size=0):
        loops_bedpe_in = (
            pd.read_csv(bedpe, comment="#", header=None, sep="\t")
            .iloc[:, 0:6]
            .rename(
                columns={
                    i: v
                    for i, v in enumerate(
                        ["chr1", "x1", "y1", "chr2", "x2", "y2"]
                    )
                }
            )
        )
        return cls(loops_bedpe_in, name, padding_size)

    @classmethod
    def from_multiple_bedpe(cls, bedpe_list, name, padding_size=0):
        combined = pd.DataFrame()
        for bedpe in bedpe_list:
            loops_bedpe_in = (
                pd.read_csv(bedpe, comment="#", header=None, sep="\t")
                .iloc[:, 0:6]
                .rename(
                    columns={
                        i: v
                        for i, v in enumerate(
                            ["chr1", "x1", "y1", "chr2", "x2", "y2"]
                        )
                    }
                )
            )
            combined = pd.concat([combined, loops_bedpe_in]).reset_index(
                drop=True
            )

        return cls(combined, name, padding_size)

    def loop_coors(self, update=False):
        # drop annotated columns from loop list
        if update:
            self.loops_bedpe = self.loops_bedpe.iloc[:, :6]
            return None
        else:
            return self.loops_bedpe.iloc[:, :6]

    def anchor_coors(self):
        # drop annotated columns from anchor list
        self.all_anchors = self.all_anchors.drop()
        self.anchor_i = self.anchor_i.drop()
        self.anchor_j = self.anchor_j.drop()

    def annotate_loops(self, columns, update=True):
        """
        Join anchor info columns to loops. Require exactly coordinate match.
        """
        loops_bedpe = self.loops_bedpe.copy()
        left_anchor_cols = ["chr1", "x1", "y1"]
        right_anchor_cols = ["chr2", "x2", "y2"]

        loops_bedpe = loops_bedpe.merge(
            self.anchor_i.df[["Chromosome", "Start", "End"] + columns],
            left_on=left_anchor_cols,
            right_on=["Chromosome", "Start", "End"],
            how="left",
        )
        loops_bedpe = loops_bedpe.drop(columns=["Chromosome", "Start", "End"])
        loops_bedpe = loops_bedpe.rename(
            columns={coln: f"{coln}_I" for coln in columns}
        )

        loops_bedpe = loops_bedpe.merge(
            self.anchor_j.df[["Chromosome", "Start", "End"] + columns],
            left_on=right_anchor_cols,
            right_on=["Chromosome", "Start", "End"],
            how="left",
        )
        loops_bedpe = loops_bedpe.drop(columns=["Chromosome", "Start", "End"])
        loops_bedpe = loops_bedpe.rename(
            columns={coln: f"{coln}_J" for coln in columns}
        )

        if update:
            self.loops_bedpe = loops_bedpe
            return None
        else:
            return loops_bedpe

    def annotate_anchors(self, columns):
        self.anchor_i = pr.PyRanges(
            self.anchor_i.df.merge(
                self.all_anchors.df[["Chromosome", "Start", "End"] + columns],
                on=["Chromosome", "Start", "End"],
                how="left",
            )
        )
        self.anchor_j = pr.PyRanges(
            self.anchor_j.df.merge(
                self.all_anchors.df[["Chromosome", "Start", "End"] + columns],
                on=["Chromosome", "Start", "End"],
                how="left",
            )
        )

    def count_peak_at_anchors(
        self, peak_gr, name, extend_size=5000, update=True
    ):
        all_anchors = self.all_anchors.df.copy()
        # extend
        all_anchors["Start"] = all_anchors.Start - extend_size
        all_anchors["End"] = all_anchors.End + extend_size
        # count overlapping
        all_anchors = (
            pr.PyRanges(all_anchors)
            .coverage(
                peak_gr,
                overlap_col=f"Num{name}Peak",
                fraction_col=f"Frac{name}Peak",
            )
            .df
        )
        # shrink
        all_anchors["Start"] = all_anchors.Start + extend_size
        all_anchors["End"] = all_anchors.End - extend_size
        self.all_anchors = pr.PyRanges(all_anchors)
        self.annotate_anchors([f"Num{name}Peak", f"Frac{name}Peak"])

        # add peak count info to loops
        return self.annotate_loops([f"Num{name}Peak"], update)

    # def quantify_loops(self, clr, count_name="Count", update=True):
    #     bins = clr.bins()[:].copy()
    #     bins["BID"] = bins.index.values
    #     bins = pr.PyRanges(
    #         bins.rename(
    #             columns={"chrom": "Chromosome", "start": "Start", "end": "End"}
    #         )
    #     )

    #     anchor_i_bid = self.anchor_i.join(bins, how="left").df[
    #         ["Chromosome", "Start", "End", "BID"]
    #     ]
    #     anchor_j_bid = self.anchor_j.join(bins, how="left").df[
    #         ["Chromosome", "Start", "End", "BID"]
    #     ]

    #     left_anchor_cols = ["chr1", "x1", "y1"]
    #     right_anchor_cols = ["chr2", "x2", "y2"]
    #     pr_cols = ["Chromosome", "Start", "End"]
    #     loops_bedpe = pd.DataFrame()
    #     for chrom, chro_loops in self.loops_bedpe[
    #         left_anchor_cols + right_anchor_cols
    #     ].groupby("chr1"):
    #         chro_loops_expand = chro_loops.merge(
    #             anchor_i_bid,
    #             how="left",
    #             left_on=left_anchor_cols,
    #             right_on=pr_cols,
    #         ).drop(columns=pr_cols)
    #         chro_loops_expand = chro_loops_expand.merge(
    #             anchor_j_bid,
    #             how="left",
    #             left_on=right_anchor_cols,
    #             right_on=pr_cols,
    #             suffixes=("_I", "_J"),
    #         ).drop(columns=pr_cols)

    #         # count loop strength
    #         mat = clr.matrix(balance=True).fetch(chrom)
    #         offset = clr.offset(chrom)
    #         chro_loops_expand[count_name] = mat[
    #             (
    #                 tuple(chro_loops_expand.BID_I.values - offset),
    #                 tuple(chro_loops_expand.BID_J.values - offset),
    #             )
    #         ]
    #         chro_loops_agg = chro_loops_expand.groupby(
    #             by=left_anchor_cols + right_anchor_cols, as_index=False
    #         ).agg({count_name: "mean"})
    #         loops_bedpe = pd.concat([loops_bedpe, chro_loops_agg]).reset_index(
    #             drop=True
    #         )

    #     if update:
    #         self.loops_bedpe[count_name] = (
    #             self.loops_bedpe.merge(
    #                 loops_bedpe,
    #                 on=left_anchor_cols + right_anchor_cols,
    #                 how="left",
    #             )
    #         )[count_name].values
    #         return None
    #     else:
    #         return loops_bedpe

    def cluster_adjacent_anchors(self, res=10000, slack=0, update=True):
        all_anchors = self.all_anchors.cluster(count=True, slack=slack).df

        all_anchors = pd.concat(
            map(
                lambda df: pd.DataFrame(
                    {
                        "Chromosome": df.Chromosome,
                        "Start": df.Start,
                        "End": df.End,
                        "Cluster": df.Cluster,
                        "Cluster_start": np.ones(len(df), dtype=int)
                        * ((df.Start.min() + df.End.max()) // 2 - res // 2),
                        "Cluster_end": np.ones(len(df), dtype=int)
                        * ((df.Start.min() + df.End.max()) // 2 + res // 2),
                    }
                ),
                (grp[1] for grp in all_anchors.groupby("Cluster")),
            ),
            ignore_index=True,
        )
        self.all_anchors = pr.PyRanges(all_anchors)
        # i
        # anchor_i = self.anchor_i.cluster(count=True, slack=slack).df

        # anchor_i = pd.concat(
        #     map(
        #         lambda df: pd.DataFrame(
        #             {
        #                 "Chromosome": df.Chromosome,
        #                 "Start": df.Start,
        #                 "End": df.End,
        #                 "Cluster": df.Cluster,
        #                 "Cluster_start": np.ones(len(df), dtype=int)
        #                 * ((df.Start.min() + df.End.max()) // 2 - res // 2),
        #                 "Cluster_end": np.ones(len(df), dtype=int)
        #                 * ((df.Start.min() + df.End.max()) // 2 + res // 2),
        #             }
        #         ),
        #         (grp[1] for grp in anchor_i.groupby("Cluster")),
        #     ),
        #     ignore_index=True,
        # )
        # self.anchor_i = pr.PyRanges(anchor_i)
        # # j
        # anchor_j = self.anchor_j.cluster(count=True, slack=slack).df

        # anchor_j = pd.concat(
        #     map(
        #         lambda df: pd.DataFrame(
        #             {
        #                 "Chromosome": df.Chromosome,
        #                 "Start": df.Start,
        #                 "End": df.End,
        #                 "Cluster": df.Cluster,
        #                 "Cluster_start": np.ones(len(df), dtype=int)
        #                 * ((df.Start.min() + df.End.max()) // 2 - res // 2),
        #                 "Cluster_end": np.ones(len(df), dtype=int)
        #                 * ((df.Start.min() + df.End.max()) // 2 + res // 2),
        #             }
        #         ),
        #         (grp[1] for grp in anchor_j.groupby("Cluster")),
        #     ),
        #     ignore_index=True,
        # )
        # self.anchor_j = pr.PyRanges(anchor_j)

        # assign cluster information back to original anchors
        self.annotate_anchors(["Cluster", "Cluster_start", "Cluster_end"])
        return self.annotate_loops(
            ["Cluster", "Cluster_start", "Cluster_end"], update
        )

    def rounding_cluster_adjacent_anchors(
        self,
        bins: pr.PyRanges,
    ):
        """
        Find the bins that overlapping with anchors and merge cluster by the anchor bins.
        Handle case: two anchors do not directly overlapping but an genomic bin is overlapping with both of them.
        They will be merged and given same cluster ID.
        """
        # TODO: implement distance for merging

        # split anchors by bins
        anchors = self.all_anchors.drop()
        anchor_bins = bins.join(anchors, how="right")
        # cluster; continuous bins will be merge into a cluster
        anchor_bins = (
            anchor_bins.cluster()
            .df[["Chromosome", "Start_b", "End_b", "Cluster"]]
            .drop_duplicates(ignore_index=True)
        )
        anchor_bins = anchor_bins.rename(
            columns={"Start_b": "Start", "End_b": "End"}
        )
        self.all_anchors = pr.PyRanges(
            self.all_anchors.df.merge(
                anchor_bins,
                on=["Chromosome", "Start", "End"],
                how="left",
            )
        )

        self.annotate_anchors(["Cluster"])

    def filter_loops_by_mad_max(self, clr, mad_max=5):
        return LoopSet(
            _drop_loops_at_low_signal_bins(self.loops_bedpe, clr, mad_max)
        )

    def filter_loops_by_distance(self, short, long):
        return LoopSet(_distance_filter(self.loops_bedpe, short, long))


def _distance_filter(loop_df, short, long):
    d = loop_df.x2 - loop_df.x1
    lpbp = loop_df[(d >= short) & (d <= long)].reset_index(drop=True)
    return lpbp


def _drop_loops_at_low_signal_bins(
    loop_bedpe: pd.DataFrame, clr: cooler.Cooler, mad_max=5
):
    """
    Mask loops with anchors overlaping mad max bins
    """
    mask_bins = mad_max_bin_mask(clr, mad_max)
    mask_bins_pr = pr.PyRanges(
        mask_bins.rename(
            columns={
                old: new
                for old, new in zip(
                    mask_bins.columns[:3], ["Chromosome", "Start", "End"]
                )
            }
        )
    )
    left_unmask = pr.PyRanges(
        loop_bedpe.iloc[:, :3].rename(
            columns={
                old: new
                for old, new in zip(
                    loop_bedpe.columns[:3], ["Chromosome", "Start", "End"]
                )
            }
        )
    ).count_overlaps(mask_bins_pr, overlap_col="Count")
    left_unmask = left_unmask.df["Count"] == 0
    right_unmask = pr.PyRanges(
        loop_bedpe.iloc[:, 3:6].rename(
            columns={
                old: new
                for old, new in zip(
                    loop_bedpe.columns[3:6], ["Chromosome", "Start", "End"]
                )
            }
        )
    ).count_overlaps(mask_bins_pr, overlap_col="Count")
    right_unmask = right_unmask.df["Count"] == 0
    unmask = left_unmask & right_unmask
    print(f"{len(loop_bedpe)} loops were dropped to {unmask.sum()}")
    return loop_bedpe[unmask].reset_index(drop=True)


def combine_multi_res_loops(
    res_loops: Dict[int, str],
    bins,
    name=None,
    assume_dedup=True,
    mcool=None,
    **kwargs,
):
    """
    Keep the highest resolution loop as representative.
    If a loop cluster has multiple loop instances at the highest resolution, and loops have already been merged at that resolution,
    report all those instances as individual loop cluster.
    """
    # merge loops at different resolutions
    pooled_loops = pd.DataFrame()
    for res, loop_file in res_loops.items():
        if assume_dedup:
            res_final_loops = pd.read_csv(
                loop_file,
                sep="\t",
                header=None,
            ).iloc[:, :6]
            res_final_loops = res_final_loops.rename(
                columns={
                    i: coln
                    for i, coln in zip(
                        range(6), ["chr1", "x1", "y1", "chr2", "x2", "y2"]
                    )
                }
            )
        else:
            loopset = LoopSet.from_bedpe(loop_file)
            loopset.cluster_adjacent_anchors(res=res)
            # representative of loop cluster by centering
            clusters_of_merged_loops = loopset.loops_bedpe.groupby(
                by=["Cluster_I", "Cluster_J"], as_index=False
            )
            res_final_loops = clusters_of_merged_loops.apply(
                lambda df: pd.Series(
                    {
                        "chr1": df.chr1.values[0],
                        "x1": df.Cluster_start_I,
                        "y1": df.Cluster_end_I,
                        "chr2": df.chr2.values[0],
                        "x2": df.Cluster_start_J,
                        "y2": df.Cluster_end_J,
                    }
                )
            )

        if mcool is not None:
            clr_path = mcool + f"::resolutions/{res}"
            clr = cooler.Cooler(clr_path)
            res_final_loops = _drop_loops_at_low_signal_bins(
                res_final_loops, clr, **kwargs
            )

        res_final_loops["res"] = res
        pooled_loops = pd.concat(
            [pooled_loops, res_final_loops], ignore_index=True
        )

    # pool loops at different resolution and use rounding merge
    pooled_loopset = LoopSet(pooled_loops, name)
    pooled_loopset.rounding_cluster_adjacent_anchors(bins)
    pooled_merged_loops = pooled_loopset.annotate_loops(
        ["Cluster"], update=False
    )

    # pick the representative of cluster
    clustered_repr = pooled_merged_loops.groupby(
        by=["Cluster_I", "Cluster_J"]
    ).res.transform(lambda x: x == x.min())
    clustered_loops = pooled_merged_loops[clustered_repr].reset_index(drop=True)

    return clustered_loops


def merge_loopset(
    loopset_to_merge,
    clr_path=None,
    anchor_merge_dist=0,
    res=10_000,
    aggressive_collapse=False,
):
    # merge data
    merged_loops = pd.concat(
        [lps.loops_bedpe for lps in loopset_to_merge], ignore_index=True
    )
    if clr_path is not None:
        for clrp in clr_path:
            clr = cooler.Cooler(clrp)
            merged_loops = _drop_loops_at_low_signal_bins(
                merged_loops, clr, mad_max=5
            )
    names = [lps.loops_bedpe.Name.iloc[0] for lps in loopset_to_merge]
    merged_loopset = LoopSet(merged_loops, None)
    merged_loopset.cluster_adjacent_anchors(res=res, slack=anchor_merge_dist)
    for name in names:
        print(f"#{name}:", (merged_loopset.loops_bedpe.Name == name).sum())
    # membership of loops
    # representative of loop cluster by centering
    loop_grp = merged_loopset.loops_bedpe.groupby(
        by=["Cluster_I", "Cluster_J"], as_index=False, group_keys=False
    )
    merged_loops = loop_grp.apply(
        lambda df: _solve_cluster(df, aggressive_collapse, names)
    ).reset_index(drop=True)

    return LoopSet(merged_loops)


def _solve_cluster(df, collapse, names):
    if collapse:
        return pd.Series(
            {
                "chr1": df.chr1.values[0],
                "x1": df.Cluster_start_I.values[0],
                "y1": df.Cluster_end_I.values[0],
                "chr2": df.chr2.values[0],
                "x2": df.Cluster_start_J.values[0],
                "y2": df.Cluster_end_J.values[0],
                **{name: name in df.Name.unique() for name in names},
            }
        )
    else:
        # maintain the total number and not collapse in to one representation
        # by greedy assigning sets with smallest pairwise distance sum
        coors = np.array([(df.y1 + df.x1) // 2, (df.y2 + df.x2) // 2]).T
        source = df.Name.values
        indice = np.arange(len(df))
        splitted_set = _find_closest_set(coors, source, indice)
        sz = len(splitted_set)
        chr1 = [df.chr1.values[0]] * sz
        x1 = [df.Cluster_start_I.values[0]] * sz
        y1 = [df.Cluster_end_I.values[0]] * sz
        chr2 = [df.chr2.values[0]] * sz
        x2 = [df.Cluster_start_J.values[0]] * sz
        y2 = [df.Cluster_end_J.values[0]] * sz
        indicators = {
            name: [name in df.Name.iloc[i].unique() for i in splitted_set]
            for name in names
        }
        cnt = 0
        for _, i in indicators.items():
            cnt += np.array(i).sum()
        if cnt != len(source):
            ValueError(coors, cnt)

        return pd.DataFrame(
            {
                "chr1": chr1,
                "x1": x1,
                "y1": y1,
                "chr2": chr2,
                "x2": x2,
                "y2": y2,
                **indicators,
            }
        )


def _find_closest_set(coors, source, indice):
    pairs = []
    d = distance_matrix(coors, coors)
    while True:
        source_unq = np.unique(source)
        n_unq = len(source_unq)
        if n_unq == 1:
            # Terminated: no pairs find for the rest
            pairs.extend([np.array([i]) for i in indice])
            return pairs
        elif n_unq == 0:
            # Terminated: nothing left
            return pairs
        else:
            # find a set of len(source_unq) with smallest pairwise distance sum
            min_dist_sum, min_dist_set = 1e30, None
            for _comb_idx in combinations(np.arange(len(source)), n_unq):
                comb_idx = np.array(_comb_idx)
                if len(np.unique(source[comb_idx])) != n_unq:
                    continue
                # sum pairwise distance
                comb = indice[comb_idx]
                ds = np.array([d[x, y] for x, y in combinations(comb, 2)]).sum()
                if ds <= min_dist_sum:
                    min_dist_set = comb
                    min_dist_sum = ds
            # prepare for next iteration
            # remove found set (greedy)
            pairs.append(min_dist_set)
            keep_idx = np.where(~np.isin(indice, min_dist_set))[0]
            coors = coors[keep_idx, :]
            source = source[keep_idx]
            indice = indice[keep_idx]
