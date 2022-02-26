# %%
import pandas as pd
import numpy as np
import cooler
import subprocess
import logging

#%% add norm vector
def juicer_dump_norm(hic_file, chro, res, norm, juicer):
    dump = subprocess.Popen(
        f"java -jar {juicer} dump norm {norm} {hic_file} {chro} BP {res}",
        shell=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    return [float(v) for v in dump.stdout if not v.startswith(b"W")]


def juicer_dump_pixels(hic_file, chro, res, juicer):
    dump = subprocess.Popen(
        f"java -jar {juicer} dump observed NONE {hic_file} {chro} {chro} BP {res}",
        shell=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    return pd.read_csv(
        dump.stdout,
        sep="\t",
        comment=b"W",
        header=None,
        names=["bin1_coor", "bin2_coor", "count"],
    )


def make_pixel_id_table(pixel_coor, chro_bins):
    chro_bins = chro_bins.reset_index()
    return pd.DataFrame(
        {
            "bin1_id": pixel_coor.merge(
                chro_bins, left_on="bin1_coor", right_on="start", how="left"
            )["index"].values,
            "bin2_id": pixel_coor.merge(
                chro_bins, left_on="bin2_coor", right_on="start", how="left"
            )["index"].values,
            "count": pixel_coor["count"].values,
        }
    ).sort_values(by=["bin1_id", "bin2_id"], ignore_index=True)


def juice_to_cool(
    hic_file, cool_out, norm, chrsize_file, resolution, mode, juicer
):
    # genomic bins
    chrsize = pd.read_csv(chrsize_file, sep="\t", header=None)
    bins = cooler.util.binnify(chrsize.set_index(0)[1], resolution)
    choromosomes = bins.chrom.unique()
    # norm vector from juice
    norm_vec = np.zeros(len(bins)) * np.nan
    for chro in choromosomes:
        chro_norm_vec = juicer_dump_norm(
            hic_file, chro, resolution, norm, juicer
        )
        # handle non convergent case
        if len(chro_norm_vec) != 0:
            norm_vec[bins.chrom == chro] = chro_norm_vec

    norm_vec = 1 / norm_vec
    # create cool file without normalization
    chro_pixels = lambda chro: juicer_dump_pixels(
        hic_file, chro, resolution, juicer
    )
    pixel_generator = (
        make_pixel_id_table(chro_pixels(chro), bins[bins.chrom == chro])
        for chro in choromosomes
    )
    cooler.create_cooler(
        cool_out, bins, pixel_generator, ordered=True, mode=mode
    )
    # add juicer normalization
    clr = cooler.Cooler(cool_out)
    with clr.open("r+") as grp:
        h5opts = dict(compression="gzip", compression_opts=6)
        grp["bins"].create_dataset(norm, data=norm_vec, **h5opts)
    # use rescaled norm vec as default cool normlization ("weight")
    rescale_norm_vec = np.zeros(len(bins)) * np.nan
    for chro in choromosomes:
        logging.info(f"Rescaling {norm} for {chro} at {resolution} bp")
        # clr created matrix are symetric; but pixels are upper only
        chro_mat = clr.matrix(balance=norm, sparse=True).fetch(chro).tocsr()
        chro_mat.data[np.isnan(chro_mat.data)] = 0
        row_sum = chro_mat.sum(axis=1).A1
        scale_factor = row_sum[row_sum != 0].mean()
        chro_bin_index = bins.chrom == chro
        rescale_norm_vec[chro_bin_index] = norm_vec[chro_bin_index] / np.sqrt(
            scale_factor
        )
    with clr.open("r+") as grp:
        h5opts = dict(compression="gzip", compression_opts=6)
        grp["bins"].create_dataset("weight", data=rescale_norm_vec, **h5opts)

    logging.info(f"######Finished {cool_out}#####")


def mad_max_bin_mask(clr: cooler.Cooler, mad_max=5) -> pd.DataFrame:
    bins = clr.bins()[:]
    bin_signal = np.zeros(len(bins))
    for chro, chro_bins in bins.groupby("chrom"):
        raw_mat = clr.matrix(balance=False, sparse=True).fetch(chro).tocsr()
        # fill nan
        raw_mat.data[np.isnan(raw_mat.data)] = 0
        row_sum = raw_mat.sum(axis=1).A1.astype(float)
        row_sum /= np.median(row_sum[row_sum > 0])
        bin_signal[chro_bins.index.values] = row_sum

    log_nz_bin_signal = np.log(bin_signal[bin_signal > 0])
    med_log_nz_bin_signal = np.median(log_nz_bin_signal)
    dev_log_nz_bin_signal = np.median(
        np.abs(log_nz_bin_signal - med_log_nz_bin_signal)
    )
    cutoff = np.exp(med_log_nz_bin_signal - mad_max * dev_log_nz_bin_signal)
    return bins[bin_signal < cutoff]
