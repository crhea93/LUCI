import math
import os
import statistics as stats
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# --------------------------------------- WVT ALGORITHM ITSELF BELOW THIS ---------------------------------------#


def plot_Bins(Bins, x_min, x_max, y_min, y_max, StN_Target, file_dir, filename):
    if not os.path.exists(file_dir + "/histograms/"):
        os.mkdir(file_dir + "/histograms/")
    fig = plt.figure()
    # fig.set_size_inches(7, 6.5)
    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
    N = len(Bins)
    StN_list = []
    SNR_list = []
    bin_nums = []
    StN_list = [bin.StN[0] for bin in Bins]
    StN_list = [v for v in StN_list if not (math.isinf(v) or math.isnan(v))]
    if len(StN_list) < 2:
        # Nothing to summarise -- fewer than two bins have a finite S/N. This is a diagnostic plot,
        # so skip it rather than letting statistics.stdev raise and take the whole run down.
        logger.warning("plot_Bins: only %d bin(s) with finite S/N; skipping the %s plots", len(StN_list), filename)
        return None
    median_StN = np.median(StN_list)
    stand_dev = stats.stdev(StN_list)
    mini_pallete = [
        "mediumspringgreen",
        "salmon",
        "cyan",
        "orchid",
        "yellow",
        "blue",
        "red",
        "magenta",
        "black",
        "white",
    ]
    # The bin mosaic used to be drawn as one matplotlib Rectangle per pixel, then saved. On a
    # full-field run that is hundreds of thousands of patches (twice -- for bin_acc and final),
    # minutes of pure plotting. Rasterise instead: paint each pixel's palette index into an array
    # and imshow it once. Same picture -- ten cycling colours keyed on bin number -- in one draw.
    mosaic = np.full((y_max - y_min, x_max - x_min), -1, dtype=int)
    for binNumber, bin in enumerate(Bins):
        bin_nums.append(bin.bin_number)
        SNR_list.append(bin.StN[0] / median_StN)
        for pixel in bin.pixels:
            xi = pixel.pix_x - x_min
            yi = pixel.pix_y - y_min
            if 0 <= yi < mosaic.shape[0] and 0 <= xi < mosaic.shape[1]:
                mosaic[yi, xi] = binNumber % 10
    cmap = mcolors.ListedColormap(mini_pallete)
    ax.imshow(
        np.ma.masked_less(mosaic, 0),
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap=cmap,
        vmin=0,
        vmax=9,
        interpolation="nearest",
        aspect="auto",
    )
    SNR_list = [v for v in SNR_list if not (math.isinf(v) or math.isnan(v))]
    centroids_x = [Bins[i].centroidx[0] for i in range(len(Bins))]
    centroids_y = [Bins[i].centroidy[0] for i in range(len(Bins))]
    ax.scatter(centroids_x, centroids_y, marker="+", c="black")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bin Mosaic")
    plt.savefig(file_dir + "/" + filename + ".png")
    plt.clf()
    plt.hist(StN_list)
    plt.xlim((median_StN - 3 * stand_dev, median_StN + 3 * stand_dev))
    plt.ylabel("Number of Bins")
    plt.xlabel("Signal-to-Noise")
    plt.title("Signal-to-Noise per Bin")
    plt.savefig(file_dir + "/histograms/" + filename + ".png")
    plt.clf()
    SNR_std = stats.stdev(SNR_list)
    SNR_med = np.median(SNR_list)
    SNR_nel = len(SNR_list)
    n_el = SNR_nel  # Just change this
    plt.scatter(np.arange(n_el), SNR_list, marker="+", color="salmon", label="Data Points")
    plt.plot(np.arange(n_el), [SNR_med for i in range(n_el)], linestyle="solid", color="forestgreen", label="Median")
    plt.plot(np.arange(n_el), [SNR_med + SNR_std for i in range(n_el)], linestyle="--", color="black")
    plt.plot(np.arange(n_el), [SNR_med - SNR_std for i in range(n_el)], linestyle="--", color="black", label="sigma")
    plt.title("Signal to Noise Ratio")
    plt.ylim((min(SNR_list), max(SNR_list)))
    plt.xlabel("Bin Number")
    plt.ylabel("Signal-to-Noise Normalized by Median Value")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fancybox=True, shadow=True)
    plt.savefig(file_dir + "/" + filename + "_scatter.png", bbox_inches="tight")
    plt.clf()
    return None


class Bin:
    """
    Bin Class Information
    Everything in here should be self-explanatory... if not let me know and I
    will most happily comment it! :)
    """

    def __init__(self, number):
        self.bin_number = number
        self.pixels = []
        self.pixel_neighbors = []
        self.centroidx = [0]  # So I can pass by value
        self.centroidy = [0]
        self.centroidx_prev = [0]
        self.centroidy_prev = [0]
        self.StN = [0]
        self.StN_prev = [0]
        self.StN_sum = 0.0  # Sum of the member pixels' S/N; self.StN is this over sqrt(N)
        # self.Signal = [0]
        # self.Noise = [0]
        self.Area = [0]
        self.Area_prev = [0]
        self.scale_length = [0]
        self.scale_length_prev = [0]
        self.successful = False
        self.WVT_successful = False
        self.avail_reassign = True  # Can a pixel be reassigned to you?

    def _recalculate_StN(self):
        """
        S/N of the binned spectrum: sum of the member signals over the combined noise.

        Summing N pixels sums their signal but only adds their noise in quadrature, so for
        roughly uniform noise the bin's S/N goes as sum(S/N) / sqrt(N) -- i.e. sqrt(N) times a
        single pixel, which is the whole point of binning (Cappellari & Copin 2003).

        This used to be a plain running sum, sum(S/N), which overstates the bin by sqrt(N). Bin
        accretion stops once the bin reaches 0.75 * the target, so bins were being declared
        finished at sqrt(N) times too little signal: on an M86 SN4 cube, 3-4 pixel bins claimed
        S/N 20 while actually carrying about 8.
        """
        n = len(self.pixels)
        self.StN[0] = self.StN_sum / np.sqrt(n) if n else 0.0

    def add_pixel(self, Pixel):
        self.pixels.append(Pixel)
        self.StN_sum += Pixel.StN
        self._recalculate_StN()
        for neigh in Pixel.neighbors:
            if neigh not in self.pixel_neighbors:
                self.pixel_neighbors.append(neigh)

    def clear_pixels(self):
        self.pixels = []
        self.StN_sum = 0.0
        self.StN[0] = 0
        self.successful = False
        self.WVT_successful = False
        self.avail_reassign = True

    def remove_pixel(self, Pixel):
        self.pixels.remove(Pixel)
        self.StN_sum -= Pixel.StN
        self._recalculate_StN()

    def success(self):
        self.successful = True

    def WVT_success(self):
        self.WVT_successful = True

    def availabilty(self):
        self.avail_reassign = False

    def update_StN_prev(self):
        self.StN_prev[0] = self.StN[0]

    def CalcCentroid(self):
        self.centroidx_prev[0] = self.centroidx[0]
        self.centroidy_prev[0] = self.centroidy[0]
        self.centroidx[0] = 0
        self.centroidy[0] = 0
        n_cent = 0
        for pixel in self.pixels:
            self.centroidx[0] += pixel.pix_x
            self.centroidy[0] += pixel.pix_y
            n_cent += 1
        self.centroidx[0] *= 1 / n_cent
        self.centroidy[0] *= 1 / n_cent

    def CalcArea(self, pixel_length):
        self.Area_prev[0] = self.Area[0]
        self.Area[0] = 0
        self.Area[0] += pixel_length**2 * len(self.pixels)

    def CalcScaleLength(self, StN_Target):
        self.scale_length_prev[0] = self.scale_length[0]
        self.scale_length[0] = 0
        self.scale_length[0] += np.sqrt((self.Area[0] / np.pi) * (StN_Target / self.StN[0]))


class Pixel:
    def __init__(self, number, pix_x, pix_y, SNR):
        self.pix_number = number
        self.pix_x = pix_x
        self.pix_y = pix_y
        # self.Signal = signal
        # self.Noise = noise
        self.StN = SNR
        # if self.Noise != 0:
        #    self.StN = self.Signal/np.sqrt(self.Noise)
        self.neighbors = []
        self.neighbors_x = []
        self.neighbors_y = []
        self.assigned_to_bin = False
        self.assigned_bin = None

    def add_neighbor(self, pixel, x_pixel, y_pixel):
        self.neighbors.append(pixel)
        self.neighbors_x.append(x_pixel)
        self.neighbors_y.append(y_pixel)

    def add_to_bin(self, bin):
        self.assigned_to_bin = True
        self.assigned_bin = bin

    def clear_bin(self):
        self.assigned_to_bin = False
        self.assigned_bin = None


def read_in(SNR_map, snr_floor=None):
    """
    Build the pixel list from an S/N map, optionally skipping pixels with no signal.

    Args:
        SNR_map: Path to the S/N map FITS written by `create_snr_map`
        snr_floor: Skip pixels at or below this S/N (default None, keep every pixel).
            Most of a SITELLE field is blank sky. Binning it is not just wasted work -- accretion
            and refinement both scale with the pixel count -- it is wasted on bins that never reach
            the target and are dropped by the `0.5 * StN_Target` test anyway. Pixels left out keep
            `BIN_MAP_UNASSIGNED` and are simply not fitted, so the maps stay full-field.

    Return:
        (Pixels, x_min, x_max, y_min, y_max), the bounds being the full extent of the map
    """
    # Collect Pixel Data
    logger.info(os.getcwd())
    hdu_list = fits.open(SNR_map, memmap=True)
    counts = hdu_list[0].data
    y_len = counts.shape[0]
    x_len = counts.shape[1]
    hdu_list.close()
    x_min = 0
    y_min = 0
    x_max = x_len
    y_max = y_len
    Pixels = []
    pixel_count = 0
    for col in range(int(x_len)):
        for row in range(int(y_len)):
            SNR = counts[row][col]
            if snr_floor is not None and not (SNR > snr_floor):  # `not >` also drops NaN
                continue
            Pixels.append(Pixel(pixel_count, x_min + col, y_min + row, SNR))  # Bottom Left Corner!
            pixel_count += 1
    if snr_floor is not None:
        logger.info(
            "We have %d Pixels above S/N %.3g (of %d in the map).", pixel_count, snr_floor, x_len * y_len
        )
        if pixel_count == 0:
            raise ValueError(
                "No pixel exceeds the S/N floor of %.3g, so there is nothing to bin. The S/N map "
                "peaks at %.3g." % (snr_floor, np.nanmax(counts))
            )
    else:
        logger.info("We have " + str(pixel_count) + " Pixels.")
    return Pixels, x_min, x_max, y_min, y_max


def Nearest_Neighbors(pixel_list):
    xvals = []
    yvals = []
    num_neigh = 9
    for pixel in pixel_list:
        xvals.append(pixel.pix_x)
        yvals.append(pixel.pix_y)
    X = np.column_stack((xvals, yvals))
    nbrs = NearestNeighbors(n_neighbors=num_neigh, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)
    pix_num = 0
    for pixel in pixel_list:
        for j in range(num_neigh - 1):
            if distances[pix_num][j + 1] == 1:
                index = indices[pix_num][j + 1]
                pixel.add_neighbor(pixel_list[index], xvals[index], yvals[index])
            else:
                pass  # not adjacent
        pix_num += 1
    return None


def dist(p1x, p1y, p2x, p2y):
    # math.sqrt rather than np.sqrt: both are correctly rounded so the value is identical, but this
    # is called with scalars tens of millions of times per accretion run and np.sqrt pays for
    # building a numpy scalar on every one of them.
    return math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)


def _linear_nearest_unassigned(p1x, p1y, all_pixels):
    """Exact fallback: lowest-index unassigned pixel at the minimum distance, by a full scan."""
    best, best_d, best_idx = None, math.inf, -1
    for idx, p in enumerate(all_pixels):
        if not p.assigned_to_bin:
            d = (p.pix_x - p1x) ** 2 + (p.pix_y - p1y) ** 2
            if d < best_d:
                best_d, best, best_idx = d, p, idx
    return best


def _nearest_unassigned(p1x, p1y, all_pixels, tree, k_cap=8192):
    """
    The unassigned pixel nearest to (p1x, p1y), matching a full linear scan exactly.

    This replaces a scan over every unassigned pixel. The tree holds *all* pixels -- it cannot have
    points deleted -- so we ask for the k nearest, keep those still unassigned, and among them take
    the one at the smallest distance, breaking ties by lowest pixel index. That tie rule is not
    cosmetic: early bin centroids are simple fractions, so several grid pixels sit at *exactly* the
    same distance, and the original scan (`min` then `list.index`) always took the lowest-indexed of
    them. Breaking a tie differently reseeds a bin, and because seeds chain, one difference reshapes
    the whole tessellation.

    Distances are recomputed from the pixel coordinates -- identical arithmetic to the original -- so
    the comparison and the tie test are bit-for-bit the same; the tree's own distances are used only
    to prove the candidate set is complete (no unqueried pixel is as close), widening k until it is.

    Args:
        p1x, p1y: The point to search from (a bin centroid)
        all_pixels: Pixel list, indexed the same as the tree's points
        tree: cKDTree over every pixel's (x, y)
        k_cap: Widen the query up to this many neighbours before falling back to a linear scan

    Return:
        The nearest unassigned Pixel, or None if every pixel is assigned
    """
    n = len(all_pixels)
    k = min(32, n)
    while True:
        kk = min(k, n)
        dists, idxs = tree.query([p1x, p1y], k=kk)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        best, best_d, best_idx = None, math.inf, -1
        for idx in idxs:
            idx = int(idx)
            if not all_pixels[idx].assigned_to_bin:
                p = all_pixels[idx]
                d = (p.pix_x - p1x) ** 2 + (p.pix_y - p1y) ** 2
                # Nearest wins; equal distances go to the lower index, explicitly -- the tree does
                # not return tied points in index order, so this cannot be left to encounter order.
                if d < best_d or (d == best_d and idx < best_idx):
                    best_d, best, best_idx = d, p, idx
        # Complete only if the winner is strictly nearer than the farthest pixel we queried: then no
        # unqueried pixel (all at >= dists[-1]) can match or beat it. Otherwise widen k.
        if best is not None and (kk >= n or math.sqrt(best_d) < dists[-1]):
            return best
        if kk >= n or k >= k_cap:
            break
        k = min(k * 4, n)
    # Neighbourhood exhausted or the cap hit; a single linear scan is exact and O(n).
    return _linear_nearest_unassigned(p1x, p1y, all_pixels)


def closest_node(Bin_current, all_pixels, tree):
    closest_val = 1e16  # just some big number
    p1x = Bin_current.centroidx[0]
    p1y = Bin_current.centroidy[0]
    closest_pixel = None
    for pix_neigh in Bin_current.pixel_neighbors:
        if pix_neigh.assigned_to_bin == False:  # Dont bother with already assigned pixels!
            p2x = pix_neigh.pix_x
            p2y = pix_neigh.pix_y
            new_dist = dist(p1x, p1y, p2x, p2y)
            if new_dist < closest_val:
                closest_val = new_dist
                closest_pixel = pix_neigh
    if closest_val == 1e16:
        # Every neighbour is taken, so fall back to the nearest unassigned pixel anywhere. A linear
        # scan here was ~half the cost of accretion (it fires roughly once per bin, over all
        # unassigned pixels); the tree turns it into a k-nearest query. See `_nearest_unassigned`.
        closest_pixel = _nearest_unassigned(p1x, p1y, all_pixels, tree)
    return closest_pixel


def adjacency(current_bin, closest_node):
    if closest_node in current_bin.pixel_neighbors:
        return True
    else:
        return False


def Roundness(current_bin, closest_pixel, pixel_length):
    pixel_list = current_bin.pixels
    n = len(pixel_list) + 1  # Including new
    rad_equiv = np.sqrt(n / np.pi) * (pixel_length)
    xvals = [pixel.pix_x for pixel in current_bin.pixels]
    yvals = [pixel.pix_y for pixel in current_bin.pixels]
    cen_x_new = (sum(xvals) + closest_pixel.pix_x) / n
    cen_y_new = (sum(yvals) + closest_pixel.pix_y) / n
    dists = []
    for i in range(n - 1):
        dists.append(np.sqrt((xvals[i] - cen_x_new) ** 2 + (yvals[i] - cen_y_new) ** 2))
    dists.append(np.sqrt((closest_pixel.pix_x - cen_x_new) ** 2 + (closest_pixel.pix_y - cen_y_new) ** 2))
    # maximum distance between the centroid of the bin and any of the bin pixels. pix_x/pix_y are
    # pixel indices, so this is in pixels and has to be put on the same footing as rad_equiv,
    # which carries a factor of pixel_length. Without this the comparison is dimensionally
    # inconsistent and roundness scales as 1/pixel_length: at the default pixel_size=0.436 even
    # two adjacent pixels score 0.437, above the 0.3 default criterion, so no bin could ever
    # accrete a second pixel. That only went unnoticed because bin accretion stops early when a
    # single pixel already exceeds 0.75 * the S/N target, which is the case for bright cubes.
    # Scaling both sides makes roundness what it should be -- a scale-free shape measure.
    rad_max = max(dists) * pixel_length
    roundness = rad_max / rad_equiv - 1.0
    return roundness


def Potential_SN(Current_bin, closest_pix):
    Current_bin.add_pixel(closest_pix)
    new_StN = Current_bin.StN[0]
    Current_bin.remove_pixel(closest_pix)
    return new_StN


def Bin_data(Bins, missing_pixels, min_x, min_y, output_directory, filename):
    Bins.sort(key=lambda bin: bin.bin_number)
    file = open(output_directory + "/" + filename + ".txt", "w+")
    file.write(
        "This text file contains information necessary for chandra to bin the pixels appropriately for image.fits \n"
    )
    file.write("pixel_x pixel_y bin \n")
    file2 = open(output_directory + "/" + filename + "_bins.txt", "w+")
    file2.write("Bin data for paraview script to plot Weighted Voronoi Diagram \n")
    file2.write("centroidx centroidy weight \n")
    binCount = 0
    for bin in Bins:
        for pixel in bin.pixels:
            file.write(str(pixel.pix_x - min_x) + " " + str(pixel.pix_y - min_y) + " " + str(binCount) + " \n")
        file2.write(str(bin.centroidx[0]) + " " + str(bin.centroidy[0]) + " " + str(bin.scale_length[0]) + " \n")
        binCount += 1
    file.close()
    file2.close()
    return None


def bin_num_pixel(binList, currentPixel):
    bin_number_interest = None
    for binNumber in range(len(binList)):
        if currentPixel in binList[binNumber].pixels:
            bin_number_interest = binNumber
            break
    return bin_number_interest


def assigned_missing_pixels(pixels):
    # get all pixels in domain but not already assigned
    pos_x = [pix.pix_x for pix in pixels]
    pos_y = [pix.pix_y for pix in pixels]
    already_pixel = {}
    for i in range(len(pos_x)):
        already_pixel[(pos_x[i], pos_y[i])] = True
    x_range = [i for i in range(min(pos_x), max(pos_x) + 1)]
    y_range = [i for i in range(min(pos_y), max(pos_y) + 1)]
    Pixels_unbinned = []
    unbinned_num = max([pix.pix_number for pix in pixels]) + 1
    for x in x_range:
        for y in y_range:
            if (x, y) not in already_pixel.keys():
                new_pixel = Pixel(unbinned_num, x, y, 0)
                Pixels_unbinned.append(new_pixel)
                unbinned_num += 1
            else:
                pass  # Pixel already binned
    logger.info("We have " + str(len(Pixels_unbinned)) + " unbinned pixels")
    return Pixels_unbinned


def reassign_pixels(bin, bins_successful, sucessful_centroids):
    if not sucessful_centroids:
        logger.warning("reassign_pixels called with no successful bins; %d pixels left unassigned", len(bin.pixels))
        return None
    # Centroids as an array once, rather than a fresh Python list of distances per candidate per
    # pixel. Rejected candidates are masked out instead of deleted, which keeps the tie-breaking:
    # deleting preserved the relative order of what remained, so the first minimum was always the
    # lowest surviving index, and that is what `np.argmin` over the masked distances returns.
    centroids = np.asarray(sucessful_centroids, dtype=float)
    for pixel in bin.pixels:
        pixel.clear_bin()
        alive = np.ones(len(bins_successful), dtype=bool)
        while pixel.assigned_to_bin == False:
            if not alive.any():
                # No candidate bin left. `pass` here spun forever: the loop only
                # exits once the pixel is assigned, so a raise every iteration
                # meant an infinite loop rather than an error (B24).
                logger.warning("no bin available for pixel (%s, %s)", pixel.pix_x, pixel.pix_y)
                break
            dx = centroids[:, 0] - pixel.pix_x
            dy = centroids[:, 1] - pixel.pix_y
            distances = dx * dx + dy * dy  # Squared: monotonic, so the minimiser is unchanged
            distances[~alive] = np.inf
            closest_bin_index = int(np.argmin(distances))
            closest_bin = bins_successful[closest_bin_index]
            if closest_bin.availabilty == False and alive.sum() > 1:
                alive[closest_bin_index] = False
            else:
                pixel.add_to_bin(closest_bin)
                closest_bin.add_pixel(pixel)
    return None


def Bin_Acc(Pixels, pixel_length, StN_Target, roundness_crit):
    # step 1:setup list of bin objects
    logger.info("Starting Bin Accretion Algorithm")
    # Membership is tracked by each pixel's `assigned_to_bin` flag plus a running count, not by
    # removing from a list. The old `unassigned_pixels = Pixels[:]` with a `.remove()` per pixel was
    # O(n) per removal -- O(n^2) over a full run, minutes on a 200k-pixel field. The k-d tree serves
    # `closest_node`'s fallback (see `_nearest_unassigned`); it is built once here.
    n_unassigned = len(Pixels)
    tree = cKDTree(np.array([(p.pix_x, p.pix_y) for p in Pixels], dtype=float))
    binCount = 0
    Bin_list = []
    bins_successful = []
    criteria_a = False
    criteria_b = False
    criteria_c = False
    StN_list_pixels = [pixel.StN for pixel in Pixels]
    max_StN_ind = StN_list_pixels.index(max(StN_list_pixels))
    max_StN_pix = Pixels[max_StN_ind]
    Current_bin = Bin(binCount)
    Bin_list.append(Current_bin)
    Current_bin.add_pixel(max_StN_pix)
    Current_bin.CalcCentroid()
    max_StN_pix.add_to_bin(binCount)
    n_unassigned -= 1
    closest_pix = None
    closest_not_in_pix = None
    while n_unassigned != 0:
        criteria_a = True
        criteria_b = True
        criteria_c = True
        while (criteria_a == True and criteria_b == True and criteria_c == True) and n_unassigned != 0:
            closest_pix = closest_node(Current_bin, Pixels, tree)
            criteria_a = adjacency(Current_bin, closest_pix)
            criteria_b = True if (Roundness(Current_bin, closest_pix, pixel_length) < roundness_crit) else False
            criteria_c = True if (Potential_SN(Current_bin, closest_pix) < 0.75 * StN_Target) else False
            if criteria_a == True and criteria_b == True and criteria_c == True:
                Current_bin.add_pixel(closest_pix)
                Current_bin.CalcCentroid()
                closest_pix.add_to_bin(binCount)
                n_unassigned -= 1
            else:
                closest_not_in_pix = closest_pix
        if Current_bin.StN[0] > 0.5 * StN_Target:
            Current_bin.success()
            bins_successful.append(Current_bin)
        # else:
        # print(Current_bin.StN[0])
        if n_unassigned == 0:
            break  # All pixels assigned so dont create a new bin. that would be silly
        else:
            binCount += 1
            Current_bin = Bin(binCount)
            Bin_list.append(Current_bin)
            Current_bin.add_pixel(closest_not_in_pix)
            Current_bin.CalcCentroid()
            closest_not_in_pix.add_to_bin(binCount)
            n_unassigned -= 1
    for bin in bins_successful:
        bin.CalcCentroid()
        bin.CalcArea(pixel_length)
        bin.CalcScaleLength(StN_Target)
    sucessful_centroids = [(bin.centroidx[0], bin.centroidy[0]) for bin in bins_successful]
    for bin in Bin_list:
        if bin.successful == False:
            reassign_pixels(bin, bins_successful, sucessful_centroids)
            bin.clear_pixels()
        else:
            bin.CalcCentroid()
            bin.CalcArea(pixel_length)
            bin.CalcScaleLength(StN_Target)
    for bin in bins_successful:
        bin.update_StN_prev()
    logger.info("Completed Bin Accretion Algorithm")
    logger.info("There are a total of " + str(len(bins_successful) + 1) + " bins!")
    return bins_successful


def converged_met(Bins, ToL):
    True_count = 0
    for bin in Bins:
        StN_old = bin.StN_prev[0]
        StN_new = bin.StN[0]
        if abs(StN_new - StN_old) / StN_old < ToL:
            True_count += 1
        else:
            pass
    if True_count / len(Bins) > 0.9:
        return True
    else:
        return False


def _brute_force_nearest(pix_x, pix_y, cent_x, cent_y, scale_length, chunk_bytes=128 << 20):
    """
    Every pixel against every bin, in chunks bounded by `chunk_bytes`.

    Distances are compared squared, against squared scale lengths: sqrt is monotonic so the
    minimiser is unchanged, and ties go to the lowest bin index either way -- `np.argmin` and
    `list.index(min(...))` both take the first minimum.
    """
    n_bins = int(cent_x.size)
    best = np.empty(pix_x.size, dtype=np.intp)
    scale_sq = scale_length**2
    rows = max(1, int(chunk_bytes // (8 * n_bins)))
    for start in range(0, pix_x.size, rows):
        stop = min(start + rows, pix_x.size)
        dx = pix_x[start:stop, None] - cent_x[None, :]
        dy = pix_y[start:stop, None] - cent_y[None, :]
        dx *= dx
        dy *= dy
        dx += dy
        dx /= scale_sq[None, :]
        best[start:stop] = np.argmin(dx, axis=1)
    return best


def _best_of_candidates(metric, indices):
    """
    Winning bin per pixel, resolving ties to the lowest bin index.

    The candidate columns come back from the tree ordered by distance, not by bin number, so an
    `argmin` over them would break ties by proximity. The brute-force original broke them by index,
    and the tessellation depends on it, so pick the smallest bin number among the exact minima.
    """
    best_val = metric.min(axis=1, keepdims=True)
    tied = np.where(metric == best_val, indices, np.iinfo(np.intp).max)
    return tied.min(axis=1), best_val[:, 0]


def nearest_weighted_bin(
    pix_x, pix_y, cent_x, cent_y, scale_length, chunk_bytes=128 << 20, max_candidates=64
):
    """
    For each pixel, the index of the bin minimising distance / scale_length.

    This is the assignment step of the weighted Voronoi tessellation, and it is the single most
    expensive thing in a WVT run. Comparing every pixel against every bin is O(N_pixels * N_bins):
    for a full SITELLE field (4.2 million pixels, ~341,000 bins) that is 1.4e12 distance
    evaluations per iteration and `WVT` runs up to five of them -- about 18 days as a Python list
    comprehension, and still ~8 hours vectorised, because vectorising only buys a constant factor.

    So the candidates are narrowed with a k-d tree first. A bin can only win if it is *spatially*
    close, and the weights bound how far "close" reaches: having found some candidate scoring
    `U = d/s`, any bin beating it needs `d_i < U * s_i <= U * max(s)`. So if the k-th nearest
    centroid already lies beyond `U * max(s)`, no unqueried bin can win and the answer from those k
    candidates is exact. Where that bound is not met the search widens, and any pixel still
    unresolved at `max_candidates` falls back to brute force. The result is therefore identical to
    comparing against every bin, not an approximation.

    Args:
        pix_x: Pixel x coordinates
        pix_y: Pixel y coordinates
        cent_x: Bin centroid x from the previous iteration
        cent_y: Bin centroid y from the previous iteration
        scale_length: Bin scale lengths from the previous iteration
        chunk_bytes: Memory ceiling for one pixel-by-bin distance block (default 128 MB)
        max_candidates: Widest tree query before falling back to brute force (default 64)

    Return:
        Array of bin indices, one per pixel
    """
    pix_x = np.asarray(pix_x, dtype=float)
    pix_y = np.asarray(pix_y, dtype=float)
    cent_x = np.asarray(cent_x, dtype=float)
    cent_y = np.asarray(cent_y, dtype=float)
    scale_length = np.asarray(scale_length, dtype=float)

    n_bins = int(cent_x.size)
    if n_bins == 0:
        raise ValueError("Cannot assign pixels to bins: no bins were given.")
    # Below this the tree costs more than it saves, and the bound needs a usable max scale length
    scale_max = scale_length.max() if n_bins else 0.0
    if n_bins <= 16 or pix_x.size <= 16 or not np.isfinite(scale_max) or scale_max <= 0:
        return _brute_force_nearest(pix_x, pix_y, cent_x, cent_y, scale_length, chunk_bytes)

    tree = cKDTree(np.column_stack((cent_x, cent_y)))
    best = np.empty(pix_x.size, dtype=np.intp)
    # One chunk of pixels holds (chunk x max_candidates) floats, so this is far smaller than the
    # brute-force block; keep the same ceiling so the caller has one knob.
    rows = max(1, int(chunk_bytes // (8 * min(max_candidates, n_bins))))
    for start in range(0, pix_x.size, rows):
        stop = min(start + rows, pix_x.size)
        points = np.column_stack((pix_x[start:stop], pix_y[start:stop]))
        pending = np.arange(stop - start)
        k = min(8, n_bins)
        while pending.size:
            dists, indices = tree.query(points[pending], k=k)
            dists = np.atleast_2d(dists)
            indices = np.atleast_2d(indices)
            with np.errstate(divide="ignore", invalid="ignore"):
                # A zero scale length scores inf and simply never wins, as in the original
                metric = dists / scale_length[indices]
            winner, best_val = _best_of_candidates(metric, indices)
            best[start + pending] = winner
            if k >= n_bins:
                break  # Queried every bin, so this is already exact
            # Exact wherever the k-th neighbour is already too far to be beaten
            settled = best_val * scale_max <= dists[:, -1]
            pending = pending[~settled]
            if not pending.size:
                break
            if k >= max_candidates:
                # Rare: widen no further, just do these few against every bin
                idx = start + pending
                best[idx] = _brute_force_nearest(
                    pix_x[idx], pix_y[idx], cent_x, cent_y, scale_length, chunk_bytes
                )
                break
            k = min(k * 4, n_bins)
    return best


def Rebin_Pixels(binList, pixel_list, pixel_length, StN_Target):
    for bin in binList:
        bin.clear_pixels()
    WVT_successful_bins = []
    n_pix, n_bins = len(pixel_list), len(binList)
    closest_indices = nearest_weighted_bin(
        np.fromiter((p.pix_x for p in pixel_list), dtype=float, count=n_pix),
        np.fromiter((p.pix_y for p in pixel_list), dtype=float, count=n_pix),
        np.fromiter((b.centroidx_prev[0] for b in binList), dtype=float, count=n_bins),
        np.fromiter((b.centroidy_prev[0] for b in binList), dtype=float, count=n_bins),
        np.fromiter((b.scale_length_prev[0] for b in binList), dtype=float, count=n_bins),
    )
    # Computed in bulk, but applied in the original pixel order on purpose: the order in which bins
    # first receive a pixel is what fixes the order of WVT_successful_bins, and `reassign_pixels`
    # walks that list to rehome the pixels of bins that never took one.
    for pixel, bin_index in zip(pixel_list, closest_indices):
        pixel.clear_bin()
        closest_bin = binList[bin_index]
        pixel.add_to_bin(closest_bin)
        closest_bin.add_pixel(pixel)
        if closest_bin.StN[0] > 0 and closest_bin.WVT_successful == False:
            WVT_successful_bins.append(closest_bin)
            closest_bin.WVT_success()
    sucessful_centroids = [(bin.centroidx[0], bin.centroidy[0]) for bin in WVT_successful_bins]
    for bin in binList:
        if bin not in WVT_successful_bins:
            reassign_pixels(bin, WVT_successful_bins, sucessful_centroids)
    for bin in WVT_successful_bins:
        bin.CalcCentroid()
        bin.CalcArea(pixel_length)
        bin.CalcScaleLength(StN_Target)
    return WVT_successful_bins


def WVT(Bin_list_init, Pixel_Full, StN_Target, ToL, pixel_length, image_dir):
    logger.info("Beginning WVT")
    Bin_list_prev = Bin_list_init[:]
    converged = False
    its_to_conv = 0
    if not os.path.exists(image_dir + "/histograms/"):
        os.mkdir(image_dir + "/histograms/")
    while converged == False and its_to_conv < 5:
        logger.info("We are on step " + str(its_to_conv + 1))
        bins_with_SN = Rebin_Pixels(Bin_list_prev, Pixel_Full, pixel_length, StN_Target)[:]
        converged = converged_met(bins_with_SN, ToL)
        Bin_list_prev = bins_with_SN[:]
        for bin in bins_with_SN:
            bin.update_StN_prev()
        bin_SN_List = [bin.StN[0] for bin in bins_with_SN]
        bin_SN_List = [v for v in bin_SN_List if not (math.isinf(v) or math.isnan(v))]
        plt.hist(bin_SN_List)
        plt.ylabel("Number of Bins")
        plt.xlabel("Signal-to-Noise")
        its_to_conv += 1
        plt.xlim(0, StN_Target * 2)
        # plt.patch.set_facecolor('white')
        plt.savefig(image_dir + "/histograms/iteration_" + str(its_to_conv) + ".png")
        plt.clf()
    if its_to_conv < 5:
        logger.info("Completed WVT in " + str(its_to_conv) + " step(s)!")
    else:
        logger.info("Stopped WVT algorithm after 5 steps.")
    logger.info("There are a total of " + str(len(bins_with_SN) + 1) + " bins!")
    return bins_with_SN


# ---------------------------------------------------------------------------
# Cube-level entry points, moved off the Luci god-class. Each takes the cube.
# ---------------------------------------------------------------------------

import glob  # noqa: E402
import time  # noqa: E402

from astropy.nddata import Cutout2D  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from luci.engine.maps import FitMaps  # noqa: E402
from luci.engine.runner import deep_image_cutout  # noqa: E402
from luci.log import get_logger

logger = get_logger(__name__)


#: Label used in the bin map for a pixel no bin claimed. It has to be distinguishable from a bin
#: number: the map used to start at 0, which is a real bin, so every unassigned pixel was silently
#: absorbed into bin 0 and fitted along with it.
BIN_MAP_UNASSIGNED = -1

#: Directory (under the cube's output dir) holding the bin map.
BIN_DIR = "Numpy_Voronoi_Bins"
BIN_MAP_NAME = "bin_map.npy"


def save_bin_map(output_dir, bin_map):
    """
    Write the bin label map, replacing any per-bin masks from an older run.

    One int32 label per pixel, rather than one full-field boolean mask per bin. The old scheme wrote
    `cube.shape[:2]` booleans for every bin -- 4.2 MB on a SITELLE field, to record the dozen pixels
    that bin actually holds -- so an 8,992-bin run cost 36 GB and a full-field run at ~341,000 bins
    would have needed about 1.4 TB. The label map is 17 MB whatever the bin count.

    Args:
        output_dir: The cube's output directory
        bin_map: int32 array of bin numbers, `BIN_MAP_UNASSIGNED` where no bin claimed the pixel

    Return:
        Path the map was written to
    """
    bin_dir = os.path.join(output_dir, BIN_DIR)
    os.makedirs(bin_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(bin_dir, "bool_bin_map_*.npy")):
        os.remove(stale)
    path = os.path.join(bin_dir, BIN_MAP_NAME)
    np.save(path, bin_map.astype(np.int32))
    return path


def load_bin_regions(output_dir, cube_shape=None):
    """
    Pixel indices of every bin, as a list of (xs, ys) index arrays.

    Reads the label map written by `save_bin_map`. If a run predating the label map is found instead
    -- a directory of `bool_bin_map_*.npy` full-field masks -- those are read in their original
    numeric order so old output stays fittable.

    Args:
        output_dir: The cube's output directory
        cube_shape: (dimx, dimy) used to check the map matches the cube (default None, no check)

    Return:
        List of (xs, ys) arrays, one per bin, indexed by bin number
    """
    bin_dir = os.path.join(output_dir, BIN_DIR)
    path = os.path.join(bin_dir, BIN_MAP_NAME)
    if not os.path.exists(path):
        legacy = glob.glob(os.path.join(bin_dir, "bool_bin_map_*.npy"))
        if not legacy:
            raise FileNotFoundError(
                "No bin map at %s and no legacy bool_bin_map_*.npy beside it. Run create_wvt first." % path
            )
        logger.info("Reading %d per-bin masks from a run predating the bin map", len(legacy))
        legacy.sort(key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
        return [tuple(np.where(np.load(p))) for p in legacy]

    bin_map = np.load(path)
    if cube_shape is not None and tuple(bin_map.shape) != tuple(cube_shape[:2]):
        raise ValueError(
            "Bin map is %s but the cube is %s. It belongs to a different cube or region -- rerun "
            "create_wvt." % (bin_map.shape, tuple(cube_shape[:2]))
        )
    # Group the pixels by label in one pass rather than scanning the whole map once per bin, which
    # is what made the old per-bin masks quadratic in the number of bins.
    flat = bin_map.ravel()
    positions = np.flatnonzero(flat >= 0)
    labels = flat[positions]
    order = np.argsort(labels, kind="stable")
    positions, labels = positions[order], labels[order]
    n_bins = int(labels[-1]) + 1 if labels.size else 0
    edges = np.searchsorted(labels, np.arange(n_bins + 1))
    return [
        np.unravel_index(positions[edges[b] : edges[b + 1]], bin_map.shape) for b in range(n_bins)
    ]


def create_wvt(
    cube,
    x_min_init,
    x_max_init,
    y_min_init,
    y_max_init,
    pixel_size,
    stn_target,
    roundness_crit,
    ToL,
    n_threads,
    snr_floor=None,
    snr_method=1,
    snr_percentile=None,
    snr_precomputed=False,
):
    """
    Written by Benjamin Vigneron.

    Functionality to create a weighted Voronoi tesselation map from a region and according to
    arguments passed by the user. It creates a folder containing all the Voronoi bins that can
    then be used for the fitting procedure.

    Args:
        x_min_init: Minimal X value
        x_max_init: Maximal X value
        y_min_init: Minimal Y value
        y_max_init: Maximal Y value
        pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
        stn_target: Signal-to-Noise target value for the Voronoi bins.
        roundness_crit: Roundness criteria for the pixel accretion into bins
        ToL: Convergence tolerance parameter for the SNR of the bins
        n_threads: Number of threads to use
        snr_floor: Only bin pixels above this S/N (default None, bin everything). Most of a SITELLE
            field is blank sky whose bins never reach the target and get dropped anyway, and both
            accretion and refinement scale with the pixel count, so a floor is the cheapest way to
            make a full-field run tractable. Note that for SN4 the S/N flux window (15150-15300
            cm-1) spans Halpha and both NII lines, so this is a cut on the whole complex.
        snr_method: Which `create_snr_map` estimator to use (default 1, as before; 2 is
            flux-in-window over the noise standard deviation)
        snr_percentile: Set `snr_floor` to this percentile of the S/N map instead of an absolute
            value (default None). The useful cut depends on the estimator and the cube -- method 1
            is not a calibrated S/N -- so asking for "the brightest 15% of the field"
            (snr_percentile=85) is more portable than guessing a number.

    Return:
        The int32 bin label map, as written to disk
    """
    logger.info("#----------------WVT Algorithm----------------#")
    Pixels = []
    if snr_precomputed:
        # Bin on an S/N map already written to SNR/<object>_SNR.fits by the caller -- e.g. a
        # continuum-subtracted single-line tracer -- rather than one of create_snr_map's built-in
        # estimators. Used when the built-in windows trace the wrong thing (on M86 the wide SN4
        # window tracks the stellar continuum, not the Halpha/NII line).
        logger.info("#----------------Using precomputed SNR Map--------------#")
    else:
        logger.info("#----------------Creating SNR Map--------------#")
        cube.create_snr_map(x_min_init, x_max_init, y_min_init, y_max_init, method=snr_method, n_threads=n_threads)
    logger.info("#----------------Algorithm Part 1----------------#")
    start = time.time()
    snr_path = cube.output_dir + "/SNR/" + cube.object_name + "_SNR.fits"
    if snr_percentile is not None:
        if snr_floor is not None:
            raise ValueError("Give either snr_floor or snr_percentile, not both.")
        snr_values = fits.open(snr_path)[0].data
        snr_floor = float(np.nanpercentile(snr_values, snr_percentile))
        logger.info(
            "S/N percentile %.4g of the map is %.4g; binning the pixels above it.", snr_percentile, snr_floor
        )
    Pixels, x_min, x_max, y_min, y_max = read_in(snr_path, snr_floor=snr_floor)
    Nearest_Neighbors(Pixels)
    Init_bins = Bin_Acc(Pixels, pixel_size, stn_target, roundness_crit)
    plot_Bins(Init_bins, x_min, x_max, y_min, y_max, stn_target, cube.output_dir, "bin_acc")
    total_time = time.gmtime(float(time.time() - start))
    logger.info("The first part of the algorithm took %s." % (time.strftime("%H:%M:%S", total_time)))
    logger.info("#----------------Algorithm Part 2----------------#")
    Final_Bins = WVT(Init_bins, Pixels, stn_target, ToL, pixel_size, cube.output_dir)
    logger.info("#----------------Algorithm Complete--------------#")
    plot_Bins(Final_Bins, x_min, x_max, y_min, y_max, stn_target, cube.output_dir, "final")
    Bin_data(Final_Bins, Pixels, x_min, y_min, cube.output_dir, "WVT_data")
    logger.info("#----------------Bin Mapping--------------#")
    # `Bin_data` above sorted Final_Bins by bin_number and labelled the pixels it wrote to
    # WVT_data.txt by position in that sorted order, so enumerating it here gives the same labels
    # without parsing millions of lines of text back in.
    bin_map = np.full(cube.cube_final.shape[:2], BIN_MAP_UNASSIGNED, dtype=np.int32)
    for bin_num, bin_ in enumerate(Final_Bins):
        for pixel in bin_.pixels:
            bin_map[x_min_init + pixel.pix_x, y_min_init + pixel.pix_y] = bin_num
    save_bin_map(cube.output_dir, bin_map)
    n_assigned = int((bin_map >= 0).sum())
    logger.info("Mapped %d pixels into %d bins", n_assigned, len(Final_Bins))
    return bin_map


def fit_wvt(
    cube,
    lines,
    fit_function,
    vel_rel,
    sigma_rel,
    bkg=None,
    absorp=None,
    bayes_bool=False,
    uncertainty_bool=False,
    n_threads=1,
    initial_values=[False],
    n_stoch=1,
    stn_target=10,
):
    """
    Function that takes the wvt mapping created using `cube.create_wvt()` and fits the bins.
    Written by Benjamin Vigneron

    Args:
        lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        fit_function: Fitting function to use (e.x. 'gaussian')
        vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
        sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
        bkg: Background Spectrum (1D numpy array; default None)
        absorp: Stellar absorption template on the full spectral axis, as returned by
            `cube.build_absorption_template` (1D numpy array; default None)
        bayes_bool: Boolean to determine whether or not to run Bayesian analysis
        uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
        n_threads: Number of threads to use
        initial_values: Initial values of velocity and broadening for fitting specific lines (must be list)
        n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
        stn_target: Target signal to noise ratio (default 10)

    Return:
        Velocity, Broadening and Flux arrays (2d). Also return amplitudes array (3D) and header for saving
        figure.
    """
    x_min = 0
    x_max = cube.cube_final.shape[0]
    y_min = 0
    y_max = cube.cube_final.shape[1]
    maps = FitMaps.allocate(x_max - x_min, y_max - y_min, len(lines))
    if len(initial_values) == 2:
        # Obtain initial condition maps from files
        vel_init = fits.open(initial_values[0])[0].data
        broad_init = fits.open(initial_values[1])[0].data
    ct = 0
    if not os.path.exists(cube.output_dir + "/" + cube.object_name + "_deep.fits"):
        cube.create_deep_image()
    wcs = WCS(cube.header, naxis=2)
    cutout = Cutout2D(
        fits.open(cube.output_dir + "/" + cube.object_name + "_deep.fits")[0].data,
        position=((x_max + x_min) / 2, (y_max + y_min) / 2),
        size=(x_max - x_min, y_max - y_min),
        wcs=wcs,
    )
    regions = [(xs, ys) for xs, ys in load_bin_regions(cube.output_dir, cube.cube_final.shape) if xs.size]
    logger.info("Fitting %d bins on %d workers", len(regions), n_threads)
    # The cube-level constants every fit needs. Hoisted so they are pickled once per task rather than
    # read off `cube` inside the worker -- a worker that touched `cube` would be sent its ~8 GB of
    # data, which is why `fit_extracted_spectrum` is a staticmethod over plain arrays.
    fit_constants = dict(
        wavenumbers_syn=cube.wavenumbers_syn,
        delta_x=cube.hdr_dict["STEP"],
        n_steps=cube.step_nb,
        zpd_index=cube.zpd_index,
        filter_name=cube.hdr_dict["FILTER"],
        ML_bool=cube.ML_bool,
        mdn=cube.mdn,
        resolution=cube.resolution,
        Luci_path=cube.Luci_path,
        bayes_bool=bayes_bool,
        uncertainty_bool=uncertainty_bool,
        n_stoch=n_stoch,
    )

    def initial_conditions_for(xs, ys):
        # TODO: PASS INITIAL CONDITIONS
        if False not in initial_values:  # If initial conditions were passed
            return [vel_init[xs[0], ys[0]], broad_init[xs[0], ys[0]]]
        return [False]

    # Extract in the parent, fit in the workers. Extraction needs the cube; fitting needs only a few
    # kB per bin, so this is the split that lets the fit fan out at all. Chunked so the extracted
    # spectra of a whole field (~81,000 bins x 469 channels x 3 arrays) never sit in memory at once,
    # and wrapped in one `Parallel` context so the pool -- and each worker's cached ONNX predictor --
    # is reused across chunks instead of respawning per chunk.
    chunk_size = max(n_threads * 8, 64)
    with Parallel(n_jobs=n_threads) as parallel:
        for start in tqdm(range(0, len(regions), chunk_size)):
            chunk = regions[start : start + chunk_size]
            extracted = [cube.extract_region_for_fit(r, bkg=bkg, absorp=absorp) for r in chunk]
            results = parallel(
                delayed(cube.fit_extracted_spectrum)(
                    sky,
                    axis,
                    trans_filter,
                    theta,
                    fit_function,
                    lines,
                    vel_rel,
                    sigma_rel,
                    initial_values=initial_conditions_for(xs, ys),
                    **fit_constants,
                )
                for (xs, ys), (sky, axis, trans_filter, theta) in zip(chunk, extracted)
            )
            for (xs, ys), bin_fit_dict in zip(chunk, results):
                # The maps are shaped (n_y, n_x) and indexed [y, x] -- that is what `FitMaps.scatter`
                # does for the fit_cube path. This used to index them [x, y], which transposed every
                # WVT map and, on a cube whose y extent exceeds its x extent (SITELLE is 2048 x
                # 2064), raised IndexError as soon as a bin reached y >= 2048.
                _scatter_bin(maps, xs, ys, bin_fit_dict)
    maps.save(
        cube.output_dir, cube.object_name, lines, cutout.wcs.to_header(), binning=1, suffix="_wvt_%i" % stn_target
    )
    return maps.velocities, maps.broadenings, maps.fluxes, maps.chi2, cutout.wcs.to_header()


def _scatter_bin(maps, xs, ys, bin_fit_dict):
    """Write one bin's fit into every map, at all of its pixels."""
    maps.amplitudes[ys, xs] = bin_fit_dict["amplitudes"]
    maps.fluxes[ys, xs] = bin_fit_dict["fluxes"]
    maps.flux_errors[ys, xs] = bin_fit_dict["flux_errors"]
    maps.broadenings[ys, xs] = bin_fit_dict["sigmas"]
    maps.broadenings_errors[ys, xs] = bin_fit_dict["sigmas_errors"]
    maps.chi2[ys, xs] = bin_fit_dict["chi2"]
    maps.continuum[ys, xs] = bin_fit_dict["continuum"]
    # Wrote continuum_error into continuum_fits, so the continuum map
    # held the error and the error map stayed zero (B19).
    maps.continuum_error[ys, xs] = bin_fit_dict["continuum_error"]
    maps.velocities[ys, xs] = bin_fit_dict["velocities"]
    maps.velocities_errors[ys, xs] = bin_fit_dict["vels_errors"]


def wvt_fit_region(
    cube,
    x_min_init,
    x_max_init,
    y_min_init,
    y_max_init,
    lines,
    fit_function,
    vel_rel,
    sigma_rel,
    stn_target,
    pixel_size=0.436,
    roundness_crit=0.3,
    ToL=1e-2,
    bkg=None,
    absorp=None,
    bayes_bool=False,
    uncertainty_bool=False,
    n_threads=1,
    n_stoch=1,
    initial_values=[False],
    snr_floor=None,
    snr_method=1,
    snr_percentile=None,
    snr_precomputed=False,
):
    """
    Functionality to wrap-up the creation and fitting of weighted Voronoi bins.

    Args:
        x_min_init: Minimal X value
        x_max_init: Maximal X value
        y_min_init: Minimal Y value
        y_max_init: Maximal Y value
        lines: Lines to fit (e.x. ['Halpha', 'NII6583'])
        fit_function: Fitting function to use (e.x. 'gaussian')
        vel_rel: Constraints on Velocity/Position (must be list; e.x. [1, 2, 1])
        sigma_rel: Constraints on sigma (must be list; e.x. [1, 2, 1])
        stn_target: Signal-to-Noise target value for the Voronoi bins.
        pixel_size: Pixel size of the image. For SITELLE use pixel_size = 0.0000436.
        roundness_crit: Roundness criteria for the pixel accretion into bins
        ToL: Convergence tolerance parameter for the SNR of the bins
        bkg: Background Spectrum (1D numpy array; default None)
        absorp: Stellar absorption template on the full spectral axis, as returned by
            `cube.build_absorption_template` (1D numpy array; default None)
        bayes_bool: Boolean to determine whether or not to run Bayesian analysis
        uncertainty_bool: Boolean to determine whether or not to run the uncertainty analysis (default False)
        n_threads: Number of threads to use
        initial_values: Initial values of velocity and broadening for fitting specific lines (must be list;
        e.x. [velocity, broadening]; default [False])
        n_stoch: The number of stochastic runs -- set to 50 for fitting double components (default 1)
        snr_floor: Only bin and fit pixels above this S/N (default None, bin everything). Pixels
            below it are left unfitted rather than cropped, so the maps stay full-field. For SN4 the
            S/N flux window spans Halpha and both NII lines, so this cuts on the whole complex.
        snr_method: Which `create_snr_map` estimator to use (default 1)
        snr_percentile: Set the floor to this percentile of the S/N map rather than an absolute
            value (default None), e.x. 85 to bin the brightest 15% of the field

    Return:
        Velocity, Broadening and Flux arrays (2d). Also return amplitudes array (3D).
    """
    # Call create wvt function to create the WVT map and numpy files corresponding to each bin
    cube.create_wvt(
        x_min_init,
        x_max_init,
        y_min_init,
        y_max_init,
        pixel_size,
        stn_target,
        roundness_crit,
        ToL,
        n_threads,
        snr_floor=snr_floor,
        snr_method=snr_method,
        snr_percentile=snr_percentile,
        snr_precomputed=snr_precomputed,
    )
    logger.info("#----------------WVT Fitting--------------#")
    # Fit the bins
    velocities_fits, broadenings_fits, flux_fits, chi2_fits, header = cube.fit_wvt(
        lines,
        fit_function,
        vel_rel,
        sigma_rel,
        bkg=bkg,
        absorp=absorp,
        bayes_bool=bayes_bool,
        uncertainty_bool=uncertainty_bool,
        n_threads=n_threads,
        initial_values=initial_values,
        n_stoch=n_stoch,
        stn_target=stn_target,
    )
    # The maps `fit_wvt` wrote are already final. There used to be a pass here that reopened all
    # nine products per line, transposed each with `.T`, and rewrote them, which left the WVT maps in
    # (n_x, n_y) while every fit_cube product is (n_y, n_x) -- the two orientations disagreed, and
    # code that overlays them (`pick_bright_spots` in tools/reduce_M86_SN4.py tests for
    # `(dimy, dimx)`) was right for fit_cube and wrong for WVT. The transpose was also compensating
    # for the scatter in `fit_wvt` indexing [x, y] into a (n_y, n_x) array: the two cancelled on a
    # square region and raised IndexError on anything else. With the scatter fixed the transpose is
    # simply wrong, and the header written alongside was the same one either way.
    return None
