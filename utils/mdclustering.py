import numpy as np
from scipy.linalg import logm
from itertools import combinations
from itertools import product
from typing import Tuple, List
import warnings

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans

import pyemma.coordinates as coor
from pyemma import msm
from pyemma import config
from kneed import KneeLocator

config.show_progress_bars = False


class MethylClustering:
    def __init__(self, coords_arr, lag=10):
        self.coords_arr = np.array(coords_arr)
        self.lag = lag

        if len(coords_arr.shape) == 3 and coords_arr.shape[1] == 3:
            self.coords_arr = self._positional_average()

    def _positional_average(self):
        return np.average(self.coords_arr, axis=1)

    def cluster_coords(self, n_clusters=100, max_iter=50, stride=1):
        return coor.cluster_kmeans(self.coords_arr, n_clusters, max_iter=max_iter, stride=stride)

    def estimate_msm_its(self, cluster, lags=None, nits=5):
        if lags is None:
            lags = [1, 2, 3, 5, 7, 10, 20, 35, 30, 50]
        return msm.its(cluster.dtrajs, lags=lags, nits=nits)

    def estimate_hmm_its(self, clusters, lags=None, nits=2):
        if lags is None:
            lags = range(1, 11)
        return msm.timescales_hmsm(np.concatenate(clusters.dtrajs), nits, lags=lags)

    def estimate_msm(self, clusters=None, n_clusters=100, max_iter=50, stride=1, n_components=5):
        if clusters is None:
            clusters = self.cluster_coords(n_clusters=n_clusters, max_iter=max_iter, stride=stride)
        else:
            clusters = clusters
        msm_ = msm.estimate_markov_model(clusters.dtrajs, self.lag)
        eigvec = msm_.eigenvectors_right()
        dtrajs = np.concatenate(clusters.dtrajs)
        processes = eigvec[dtrajs, :n_components]
        its = self.estimate_msm_its(clusters, nits=n_components)
        return msm_, processes, its, clusters

    def estimate_hmm(self, clusters=None, n_clusters=100, max_iter=50, stride=1, n_components=3):
        if clusters is None:
            clusters = self.cluster_coords(n_clusters=n_clusters, max_iter=max_iter, stride=stride)
        else:
            clusters = clusters
        hmm_ = msm.estimate_hidden_markov_model(clusters.dtrajs, 2, lag=self.lag)
        dtrajs = np.concatenate(clusters.dtrajs)
        its = msm.timescales_hmsm(dtrajs, n_components, lags=self.lag, errors='bayes')
        return hmm_, None, its, clusters


    @staticmethod
    def get_indices(arr, n, max=False, min=False):
        if max:
            return arr.argsort()[-n:][::-1]
        elif min:
            return arr.argsort()[:n]
        else:
            raise ValueError("yo")


def _calculate_cluster_weights(weights1: np.ndarray, weights2: np.ndarray) -> np.ndarray:
    return np.outer(weights1, weights2)


def _draw_random_samples_from_cluster_idxs(idx1: np.ndarray, idx2: np.ndarray, len_traj: int) -> Tuple[
    Tuple[np.ndarray], Tuple[np.ndarray]]:
    n = int(np.sqrt(len_traj))

    rng = np.random.default_rng()
    rand_idx1 = rng.choice(idx1[0], n)
    rand_idx2 = rng.choice(idx2[0], n)
    indices = list(zip(*product(rand_idx1, rand_idx2)))

    return (np.array(indices[0]),), (np.array(indices[1]),)


def plot_3d_scatter(xs, ys, zs, labels):
    ax = plt.axes(projection='3d')
    ax.scatter3D(xs, ys, zs, c=labels, cmap=cm.get_cmap("viridis"))
    plt.show()


def plot_cluster_projections(coords_arr, resid, labels, means, cov, weights,
                             save=False, show=True):
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    if len(coords_arr.shape) == 3 and coords_arr.shape[1] == 3:
        coords_arr = MethylClustering(coords_arr).coords_arr

    # calculate the combinations
    xs, ys, zs = _get_coords(coords_arr)
    combs = _get_combinations(xs, ys, zs)

    # set the scene
    fig = plt.figure(figsize=(17, 6))
    outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)

    # determine if clusters are plotted
    # if labels is None:
    #     plot_clusters = False
    # else:
    # TODO: change
    plot_clusters = True

    # plot densities
    o_idx = 0
    for (i, j), (x, y) in combs:
        if plot_clusters:
            _plot_density_hist(fig, outer, o_idx, x, y, labels, means, cov, weights, i, j, plot_clusters=True)
        else:
            _plot_density_hist(fig, outer, o_idx, x, y, labels, means, cov, weights, i, j, plot_clusters=False)
        o_idx += 1

    fig.suptitle(f"{resid}", fontsize="xx-large")

    if save:
        fig.savefig(f"./out/clusters/{resid}_clustering.pdf")
        plt.close()
    if show:
        # fig.show()
        return fig


def _plot_density_hist(fig, outer, outer_idx, x, y, labels, means, cov, weights, ax1, ax2, plot_clusters=False):
    # set up the axes with gridspec
    grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[outer_idx], wspace=0.2, hspace=0.2)

    # set up main plot with histograms on x and y
    main_ax = plt.Subplot(fig, grid[:-1, 1:])
    y_hist = plt.Subplot(fig, grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = plt.Subplot(fig, grid[-1, 1:], yticklabels=[], sharex=main_ax)

    axis_dict = {0: "X",
                 1: "Y",
                 2: "Z"}

    main_ax.set_title("{}-{} coordinate projection".format(axis_dict[ax1], axis_dict[ax2]))

    # plot density map
    sns.kdeplot(x=x, y=y, cmap="rocket_r", shade=True, ax=main_ax)
    if plot_clusters:
        _plot_clusters(means, cov, weights, ax1, ax2, ax=main_ax)

    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)

    # plot histogram on the attached axes
    x_hist.hist(x, 40, histtype='stepfilled',
                orientation='vertical', color='gray')
    x_hist.invert_yaxis()

    y_hist.hist(y, 40, histtype='stepfilled',
                orientation='horizontal', color='gray')
    y_hist.invert_xaxis()

    fig.add_subplot(main_ax)
    fig.add_subplot(y_hist)
    fig.add_subplot(x_hist)


def _plot_clusters(means, cov, weights, i, j, ax=None):
    means_reduced = np.column_stack([[x[i] for x in means],
                                     [y[j] for y in means]])

    cov_reduced = np.array([[cov[:, i, i], cov[:, i, j]],
                            [cov[:, j, i], cov[:, j, j]]]).T

    n_clust = cov_reduced.shape[0]
    w_factor = 1.0 / weights.max()

    # colors = cm.get_cmap("Blues")(np.linspace(0.5, 1, n_clust))
    colors = ["tab:green" * n_clust]

    # sort to highest weights first
    zipped = zip(means_reduced, cov_reduced, weights)
    weight_sorted = sorted(zipped, key=lambda x: x[2])

    for i, (mean, covar, w) in enumerate(weight_sorted):
        _draw_ellipse(mean, covar, edgecolor="tab:green",  # colors[i],
                      linestyle=":", alpha=w * w_factor,
                      facecolor='none', ax=ax)


def plot_density_hist(x, y):
    # Set up the axes with gridspec
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    # plot density map
    sns.kdeplot(x=x, y=y, cmap="rocket_r", shade=True, ax=main_ax)
    # main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)

    # histogram on the attached axes
    x_hist.hist(x, 40, histtype='stepfilled',
                orientation='vertical', color='gray')
    x_hist.invert_yaxis()

    y_hist.hist(y, 40, histtype='stepfilled',
                orientation='horizontal', color='gray')
    y_hist.invert_xaxis()


def _calc_principal_axes(cov_matrix):
    """
    calc principal axes from covariance matrix
    """
    if cov_matrix.shape == (2, 2):
        U, s, Vt = np.linalg.svd(cov_matrix)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(cov_matrix)
    return angle, width, height


def _draw_ellipse(mean, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    angle, width, height = _calc_principal_axes(covariance)

    for nsig in range(1, 3):
        ax.add_patch(Ellipse(mean, nsig * width, nsig * height, angle, **kwargs))


def _get_coords(coords):
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1], coords[:, 2]


def _get_combinations(xs, ys, zs):
    return [((i, j), (x, y)) for (i, x), (j, y) in combinations(enumerate([xs, ys, zs]), 2)]


def plot_elbow(n_clusters, scores, elbow, resid):
    plt.plot(n_clusters, scores, c="#d73027", label="BIC")
    plt.vlines(elbow, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors="#2b83ba", label="elbow")
    plt.title(f"BIC of GMM clustering: {resid}")
    plt.xlabel("Number of clusters")
    plt.ylabel("BIC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./out/clusters/{resid}_bic.pdf")
    plt.show()
