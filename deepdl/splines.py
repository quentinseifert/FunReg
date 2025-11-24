import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.linalg import khatri_rao

import matplotlib as mpl
import matplotlib.pyplot as plt
import deepdl.spline_util as splutl
import ctypes
import scipy
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


class BSplines:
    def __init__(self, x, k, m=2, pen_order=2, standardize=False, uncenter=False, includescale=False, by=None):
        if type(x) != np.ndarray:
            x = x.to_numpy()
        if standardize:
            self.x_mean = np.mean(x)
            self.x_sd = np.std(x)
            x = (x - self.x_mean) / self.x_sd
        self.standardize = standardize
        basis, penalty, knots = splutl.b_spl(x, k=k, m=m, pen_order=pen_order)
        S = splutl.scale_penalty(basis, penalty, scale=includescale)
        self.m = m

        if includescale:
            S, s_scale = S
            self.S_scale = s_scale


        if by is None:
            if uncenter:
                self.X_uncentered = basis
                self.S_uncentered = penalty


            basis, penalty, center_mat = splutl.identconst(basis, S)

            # rescale penalty. Contributes to getting pretty much identical penalty matrix as in mgcv
            self.dim_basis = k-1
            self.basis = basis
            self.penalty = penalty
            self.center_mat = center_mat


        else:
            basis *= by
            self.basis = basis
            self.penalty = S
            self.gammas = None
            self.deltas = None
            self.x_plot = np.linspace(x.min(), x.max(), 50).reshape(50, 1)
            self.dim_basis = basis.shape[1]
            self.pen_order = pen_order


        self.knots = knots
        self.pen_order = pen_order




    def transform_new(self, x_new, center=True):
        # given the knots and F, evaluate new data
        if self.standardize:
            x_new = (x_new - self.x_mean) / self.x_sd
        basis = splutl.b_spl(x_new, k=self.knots, m=self.m, pen_order=self.pen_order, first=False)
        if center:
            basis = basis @ self.center_mat
        return basis




class TPSplines:
    """
    """

    def __init__(self, x, k, pen_order):
        """
        Create the model matrices for Thin-Plate Splines based on the data, rank and penalty order
        :param x:
        :param k:
        :param pen_order:
        :param y:
        """
        n = x.shape[0]
        if len(x.shape) == 1:
            x = x.values.reshape(n, 1)
        d = x.shape[1]
        M = np.math.comb(pen_order + d - 1, d)

        # model matrices
        X, S, UZ, map_idx = splutl.tp_spline(x, k, pen_order, n, d, M)

        # rescale penalty, this is done because it is done in mgcv. Trying to create smoothers that perform
        # comparably and that are affected by smoothing parameters comparably.
        S = splutl.scale_penalty(X, S)
        # center model matrices
        X_centered, S_centered, center_mat = splutl.identconst(X, S)

        # following the mgcv-implemetation I subtract the mean from the original data (also during creation of the model
        # matrices) for d = 2 this created some weird issues (probably just my own mistakes) so I don't do it there.
        # Doesn't change the model so can be ignored
        if d==1:
            self.x_mean = x.mean(0)
            self.x = x - self.x_mean
        if d>=2:
            self.x = x

        # save all attributes that will be useful later
        self.UZ = UZ
        self.matrices = center_mat
        self.basis = X_centered
        self.penalty = S_centered
        self.k = k
        self.M = M
        self.pen_order = pen_order
        self.dim_basis = X_centered.shape[1]
        self.center_mat = center_mat
        self.gammas = None
        self.uncentered_gammas = None
        self.map_idx = map_idx

        # self.x_plot: for plot-method
        if d == 1:
            self.x_plot = np.linspace(self.x.min(), self.x.max(), 1000).reshape(1000, 1)

        # if two-dimensional, prepare for contour-plot
        if d == 2:
            x_plot = np.linspace(x.iloc[:, 0].min(), x.iloc[:, 0].max())
            z_plot = np.linspace(x.iloc[:, 1].min(), x.iloc[:, 1].max())
            mesh = np.meshgrid(x_plot, z_plot)
            self.x_plot = np.column_stack(
                [mesh[0].reshape(2500, 1), mesh[1].reshape(2500, 1)]
            )


    def transform_new(self, x_new):
        """
        evaluate spline for new points:
        Simply create the full model matrix based on the new points
        """
        m = self.pen_order
        M = self.M
        x_orig = np.unique(self.x, axis=0)
        x_new = x_new - self.x_mean[0]
        n = x_new.shape[0]
        if len(x_new.shape) == 1:
            x_new = x_new.values.reshape(n, 1)
        d = x_orig.shape[1]
        # matrix of euclidean distances needed for eta
        E = distance_matrix(x_new, x_orig)
        E = splutl.eta(E, m, d)
        T = splutl.tp_T(x_new, M, m, d)
        ET = np.column_stack([E, T])


        return ET


    def plot(self, ax=None, intercept=0, plot_analytical=False, col='b', col_analytical='r', alpha=1):

        """
        plot-method with some parameters that I mainly created for the plots in my thesis

        """

        m = self.pen_order
        M = self.M
        x_plot = self.x_plot
        x_un = np.unique(self.x, axis=0)
        n = x_plot.shape[0]
        d = x_un.shape[1]
        if self.uncentered_gammas is None:
            self.uncenter()
        E = distance_matrix(x_plot, x_un)
        E = splutl.eta(E, m, d)
        T = splutl.tp_T(x_plot, M, m, d)
        ET = np.column_stack([E, T])
        y_fitted = intercept + ET @ self.uncentered_gammas
        if d == 1:
            # if statement to see whether the plot method is called from GAM or the smoother:
            # if it is from GAM, there is subplot in which to plot, if there is no subplot,
            # just do plt.plot
            # same for every smoother
            if ax is None:
                if plot_analytical:
                    y_plot = intercept + ET @ self.UZ @ self.center_mat @ self.analytical_gammas
                    plt.plot(x_plot + self.x_mean[0], y_plot, col_analytical)
                plt.plot(x_plot, y_fitted, col, alpha=alpha)


            else:
                if plot_analytical:
                    y_plot = intercept + ET  @ self.UZ @ self.center_mat @ self.analytical_gammas
                    ax.plot(x_plot + self.x_mean[0], y_plot, col_analytical)
                ax.plot(x_plot + self.x_mean[0], y_fitted, col, alpha=alpha)



        if d == 2:
            x_mesh = x_plot[:, 0].reshape(50, 50)
            z_mesh = x_plot[:, 1].reshape(50, 50)
            y_fitted = np.array(y_fitted).reshape(50, 50)


            if ax is None:
                plt.contour(x_mesh, z_mesh, y_fitted)
            else:
                cs = ax.contour(x_mesh, z_mesh, y_fitted)
                ax.clabel(cs, inline=True, fontsize=10)

        if d > 2:
            print("No plot :(")

    def uncenter(self):

        # multiply gammas with center mat to "uncenter"
        gammas = self.center_mat @ self.gammas
        # this part is TP-spline specific, additional to "uncentering", I also evaluate the full delta-coefficients
        self.uncentered_gammas = self.UZ @ gammas




class CubicSplines:
    def __init__(self, x, k, cyclic=False, uncenter=False, includescale=False, standardize=False, by=None):

        if standardize:
            self.x_mean = np.mean(x)
            self.x_sd = np.std(x)
            x = (x - self.x_mean) / self.x_sd
        self.standardize = standardize
        self.cyclic = cyclic
        X, S, knots, F = splutl.cr_spl_vec(x, n_knots=k, cyclic=cyclic)

        S = splutl.scale_penalty(X, S, scale=includescale)
        if includescale:
            S, s_scale = S
            self.S_scale = s_scale

        if by is None:
            if uncenter:
                self.X_uncentered = X
                self.S_uncentered = S



            # rescale penalty. Contributes to getting pretty much identical penalty matrix as in mgcv


            # center
            X_centered, S_centered, center_mat = splutl.identconst(X, S)

            self.basis = X_centered
            self.penalty = S_centered
            self.knots = knots
            self.center_mat = center_mat
            self.gammas = None
            self.deltas = None
            self.x_plot = np.linspace(x.min(), x.max(), 50).reshape(50, 1)
            self.dim_basis = X_centered.shape[1]
            self.F = F

        else:
            X *= by
            self.basis = X
            self.penalty = S
            self.knots = knots
            self.gammas = None
            self.deltas = None
            self.x_plot = np.linspace(x.min(), x.max(), 50).reshape(50, 1)
            self.dim_basis = X.shape[1]
            self.F = F



    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas


    def transform_new(self, x_new, center=True):
        # given the knots and F, evaluate new data
        if self.standardize:
            x_new = (x_new - self.x_mean) / self.x_sd
        basis = splutl.cr_spl_predict(x_new, knots=self.knots, F=self.F, cyclic=self.cyclic)
        if center:
            basis = basis @ self.center_mat
        return basis


    def plot(self, ax=None, intercept=0, plot_analytical=False, col='b', alpha=1, col_analytical='r'):

        # evaluate x_plot
        basis = splutl.cr_spl_predict(self.x_plot, knots=self.knots, F=self.F)
        # create fitted_values
        y_fitted = intercept + basis @ self.uncentered_gammas

        # plot into given subplot or simply new plot
        if ax is None:
            # if least squares solution should be plotted ...
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                plt.plot(self.x_plot, y_plot, col_analytical)
            plt.plot(self.x_plot, y_fitted, color=col, alpha=alpha)

        else:
            if plot_analytical:
                y_plot = intercept + basis @ self.center_mat @ self.analytical_gammas
                ax.plot(self.x_plot, y_plot, col_analytical)
            ax.plot(self.x_plot, y_fitted, col, alpha=alpha)






class MRFSmooth:
    def __init__(self,
                 x,
                 polygons=None,
                 penalty=None, includescale=False, center=True):
        x = pd.Categorical(x, categories=sorted(set(x)))
        ids = x.categories
        x = x.codes
        self.polygons = polygons
        basis = np.eye(len(ids))[x]
        self.ids = ids.to_numpy
        print(self.ids)
        #basis = splutl.mrf_design(regions=x, pc=polygons)
        penalty = splutl.pol2nb(pc=polygons.copy())
        penalty = splutl.scale_penalty(basis, penalty)
        S = splutl.scale_penalty(basis, penalty, scale=includescale)
        if includescale:
            S, s_scale = S
            self.S_scale = s_scale
        if center:
            basis, S, center_mat = splutl.identconst(basis, penalty)
            self.center_mat = center_mat

        self.ids = ids
        self.basis = basis
        self.penalty = S
        self.dim_basis = basis.shape[1]


    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas

    def plot(self, col1='blue', col2='red', intercept=None, plot_analytical=None, ax=None, minmax=None, bar=False):
        pols = self.polygons
        if self.polygons is None:
            print("Need map")
        else:
            if self.uncentered_gammas is None:
                self.uncenter()

            # weird behaviour of tensorflow objects requires me to do this (or I am missing something)
            full_gammas = self.uncentered_gammas.numpy()

            if minmax:
                full_gammas = (full_gammas - min(minmax)) / (max(minmax) - min(minmax))

            else:
                full_gammas = (full_gammas - min(full_gammas)) / (max(full_gammas) - min(full_gammas))

            mix_dict = {k: v for k, v in zip(pols, full_gammas)}

            # for colorbar
            # values from 0 to 1
            mix = np.linspace(0, 1, 100)
            # retrieve color codes from col1 to col2
            col_list = splutl.color_fader(col1, col2, mix)
            cmap = mpl.colors.ListedColormap(col_list)
            mapped_colors = splutl.color_bounds(self.uncentered_gammas.numpy())
            norm = mpl.colors.BoundaryNorm(mapped_colors, cmap.N)


            if ax is None:
                for i in pols.keys():
                    plt.fill(pols[i][:, 0], pols[i][:, 1],
                             color=splutl.color_fader(col1, col2, mix=mix_dict[i][0] / 1))
                plt.axis("off")

            else:
                for i in pols.keys():
                    cmap = sns.color_palette("rocket", as_cmap=True)
                    #ax.fill(pols[i][:, 0], pols[i][:, 1],
                    #        color=splutl.color_fader(col1, col2, mix=mix_dict[i][0] / 1))
                    ax.fill(pols[i][:, 0], pols[i][:, 1],
                                   color=cmap(mix_dict[i][0] / 1))
                if bar:
                    plt.colorbar(sns.color_palette("rocket", as_cmap=True))

                #plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
                ax.axis("off")


    def transform_new(self, x_new, center=True):
        x_new = np.where(self.ids.to_numpy()[:, None] == x_new[:, None, None])[1]
        new_basis = np.eye(len(self.ids))[x_new]
        if center:
            new_basis = new_basis @ self.center_mat
        return new_basis

    #def transform_new(self, x_new):
    #    return splutl.mrf_design(regions=x_new, pc=self.polygons)



class TeProductSpline:
    def __init__(self, spl1, spl2):
        X1 = spl1.X_uncentered
        X2 = spl2.X_uncentered
        S1 = spl1.S_uncentered
        S2 = spl2.S_uncentered
        X = khatri_rao(X1.T, X2.T).T
        S_te1 = np.kron(S1, np.eye(spl2.dim_basis+1))
        S_te2 = np.kron(np.eye(spl1.dim_basis+1), S2)
        S_te1 = scale_penalty(X, S_te1)
        S_te2 = scale_penalty(X, S_te2)

        self.basis = X
        self.penalty = [S_te1, S_te2]
        #


class TiProductSpline:
    def __init__(self, spl1, spl2, plot=False):
        X1 = spl1.basis
        X2 = spl2.basis
        S1 = spl1.penalty * spl1.S_scale
        S2 = spl2.penalty * spl1.S_scale
        X = khatri_rao(X1.T, X2.T).T
        S_te1 = np.kron(S1, np.eye(spl2.dim_basis))
        S_te2 = np.kron(np.eye(spl1.dim_basis), S2)
        S_te1 = splutl.scale_penalty(X, S_te1)
        S_te2 = splutl.scale_penalty(X, S_te2)
        self.spl1 = spl1
        self.spl2 = spl2

        self.basis = X
        self.penalty = [S_te1, S_te2]
        if plot:
            self.plot_basis = khatri_rao(self.spl1.transform_new(self.spl1.x_plot).T,
                                         self.spl2.transform_new(self.spl2.x_plot).T).T

            mesh = np.meshgrid(spl1.x_plot, spl2.x_plot)

            self.plot_basis = khatri_rao(self.spl1.transform_new(mesh[0].reshape(2500, 1)).T,
                                         self.spl2.transform_new(mesh[1].reshape(2500, 1)).T).T
            self.x_plot = np.column_stack(
                [mesh[0].reshape(2500, 1), mesh[1].reshape(2500, 1)]
            )

    def tranform_new(self, x, z):
        x_basis = self.spl1.transform_new(x)
        z_basis = self.spl2.transform_new(z)
        basis = khatri_rao(x_basis.T, z_basis.T).T
        return basis


    def plot(self, ax=None, surface=True):
        y_fitted = self.plot_basis @ self.gammas
        x_mesh = self.x_plot[:, 0].reshape(50, 50)
        z_mesh = self.x_plot[:, 1].reshape(50, 50)
        y_fitted = np.array(y_fitted).reshape(50, 50)

        if ax is None:
            if surface:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Surface plot
                ax.plot_surface(x_mesh, z_mesh, y_fitted, cmap='viridis')
            else:
                plt.contour(x_mesh, z_mesh, y_fitted)
        else:
            if surface:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Surface plot
                ax.plot_surface(x_mesh, z_mesh, y_fitted, cmap='viridis')
            cs = ax.contour(x_mesh, z_mesh, y_fitted)
            ax.clabel(cs, inline=True, fontsize=10)


class RandomEffect:
    def __init__(self, x, uncenter=False, includescale=False):
        x, ids = pd.factorize(x)
        basis = np.eye(len(ids))[x]
        penalty = np.eye(basis.shape[1])

        if uncenter:
            self.X_uncentered = basis
            self.S_uncentered = penalty
        penalty = splutl.scale_penalty(basis, penalty)
        S = splutl.scale_penalty(basis, penalty, scale=includescale)
        if includescale:
            S, s_scale = S
            self.S_scale = s_scale
        basis, penalty, center_mat = splutl.identconst(basis, penalty)
        self.basis = basis
        self.ids = ids
        self.penalty = penalty
        self.dim_basis = basis.shape[1]
        self.center_mat = center_mat


    def uncenter(self):
        self.uncentered_gammas = self.center_mat @ self.gammas


    def transform_new(self, x_new, center=True):
        x_new_codes = x_new.map({v: i for i, v in enumerate(self.ids)}).fillna(-1).astype(int)
        new_basis = np.eye(len(self.ids))[x_new_codes]
        if center:
            new_basis = new_basis @ self.center_mat
        return new_basis


