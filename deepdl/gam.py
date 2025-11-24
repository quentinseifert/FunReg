# Hier die Dependencies rein:
import tensorflow as tf


import numpy as np
import os as os
from statsmodels.gam.api import CyclicCubicSplines, BSplines
import matplotlib.pyplot as plt
from deepdl.splines import TPSplines, CubicSplines, MRFSmooth
# Class ohne Subclass und helperfunktionen:
import deepdl.utils as utl
import scipy

# This file contains the GAM-class which is based on a working version René created
# some time ago. I have extended it to work with my classes.

class GAM(object):

    def __init__(
        self,
        formula,
        nn=None,
        max_iter=1000,
        tol=1e-4,
        distribution="normal",
        link="identity",
        callbacks=["deviance", "diffs"],
        directory="./",
        polygons=None,
        reg_params=None,
        initializer=tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.1
        ),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam'
        ),

    ):
        # Verteilung, bis jetzt nur Normalverteilung:
        # self.distribution = normal(location and scale)
        self.model_formula = formula
        self.nn = nn
        # self.hyperparam = hyperparam
        self.max_iter = max_iter
        self.tol = tol
        self.distribution = distribution
        self.link = link
        self.callbacks = callbacks
        self.directory = directory
        if not os.path.exists(
            self.directory
        ):  # Schauen ob es das Output Dir gibt. Wenn nicht anlegen. Bis jetzt temporäres Dir,
            os.mkdir(directory)
        self._lambda_constrait = 1e9  # regularisierungs parameter (lambda)
        self._l2_constraints = 1e-3  # default l2
        self._l2_constraint_max = 1e-1  # maximum l2
        self.polygons = polygons
        self.intercept = 0
        self.reg_params = reg_params
        self.optimizer = optimizer
        self.initializer = initializer
        super(GAM, self).__init__()  # calling super, ohne variablen ausführbar machen

    def fit(self, data, plot=False):
        y_terms = next(iter(self.model_formula.items()))[
            0
        ]  # 1 Abschnitt: Model Gleichung manipulieren:  1.1 y terme auslesen:
        x_terms = utl.split_formular(
            formula=self.model_formula
        )  # 1.2 x termine auslesen,  1.2.1 generell alle auslesen.
        wanted = "spline"  # Signalworte # 1.2.2 Splines auslesen aus X_terme # hier müssen dann andere signalwörter eingeführt werden.
        fixe_terme = list(filter(lambda x: wanted not in x, x_terms))  # fixe terme

        if len(fixe_terme) > 0:
            if fixe_terme[0] == '1':
                self.intercept = True

        spline_terme = list(filter(lambda x: wanted in x, x_terms))
        self.data = data  # 2. Daten Manipulaton
        self.y = self.data[y_terms]  # 2.2.1 y auslesen:
        self.y = self.y.astype("float32")
        self.y = np.expand_dims(self.y, 1)
        self.daten_matrix = utl.get_design_matrix(
            data=self.data, formula=self.model_formula
        )
        self.daten_matrix = self.daten_matrix.astype("float32")
        tf.keras.backend.set_floatx("float32")


        spline_args = dict()
        spline_names = list()
        spline_types = list()
        for i in range(len(spline_terme)):
            foo = spline_terme[i]
            foo = foo.replace("spline(", "")
            foo = foo.replace(")", "")
            foo = foo.split(",")

            spl_name = utl.find_spl_name(foo)
            spline_names.append(spl_name)
            spline_args[spl_name] = dict()

            universal_matchers = ["type", "k"] # for tp, n_knots = k

            spl_type = [match for match in foo if "type" in match]
            spl_type = spl_type[0].split("=")[1]
            spline_types.append(spl_type)


            # this next part is pretty convoluted since the arguments for different smooth terms need
            # to be read out in a somewhat messy way. But for now, it works


            # arguments the bs, cr, and tp share
            if spl_type in ["bs", "cr", "tp"]:
                spline_args[spl_name][universal_matchers[0]] = spl_type
                k = [match for match in foo if "k" in match]
                k = int(k[0].split("=")[1])
                spline_args[spl_name][universal_matchers[1]] = k


                # bs specific arguments
                if spl_type=='bs':
                    matchers = ["degree", "pen_ord"]
                    degrees = [match for match in foo if "degree" in match]
                    degree = int(degrees[0].split("=")[1])
                    spline_args[spl_name][matchers[0]] = degree
                    penal_ord = [match for match in foo if "pen_ord" in match]
                    pen_ord = int(penal_ord[0].split("=")[1])
                    spline_args[spl_name][matchers[1]] = pen_ord


                # tp specific arguments
                if spl_type=='tp':
                    if spl_name.find("[") != -1:
                        # this allows me to have TP-splines with more than one variable
                        # a bit convoluted, but it works
                        spline_args[spl_name]["next"] = spl_name.count(",") + 1
                    else:
                        spline_args[spl_name]["next"] = 1

                    pen_ord = [match for match in foo if "pen_ord" in match]
                    pen_ord = int(pen_ord[0].split("=")[1])
                    spline_args[spl_name]["pen_ord"] = pen_ord


        # prepare array of spline types, makes it possible to use np.where
        spline_types = np.array(spline_types)

        target = tf.constant(self.y, dtype=None, shape=self.y.shape)
        design_x = tf.constant(
            self.daten_matrix[fixe_terme],
            dtype="float32",
            shape=(self.daten_matrix[fixe_terme].shape),
            name="design_x",
        )

        # 5.2 Spline Effekte
        # 5.2.1 Basis Matrix erstellen:

        # needed to change some things as compared to the old version because
        # this way I can create multidimensional tp-splines
        idx = utl.get_idx(spline_names)
        spline_dat = self.data[idx]


        include_intercept = False
        knot_kwds = None


        # initialize lists for spline design and penalty matrices
        # self.splines to be able to access the indiviual smoothers after fitting

        self.matrix_B = [None] * len(spline_args)
        self.penalty = [None] * len(spline_args)
        self.dim_basis = [None] * len(spline_args)
        self.splines = [None] * len(spline_args)

        # which splines are bs? loop through bs-splines and create objects
        idx_bs = np.where(spline_types == "bs")[0]
        if len(idx_bs) > 0:
            for i in range(len(idx_bs)):
                spl_name = spline_names[idx_bs[i]]
                df = spline_args[spl_name]['k']
                degree = spline_args[spl_name]['degree']
                basis_spline = BSplines(
                    spline_dat.iloc[:, idx_bs[i]], df=df, degree=degree,
                    include_intercept=False, knot_kwds=None
                )
                upper_limit =  max(spline_dat.iloc[:, idx_bs[i]])
            # 5.2.2 Matrzen abspeichern
                self.matrix_B[idx_bs[i]] = basis_spline.basis
                self.penalty[idx_bs[i]] = basis_spline.penalty_matrices[0]
                self.dim_basis[idx_bs[i]] = basis_spline.dim_basis
                self.splines[idx_bs[i]] = basis_spline


        # which ones are tp splines, loop through them
        idx_tp = np.where(spline_types == 'tp')[0]
        if len(idx_tp) > 0:
            for i in range(len(idx_tp)):
                spl_name = spline_names[idx_tp[i]]
                k = spline_args[spl_name]['k']
                pen_ord = spline_args[spl_name]['pen_ord']

                # if it is a multidimensional tp-spline, the entry "next" tells us how many columns
                # after the first to include
                idx_tp_plus = idx_tp[i] + spline_args[spl_name]["next"]
                basis_spline = TPSplines(
                    spline_dat.iloc[:, idx_tp[i]:idx_tp_plus], k=k, pen_order=pen_ord)

                self.matrix_B[idx_tp[i]] = basis_spline.basis
                self.penalty[idx_tp[i]] = basis_spline.penalty
                self.dim_basis[idx_tp[i]] = basis_spline.dim_basis
                self.splines[idx_tp[i]] = basis_spline


        # loop through cr splines
        idx_cr = np.where(spline_types == 'cr')[0]
        if len(idx_cr) > 0:
            for i in range(len(idx_cr)):
                spl_name = spline_names[idx_cr[i]]
                k = spline_args[spl_name]['k']
                spline = CubicSplines(
                    spline_dat.iloc[:, idx_cr[i]], k
                )
                self.matrix_B[idx_cr[i]] = spline.basis
                self.penalty[idx_cr[i]] = spline.penalty
                self.dim_basis[idx_cr[i]] = spline.dim_basis
                self.splines[idx_cr[i]] = spline

        # loop through mrf
        idx_mrf = np.where(spline_types == 'mrf')[0]
        if len(idx_mrf) > 0:
            for i in range(len(idx_mrf)):
                spl_name = spline_names[idx_mrf[i]]
                spline = MRFSmooth(
                    spline_dat.iloc[:, idx_mrf[i]], polygons=self.polygons
                )
                self.matrix_B[idx_mrf[i]] = spline.basis
                self.penalty[idx_mrf[i]] = spline.penalty
                self.dim_basis[idx_mrf[i]] = spline.dim_basis
                self.splines[idx_mrf[i]] = spline

        # 5.3. gammas erstellen


        # are reg_params supplied? if not, they are zero
        if self.reg_params == None:
            self.reg_params = np.repeat(0, len(self.matrix_B))

        # loop through reg_params and multiply with the corresponding penalty matrix
        for i in range(len(self.reg_params)):
            self.penalty[i] = self.penalty[i] * self.reg_params[i]

        ## Model Matrix aufstellen
        # penalty für fixe terme = 0
        pen_x = np.zeros((len(fixe_terme), len(fixe_terme)))

        # blockdiagonal penalty from all penalty matrices
        P_matrix = tf.constant(scipy.linalg.block_diag(*self.penalty), dtype='float32')

        # smoother design matrix
        design_z = tf.constant(np.column_stack(self.matrix_B), dtype='float32')

        # full design matrix
        self.design_matrix = tf.concat([design_x, design_z], 1)

        # initialize weights for splines a linear terms
        gammas = tf.Variable(self.initializer(shape=(design_z.shape[1], 1)), name="gammas")
        betas = tf.Variable(self.initializer(shape=(design_x.shape[1], 1)), name="betas")


        # optimize

        for j in range(self.max_iter):
            if (j+1) % 10 == 0:
                print("epoch: " + str(j+1) + "/" + str(self.max_iter))
            loss = lambda: utl.model(
                Z=design_z,
                X=design_x,
                gammas=gammas,
                betas=betas,
                y=target,
                penalty_mat=P_matrix
            )
            # if there are no betas, only gammas in var_list, else we get a warning
            if design_x.shape[1] > 0:
                self.optimizer.minimize(loss, [betas, gammas])
            else:
                self.optimizer.minimize(loss, [gammas])

        # concatenate coefficients
        gammas = tf.concat([betas, gammas], axis=0)

        # save intercept explicitely
        if self.intercept:
            self.intercept = gammas[0]
        else:
            self.intercept = 0

        for i in range(len(self.splines)):
            # split coefficients so they can be passed to the individual smoother objects and be accessed from there
            # find where they start and end with lower/upper
            lower = len(fixe_terme) + sum(self.dim_basis[:i])
            upper = len(fixe_terme) + sum(self.dim_basis[:i+1])

            self.splines[i].gammas = gammas[lower: upper]
            # call uncenter-method to create uncentered_gammas for individual smoothers (practical for
            # plotting and such
            self.splines[i].uncenter()
            # save smoothing parameter in corresponding smoother
            self.splines[i].reg_param = self.reg_params[i]
            self.splines[i].name = spline_names[i]


        self.gammas = gammas
        self.fitted_values = self.design_matrix @ gammas
        self.residuals = target - self.fitted_values
        self.len_fixe_terme = len(fixe_terme)
        self.fixe_terme = fixe_terme
        # full penalty matrix, including zero block for linear terms
        # (so I can easily calculate PLS-estimator)

        pen_x = np.zeros((len(fixe_terme), len(fixe_terme)))
        self.penalty = tf.constant(scipy.linalg.block_diag(pen_x, *self.penalty), dtype='float32')




    def plot(self, plot_analytical=False):
        """
        plot-method that plots individual smoothers in one window
        Parameters
        ----------
        plot_analytical: plot the PLS-estimator?

        Returns plt.plot-obtject
        -------

        """
        n_plots = len(self.splines)
        if n_plots == 1:
            fig, ax = plt.subplots(1, 1)
            # for plots with only one spline, plot intercept ..
            self.splines[0].plot(ax=ax, intercept=self.intercept, plot_analytical=plot_analytical)
        elif n_plots == 2 or n_plots == 3:
            fig, ax = plt.subplots(1, n_plots, figsize=(12, 3.5))
            for i in range(n_plots):
                # else not
                self.splines[i].plot(ax=ax[i], intercept=0, plot_analytical=plot_analytical)

        else:
            n_row = int(np.ceil(np.sqrt(n_plots)))
            n_col = int(np.ceil(n_plots / n_row))
            fig, ax = plt.subplots(n_row, n_col, figsize=(16, 10))
            for i in range(n_plots):
                row = int(np.floor(i / n_row))
                col = i % n_row
                self.splines[i].plot(ax=ax[row, col], intercept=0, plot_analytical=plot_analytical)

        plt.show()
        return fig, ax


    def fit_analytical(self):
        """
        PLS-estimator (X'X + lambda S)^(-1) X'y
        Returns
        -------

        """

        X = self.design_matrix
        S = self.penalty
        self.analytical_gammas = np.linalg.inv(tf.transpose(X) @ X + S) @ tf.transpose(X) @ self.y

        for i in range(len(self.splines)):
            # pass the PLS-coefficients on to smoothers
            lower = self.len_fixe_terme + sum(self.dim_basis[:i])
            upper = self.len_fixe_terme + sum(self.dim_basis[:i+1])
            self.splines[i].analytical_gammas = self.analytical_gammas[lower: upper]



    def fit_new(self, data):
        """
        simple prediction method, needs pd.Dataframe with appropriately named columns.
        """

        y_fit = self.intercept
        for i in range(len(self.fixe_terme)):
            if self.fixe_terme[i] == '1':
                continue
            y_fit += data[self.fixe_terme[i]] * self.gammas[i]

        y_fit = tf.expand_dims(y_fit, 1)

        for i in range(len(self.splines)):
            name = self.splines[i].name
            new_basis = self.splines[i].transform_new(data[name])
            y_fit += new_basis @ self.splines[i].uncentered_gammas

        return y_fit

