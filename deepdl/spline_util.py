import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigsh
import bisect
import matplotlib as mpl
import ctypes
import scipy
import scipy
import os
this_dir = os.path.dirname(__file__)                  # code/
so_path = os.path.join(this_dir, "c_splines.so")
lib = ctypes.CDLL(so_path)




def eta(E, m, d):

    """
    Calculate the eta function given a matrix of Euclidean distances, penalty order and dimensionality
    of the data
    :param E: Matrix of euclidean distances between points
    :param m: penalty order
    :param d: dimensionality of data
    :return: eta-fct of the supplied Euclidean distances
    """
    if d % 2 == 0:
        const = (((-1) ** (m + 1 + d / 2)) /
                 (2 ** (2 * m - 1) * np.pi ** (d / 2) * np.math.factorial(m - 1) * np.math.factorial(int(m - d / 2))))
        E = const * E ** (2 * m - d) * np.log(E)

    else:
        E = np.math.gamma(d / 2 - m) / \
            (2 ** (2 * m) * np.pi ** (d / 2) * np.math.factorial(m - 1)) * E ** (2 * m - d)
    np.nan_to_num(E, 0)
    return E

def tp_spline(x, k, pen_order, n, d, M):

    # subtract mean from data (try to recreate model matrix in mgcv, did not work. Doesn't change the model, so can be
    # ignored
    if d == 1:
        x = x - x.mean()

    # reduce the data to unique observations and save the index to create the full matrix later
    x_un = np.unique(x, axis=0)
    map_idx = np.all(
        (np.expand_dims(np.array(x_un), 0) == np.expand_dims(x, 1)),
                     axis=2)
    map_idx = np.argwhere(map_idx)

    # matrix of euclidean distances needed for eta
    E = distance_matrix(x_un, x_un)
    E = eta(E, pen_order, d)

    # get first k eigenvalues
    # eigsh because it is way faster than np.linalg.eigh
    eigen_values, U = eigsh(E, k, which='LA')
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    U = U[:, idx]
    D = np.diag(eigen_values)

    # U_k: first k eigenvectors
    # D_k: diagonal matrix of first k eigenvalues
    U_k = U[:, :k]
    D_k = D[:k, :k]
    T = tp_T(x_un, M, pen_order, d)


    # absorb constraint T * delta = 0

    q, r = np.linalg.qr(np.dot(U_k.T, T), mode="complete")
    Z_k = q[:, M:]

    UZ = U_k @ Z_k
    # create penalty matrix S (padded by zeros for unpenalized alpha-part)
    S = Z_k.T @ D_k @ Z_k
    S_full = np.zeros((k, k))
    S_full[:k - M, :k - M] = S
    # finalize design matrix
    X = U_k @ D_k @ Z_k
    X = np.column_stack([X, T])


    X_full = X[map_idx[:, 1], :]

    # make UZ a blockdiagonal matrix with an M-dimensional identity matrix in its lower right block.
    # This way the full delta can be evaluated without having to split up gamma = [delta, alpha]
    UZ_full = np.zeros((UZ.shape[0] + M, k))
    UZ_full[:UZ.shape[0], :k-M] = UZ
    UZ_full[UZ.shape[0]:, k-M:] = np.eye(M)


    # create matrix W to rescale columns of X (see mgcv/src/tprs.c)
    # This step is not mentioned in the TP paper or the GAM-book
    # speeds up convergence immensely
    w = np.sqrt((X_full**2).sum(0) / n)
    W = np.diag(1 / w)
    X_full = X_full @ W
    S_full = W @ S_full @ W
    UZ_full = UZ_full @ W
    return X_full, S_full, UZ_full, map_idx


def get_FS(xk, cyclic=False):
    """
    Create matrix F required to build the spline base and the penalizing matrix S,
    based on a set of knots xk (ascending order). Pretty much directly from p.201 in Wood (2017)
    :param xk: knots (for now always np.linspace(x.min(), x.max(), n_knots)
    """
    k = len(xk)
    h = np.diff(xk)
    h_shift_up = h.copy()[1:]

    if not cyclic:

        D = np.zeros((k - 2, k))
        np.fill_diagonal(D, 1 / h[:k-2])
        np.fill_diagonal(D[:, 1:], (-1 / h[:k - 2] - 1 / h_shift_up))
        np.fill_diagonal(D[:, 2:], 1 / h_shift_up)


        B = np.zeros((k - 2, k - 2))
        np.fill_diagonal(B, (h[:k - 2] + h_shift_up) / 3)
        np.fill_diagonal(B[:, 1:], h_shift_up[:k - 3] / 6)
        np.fill_diagonal(B[1:, :], h_shift_up[:k - 3] / 6)

    else:
        D = np.zeros((k-1, k-1))
        D[0, 0] = -1/h[0] - 1/h[-1]
        np.fill_diagonal(D[1:, :], 1/h)
        np.fill_diagonal(D[:, 1:], 1/h)
        np.fill_diagonal(D[1:, 1:], -1/h_shift_up - 1/h[:-1])
        D[0, k-2] = 1/h[-1]
        D[k-2, 0] = 1/h[-1]



        B = np.zeros((k-1, k-1))
        B[0, 0] = (h[-1] + h[0]) / 3
        np.fill_diagonal(B[1:, 1:], (h[:-1] + h_shift_up) / 3)
        np.fill_diagonal(B[1:, :], h/6)
        np.fill_diagonal(B[:, 1:], h/6)
        B[0, k-2] = h[-1]/6
        B[0, k-2] = h[-1]/6
        B[k-2, 0] = h[-1]/6

    F_minus = np.linalg.inv(B) @ D
    S = D.T @ np.linalg.inv(B) @ D

    if cyclic:
        return F_minus, S
    F = np.vstack([np.zeros(k), F_minus, np.zeros(k)])
    return F, S


def cr_spl(x, n_knots, cyclic=False):
    """

    :param x: x values to be evalutated
    :param n_knots: number of knots
    :return:
    """
    if isinstance(n_knots, (list, np.ndarray)):
        xk = n_knots
    else:
        xk = np.linspace(x.min(), x.max(), n_knots)
    n = len(x)
    k = len(xk)
    F, S = get_FS(xk, cyclic)
    if not cyclic:
        base = np.zeros((n, k))

        for i in range(0, len(x)):

            if x[i] < min(xk):
                j = 0
                h = xk[1] - xk[0]
                xik = x[i] - xk[0]
                c_jm = -xik * h / 3
                c_jp = -xik * h / 6
                base[i, :] = c_jm * F[0, :] + c_jp * F[1, :]
                base[i, 0] += 1 - xik / h
                base[i, 1] += xik / h
            elif x[i] > max(xk):
                j = len(xk) - 1
                h = xk[j] - xk[j - 1]
                xik = x[i] - xk[j]
                c_jm = xik * h / 6
                c_jp = xik * h / 3
                base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, 1]
                base[i, j - 1] += -xik / h
                base[i, j] += 1 + xik / h
            # find interval in which x[i] lies
            # and evaluate basis function from p.201 in Wood (2017)
            else:
                j = bisect.bisect_left(xk, x[i])
                x_j = xk[j - 1]
                x_j1 = xk[j]
                h = x_j1 - x_j
                a_jm = (x_j1 - x[i]) / h
                a_jp = (x[i] - x_j) / h
                c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
                c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
                base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += a_jm
                base[i, j] += a_jp

    else:
        base = np.zeros((n, k-1))

        for i in range(len(x)):
            j = bisect.bisect_left(xk, x[i])
            jl = j - 1
            x_j = xk[jl]
            x_j1 = xk[j]
            if j==(k-1):
                j = 0
                jl = k - 2
           #if j==k-1:
           #    j == 0
           #if j == (k - 1):
           #    j = 0
           #    jl = (k - 1)


            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
            base[i, :] = c_jm * F[jl, :] + c_jp * F[j, :]
            base[i, jl] += a_jm
            base[i, j] += a_jp


    return base, S, xk, F


def cr_spl_predict(x, knots, F, cyclic=False):
    """
    pretty much the same as cr_spl, this time evaluating it for already given knots and F
    (could probably just be integrated into cr_spl)
    """
    n = len(x)
    k = len(knots)
    F, S = get_FS(knots, cyclic)
    if not cyclic:
        base = np.zeros((n, k))

        for i in range(0, len(x)):

            if x[i] < min(knots):
                j = 0
                h = knots[1] - knots[0]
                xik = x[i] - knots[0]
                c_jm = -xik * h / 3
                c_jp = -xik * h / 6
                base[i, :] = c_jm * F[0, :] + c_jp * F[1, :]
                base[i, 0] += 1 - xik / h
                base[i, 1] += xik / h
            elif x[i] > max(knots):
                j = len(knots) - 1
                h = knots[j] - knots[j - 1]
                xik = x[i] - knots[j]
                c_jm = xik * h / 6
                c_jp = xik * h / 3
                base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, 1]
                base[i, j - 1] += -xik / h
                base[i, j] += 1 + xik / h
            # find interval in which x[i] lies
            # and evaluate basis function from p.201 in Wood (2017)
            else:
                j = bisect.bisect_left(knots, x[i])
                x_j = knots[j - 1]
                x_j1 = knots[j]
                h = x_j1 - x_j
                a_jm = (x_j1 - x[i]) / h
                a_jp = (x[i] - x_j) / h
                c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
                c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
                base[i, :] = c_jm * F[j - 1, :] + c_jp * F[j, :]
                base[i, j - 1] += a_jm
                base[i, j] += a_jp

    else:
        base = np.zeros((n, k - 1))

        for i in range(len(x)):
            j = bisect.bisect_left(knots, x[i])
            jl = j - 1
            x_j = knots[jl]
            x_j1 = knots[j]
            if j == (k - 1):
                j = 0
                jl = k - 2


            h = x_j1 - x_j
            a_jm = (x_j1 - x[i]) / h
            a_jp = (x[i] - x_j) / h
            c_jm = ((x_j1 - x[i]) ** 3 / h - h * (x_j1 - x[i])) / 6
            c_jp = ((x[i] - x_j) ** 3 / h - h * (x[i] - x_j)) / 6
            base[i, :] = c_jm * F[jl, :] + c_jp * F[j, :]
            base[i, jl] += a_jm
            base[i, j] += a_jp

    return base


def cr_spl_vec(x, n_knots, cyclic=False):
    if isinstance(n_knots, (list, np.ndarray)):
        xk = n_knots
    else:
        xk = np.linspace(x.min(), x.max(), n_knots)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    n = len(x)
    k = len(xk)
    F, S = get_FS(xk, cyclic)

    if not cyclic:
        base = np.zeros([n, k])
    else:
        base = np.zeros([n, k - 1])

    # Find interval indices
    j = np.searchsorted(xk, x, side="left")

    # Non-cyclic
    if not cyclic:
        mask_left = j == 0
        mask_right = j == k
        mask_mid = ~(mask_left | mask_right)

        # Left boundary
        if np.any(mask_left):
            h = xk[1] - xk[0]
            xik = x[mask_left] - xk[0]
            c_jm = -xik * h / 3
            c_jp = -xik * h / 6
            base[np.ix_(mask_left, range(k))] = c_jm[:, None] * F[0, :] + c_jp[:, None] * F[1, :]
            base[mask_left, 0] += 1 - xik / h
            base[mask_left, 1] += xik / h

        # Right boundary
        if np.any(mask_right):
            h = xk[-1] - xk[-2]
            xik = x[mask_right] - xk[-1]
            c_jm = xik * h / 6
            c_jp = xik * h / 3
            base[np.ix_(mask_right, range(k))] = c_jm[:, None] * F[-2, :] + c_jp[:, None] * F[-1, :]
            base[mask_right, -2] += -xik / h
            base[mask_right, -1] += 1 + xik / h

        # Interior
        if np.any(mask_mid):
            jm = j[mask_mid] - 1
            jp = j[mask_mid]
            x_j = xk[jm]
            x_j1 = xk[jp]
            h = x_j1 - x_j
            dx_left = x_j1 - x[mask_mid]
            dx_right = x[mask_mid] - x_j

            a_jm = dx_left / h
            a_jp = dx_right / h
            c_jm = (dx_left ** 3 / h - h * dx_left) / 6
            c_jp = (dx_right ** 3 / h - h * dx_right) / 6

            rows = np.where(mask_mid)[0]
            base[np.ix_(rows, range(k))] = c_jm[:, None] * F[jm, :] + c_jp[:, None] * F[jp, :]
            base[rows, jm] += a_jm
            base[rows, jp] += a_jp

    # Cyclic
    else:
        # Wrap indices for cyclic: last interval wraps to first
        jm = (j - 1) % (k - 1)
        jp = j % (k - 1)
        h = xk[jp] - xk[jm]
        # Handle wrap-around distance
        h[h <= 0] += xk[-1] - xk[0]

        dx_left = xk[jp] - x
        dx_left[dx_left < 0] += xk[-1] - xk[0]
        dx_right = x - xk[jm]
        dx_right[dx_right < 0] += xk[-1] - xk[0]

        a_jm = dx_left / h
        a_jp = dx_right / h
        c_jm = (dx_left ** 3 / h - h * dx_left) / 6
        c_jp = (dx_right ** 3 / h - h * dx_right) / 6

        rows = np.arange(n)
        base[np.ix_(rows, range(k - 1))] = c_jm[:, None] * F[jm, :] + c_jp[:, None] * F[jp, :]
        base[rows, jm] += a_jm
        base[rows, jp] += a_jp

    return base, S, xk, F


def b_spl(x, k, m=2, pen_order=2, first=True):
    if first:
        x_low = np.min(x)
        x_high = np.max(x)
        nk = k - m
        x_range = x_high - x_low
        x_low -=  x_range * 0.001
        x_high += x_range * 0.001
        dist_x = (x_high - x_low) / (nk - 1)
        knots = np.linspace(x_low - dist_x * (m + 1), x_high + dist_x * (m + 1), nk + 2 * m + 2)
    else:
        knots = k
        k = len(knots) - 2 * m

    n = len(x)


    lib.spline_basis_matrix_full.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ]


    basis = np.zeros((n, k), dtype=np.float64)

    lib.spline_basis_matrix_full(knots, len(knots), m+2, x, n, k, basis)
    if not first:
        return basis

    D = np.diff(np.eye(k), pen_order)
    S = D @ D.T

    return basis, S, knots


# call the function

def scale_penalty(basis, penalty, scale=False):
    """
    rescale the penalty matrix based on the design matrix of the smoother
    from mgcv to get penalties that react comparably to smoothing parameters
    (works for CubicSplines and MRFSmooth, not for TPSpline since the model
    matrices are completely different)
    """
    X_inf_norm = max(np.sum(abs(basis), axis=1)) ** 2
    S_norm = np.linalg.norm(penalty, ord=1)
    norm = S_norm / X_inf_norm
    penalty = penalty / norm
    if scale:
        return penalty, norm
    return penalty


def identconst(basis, penalty):
    """
    create constraint matrix and absorb identifiability constraint into model matrices:
    returns centered model matrices as well orthogonal factor Z to map centered matrices
    back to unconstrained column space
    """
    constraint_matrix = basis.mean(axis=0).reshape(-1, 1)
    q, r = np.linalg.qr(constraint_matrix, mode="complete")
    penalty = np.double(
        np.linalg.multi_dot([np.transpose(q[:, 1:]), penalty, q[:, 1:]])
    )
    basis = basis @ q[:, 1:]
    return basis, penalty, q[:, 1:]



def pol2nb(pc):
    """
    Takes a dict of polygons and finds the neighbourhood-structure. (Works by finding possible neighbour candidates
    -> if points are shared, the polygons are neighbours. Function adapted from mgcv pol2nb). The neighbourhood-structure
    functions as the penalty matrix for MRFSmooth.
    :param pc: dict of polygons
    :return: neighbourhood-structure as pd.DataFrame (so I could have named cols and rows)
    """

    num_poly = len(pc)
    lo1 = dict.fromkeys(pc.keys())
    hi1 = dict.fromkeys(pc.keys())
    lo2 = dict.fromkeys(pc.keys())
    hi2 = dict.fromkeys(pc.keys())
    for i in pc.keys():
        lo1[i] = min(pc[i][:, 0])
        lo2[i] = min(pc[i][:, 1])
        hi1[i] = max(pc[i][:, 0])
        hi2[i] = max(pc[i][:, 1])
        pc[i] = np.unique(pc[i], axis=0)

    ids = pc.keys()
    lo1 = list(lo1.values())
    lo2 = list(lo2.values())
    hi1 = list(hi1.values())
    hi2 = list(hi2.values())
    pc = list(pc.values())
    nb = dict.fromkeys(np.arange(0, num_poly))

    for k in range(num_poly):
        ol1 = np.logical_or(np.logical_or(np.logical_and(lo1[k] <= hi1, lo1[k] >= lo1),
              np.logical_and(hi1[k] <= hi1, hi1[k] >= lo1)),
              np.logical_or(np.logical_and(lo1 <= hi1[k], lo1 >= lo1[k]),
              np.logical_and(hi1 <= hi1[k], hi1 >= lo1[k])))
        ol2 = np.logical_or(np.logical_or(np.logical_and(lo2[k] <= hi2, lo2[k] >= lo2),
              np.logical_and(hi2[k] <= hi2, hi2[k] >= lo2)),
              np.logical_or(np.logical_and(lo2 <= hi2[k], lo2 >= lo2[k]),
              np.logical_and(hi2 <= hi2[k], hi2 >= lo2[k])))
        ol = np.logical_and(ol1, ol2)
        ol[k] = False
        ind = np.where(ol)[0]
        cok = pc[k]
        nb[k] = []
        if len(ind) > 0:
            for j in range(len(ind)):
                co = np.vstack([pc[ind[j]], cok])
                cou = np.unique(co, axis=0)
                n_shared = co.shape[0] - cou.shape[0]
                if n_shared > 0:
                    nb[k].append(ind[j])

    nb_mat = np.zeros((len(pc), len(pc)))
    for i in nb.keys():
        nb_mat[i, nb[i]] = -1
        nb_mat[i, i] = len(nb[i])

    nb_df = pd.DataFrame(nb_mat, columns=ids, index=ids)
    return nb_df


def mrf_design(regions, pc):
    """
    Function to create the design matrix for MRFSmooths. Simple indicator matrix.
    :param regions: x
    :param pc: dict of polygons
    :return: design matrix with columns in order in which they are in the dictionary of polygons
    """
    regions = regions.astype('int')
    ids = pc.keys()
    design_mat = np.zeros([len(regions), len(ids)])
    design_df = pd.DataFrame(design_mat, columns=ids)
    for i in range(0, len(regions)):
        print(i, design_df.shape)
        design_df.loc[i, regions[i]] = 1
    design_mat = design_df.to_numpy()
    return design_mat





def color_fader(c_1,c_2,mix=0):
    """
    Function that takes to colors as inputs and mixes them as defined by mix [0, 1]. If the input is an array, function
    returns a list of color codes, else just a single color code. (Necessary for MRFSmooth plot method)
    :param c_1: color 1
    :param c_2: color 2
    :param mix: value between 0 and 1
    :return:
    """
    c_1 = np.array(mpl.colors.to_rgb(c_1))
    c_2 = np.array(mpl.colors.to_rgb(c_2))
    if isinstance(mix, np.ndarray):
        cols = []
        for i in range(len(mix)):
            cols.append(mpl.colors.to_hex((1-mix[i]) * c_1 + mix[i] * c_2))
        return cols

    else:
        return mpl.colors.to_hex((1-mix) * c_1 + mix * c_2)


def color_bounds(values):
    """
    Also function for plotting MRFSmooths: Finds max and min of provided values and creates and maps the interval
    between the two on the interval 0 to 1. Allows me to create a colorbar with the correct ticks.
    :param values: estimated parameters of MRFSmooth
    :return: m
    """
    interval = np.linspace(0, 1, 100)
    min_v = min(values)[0]
    max_v = max(values)[0]
    mapped = min_v + ((max_v - min_v) / 1 - 0) * interval
    return mapped


def tp_T(data, M, m, d):

    """
    function to get the polynomials of the features for which the penalty is null.
    Currently calls a c-function from mgcv which returns the polynomial powers of
    the M functions. The returned values are than used to transform the data
    :param data: data
    :param M: size of nullspace
    :param m: penalty order
    :param d: dimensions of data
    :return:
    """

    # call poly_powers to get the polynomial powers with which to evaluate x
    powers = poly_powers(m, d, M)
    n = data.shape[0]
    T = np.zeros((n, M))

    # loop through row of powers
    for i in range(M):
        T[:, i] = np.prod(data ** powers[i, :], axis=1)

    return T

def poly_powers(m, d, M):
    """
    one to one translation from a function in mgcv that creates an M x d matrix
    with the polynomial powers needed for model matrix T
    Parameters
    ----------
    m: penalty order
    d: dimensions
    M: nullspace dim

    Returns matrix of polynomial powers at which to evaluate data.
    One to one from mgcv (https://github.com/cran/mgcv/blob/master/src/tprs.c: gen_tps_poly_powers)
    -------

    """

    powers = np.zeros((M, d))
    index = np.zeros(d)
    for i in range(M):
        for j in range(d):
            powers[i, j] = index[j]
        sum = 0
        for j in range(d):
            sum += index[j]
        if sum < (m-1):
            index[0] += 1
        else:
            sum -= index[0]
            index[0] = 0
            for j in range(1, d):
                index[j] += 1
                sum += 1
                if sum == m:
                    sum -= index[j]
                    index[j] = 0
                else:
                    break
    return powers








