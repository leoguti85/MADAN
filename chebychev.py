import numpy as np
from scipy import sparse

def cheby_op(G, c, signal, **kwargs):
    """
    Chebyshev polynomial of graph Laplacian applied to vector.

    Parameters
    ----------
    G : Graph
    c : ndarray or list of ndarrays
        Chebyshev coefficients for a Filter or a Filterbank
    signal : ndarray
        Signal to filter

    Returns
    -------
    r : ndarray
        Result of the filtering

    """

    if not isinstance(c, np.ndarray):
        c = np.array(c)

    c = np.atleast_2d(c)
    Nscales, M = c.shape

    if M < 2:
        raise TypeError("The coefficients have an invalid shape")

    # thanks to that, we can also have 1d signal.
    try:
        Nv = np.shape(signal)[1]
        r = np.zeros((G.N * Nscales, Nv))
    except IndexError:
        r = np.zeros((G.N * Nscales))

    a_arange = [0, G.lmax]

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.

    twf_old = signal
    #twf_cur = (G.L[:,i].toarray().ravel() - a2 * signal) / a1
    twf_cur = (G.L.dot(signal) - a2 * signal) / a1


    tmpN = np.arange(G.N, dtype=int)
    
    for i in range(Nscales):
    
        r[tmpN + G.N*i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

    factor = 2/a1 * (G.L - a2 * sparse.eye(G.N))
    for k in range(2, M):
        twf_new = factor.dot(twf_cur) - twf_old
       
        for i in range(Nscales):
            r[tmpN + G.N*i] += c[i, k] * twf_new

        twf_old = twf_cur
        twf_cur = twf_new
   
    return r

