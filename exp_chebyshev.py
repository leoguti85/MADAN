from pygsp import graphs, filters, plotting
import networkx as nx
import numpy as np
import scipy
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from chebychev import cheby_op
from utils import tic, toc
"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""
np.random.seed(42)



def processInput(G,coeff,i):
	
	delta     = np.zeros(G.N, dtype=int)
	delta[i]  = 1

	#Chebyshev polynomial of graph Laplacian applied to vector.
	f_prime   = cheby_op(G, coeff, delta)

	concent_i = np.linalg.norm(f_prime)
	
	return np.insert(f_prime,0,concent_i)

#-------------------------------------------------------------------------------------



def compute_fast_exponential(G,t):
	#tic()
	G.estimate_lmax()

	f      = filters.Heat(G, tau=t)

	#Compute Chebyshev coefficients for a Filterbank.
	coeff  = filters.compute_cheby_coeff(f, m=30)
	result = Parallel(n_jobs=8)(delayed(processInput)(G, coeff,i) for i in range(0,G.N))
	result = np.array(result)
	#toc()
	return result[:,0]


def chebychev_sequential(G,t):

	#tic()
	f      =  filters.Heat(G, tau=t)
	coeff  =  filters.compute_cheby_coeff(f, m=30)
	
	res = []
	for i in range(0,G.N):
		delta     = np.zeros(G.N, dtype=int)
		delta[i]  = 1
		f_prime   = cheby_op(G, coeff, delta)
		concent_i = np.linalg.norm(f_prime)
		res.append(concent_i)
	
	#toc_time   = toc()
	
	return np.array(res)