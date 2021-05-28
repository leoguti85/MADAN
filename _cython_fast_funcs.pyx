#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:03:14 2019

@author: alex

fast cython functions to evaluate changes in stability when moving nodes in 
the Louvain algorithm

"""

import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def sum_Sto(double[:, ::1] S , int k, list ix_cf):
    
    cdef double delta_r = 0
    cdef Py_ssize_t i
    
    for i in ix_cf:
        delta_r += S[k,i]
        delta_r += S[i,k]
    delta_r += S[k,k]
        
    return delta_r

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def sum_Sout(double[:, ::1] S , int k, list ix_ci):
    
    cdef double delta_r = 0
    cdef Py_ssize_t i
    
    for i in ix_ci:
        delta_r -= S[k,i]
        delta_r -= S[i,k]
    delta_r += S[k,k]
        
    return delta_r

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def compute_S(double[:] p1, double[:] p2, double[:,:] T):
    
    cdef Py_ssize_t imax = T.shape[0]
    cdef Py_ssize_t jmax = T.shape[1]
    
    S = np.zeros((imax,jmax), dtype=np.float64)
    cdef double[:,:] S_view = S
    
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    
    for i in range(imax):
        for j in range(jmax):
            S_view[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
    
    return S

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def cython_inplace_csr_row_normalize(double[:] X_data,
                                     int [:] X_indptr,
                                     Py_ssize_t n_row):
    """ row normalize scipy sparse csr matrices inplace.
        inspired from sklearn sparsefuncs_fast.pyx.
        
        Assumes that X_data has only positive values
        
        Call:
        -----
        cython_inplace_csr_row_normalize(double[:] X_data, Py_ssize_t [:] X_indptr, Py_ssize_t n_row)
        
    """
    
    cdef int i, j
    cdef double sum_

    for i in range(n_row):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += X_data[j]

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing            
def cython_nmi(list clusters1, list clusters2, int N, int n1, int n2):
    """ Computes normalized mutual information
    
        Call:
        -----
        
        cython_nmi(list clusters1, list clusters2, int N, int n1, int n2)
    
    """
    # loop over pairs of clusters
    cdef double[:] p1 = np.zeros(n1, dtype=np.float64) # probs to belong to clust1
    cdef double[:] p2 = np.zeros(n2, dtype=np.float64)
    cdef double[:] p12 = np.zeros(n1*n2, dtype=np.float64) # probs to belong to clust1 & clust2
    cdef double H1 = 0.0
    cdef double H2 = 0.0
    cdef double H12 = 0.0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef set clust1
    cdef set clust2
    
    
    for i,clust1 in enumerate(clusters1):
        p1[i] = len(clust1)/N
        for j, clust2 in enumerate(clusters2):
            p12[k] = len(clust1.intersection(clust2))/N
            k += 1
    for j, clust2 in enumerate(clusters2):
        p2[j] = len(clust2)/N
        
    # Shannon entropies
    for i in range(n1):
        if p1[i] != 0:
            H1 -= p1[i]*np.log2(p1[i])
    for j in range(n2):
        if p2[j] != 0:
            H2 -= p2[j]*np.log2(p2[j])
    for j in range(n1*n2):
        if p12[j] != 0:
            H12 -= p12[j]*np.log2(p12[j])


    return (H1 + H2 - H12)/max((H1,H2))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing            
def cython_nvi(list clusters1, list clusters2, int N, int n1, int n2):
    """ Computes normalized variation of information
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N, int n1, int n2)
    
    """
    # loop over pairs of clusters
    cdef double[:] p1 = np.zeros(n1, dtype=np.float64) # probs to belong to clust1
    cdef double[:] p2 = np.zeros(n2, dtype=np.float64)
    cdef double[:,:] p12 = np.zeros((n1,n2), dtype=np.float64) # probs to belong to clust1 & clust2
    cdef double H1c2 = 0.0
    cdef double H2c1 = 0.0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef set clust1
    cdef set clust2
    
    
    for i,clust1 in enumerate(clusters1):
        p1[i] = len(clust1)/N
        for j, clust2 in enumerate(clusters2):
            p12[i,j] = len(clust1.intersection(clust2))/N

    for j, clust2 in enumerate(clusters2):
        p2[j] = len(clust2)/N
        
            
    # Conditional Shannon entropies
    for i in range(n1):
        for j in range(n2):
            if p12[i,j] > 0:
                H1c2 -= p12[i,j]*np.log2(p12[i,j]/p2[j])
                H2c1 -= p12[i,j]*np.log2(p12[i,j]/p1[i])
        
    return (H1c2+H2c1)/np.log2(N)


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing   
def cython_rebuild_nnz_rowcol(double[:] T_data,
                               int [:] T_indices,
                               int [:] T_indptr,
                               long [:] zero_indices):
    """ returns a CSR matrix (data,indices,rownnz, shape) built from the CSR 
        matrix T_small but with
        added row-colums at zero_indicies (with 1 on the diagonal)
        
        Call:
        -----
        (data, indices, indptr, n_rows) = cython_rebuild_nnz_rowcol(double[:] T_data,
                                                                   int [:] T_indices,
                                                                   int [:] T_indptr,
                                                                   int [:] zero_indices)
                                    
    """
    
    cdef Py_ssize_t row_id_small_t = -1
    cdef Py_ssize_t row_id = 0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t i = 0   
    cdef Py_ssize_t data_ind = 0   
    cdef int n_rows = T_indptr.size-1 + zero_indices.size
    cdef int num_data_row, row_start, row_end

    
    cdef double[:] data = np.zeros(T_data.size+zero_indices.size,
                                   dtype=np.float64)
    cdef int[:] indices = np.zeros(T_data.size+zero_indices.size,
                                   dtype=np.int32)
    cdef int[:] indptr = np.zeros(n_rows+1,
                                   dtype=np.int32)
    cdef int[:] new_col_inds = np.zeros(T_indptr.size-1,
                                   dtype=np.int32)
    cdef int[:] Ts_indices = np.zeros(T_indices.size,
                                   dtype=np.int32)
    cdef set zero_set = set(zero_indices)
    

    # map col indices to new positions
    k = 0
    for i in range(n_rows):
        if i not in zero_set:
            new_col_inds[k] = i
            k +=1
                        
    for k,i in enumerate(T_indices):
        Ts_indices[k] = new_col_inds[i]
    
    row_id_small_t = -1
    data_ind = 0
    for row_id in range(n_rows):
        row_id_small_t +=1
        if row_id in zero_set:
            # add a row with just 1 on the diagonal
            data[data_ind] = 1.0
            indices[data_ind] = row_id
            indptr[row_id+1] = indptr[row_id]+1
            
            row_id_small_t -= 1
            data_ind += 1
            
        else:
            row_start = T_indptr[row_id_small_t]    
            row_end = T_indptr[row_id_small_t+1]  
            
            num_data_row = row_end - row_start
            
            data[data_ind:data_ind+num_data_row] = T_data[row_start:row_end]
            indices[data_ind:data_ind+num_data_row] = Ts_indices[row_start:row_end]
            indptr[row_id+1] = indptr[row_id]+num_data_row
            
            data_ind += num_data_row
            

    return (data, indices, indptr, n_rows)
            