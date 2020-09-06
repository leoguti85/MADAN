#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:17:12 2018

@author: alex

"""

from copy import deepcopy
import numpy as np
from scipy.optimize import linear_sum_assignment



USE_CYTHON = True
try:
    from _cython_fast_funcs import sum_Sto, sum_Sout, compute_S, cython_nmi, cython_nvi
except Exception as e:
    print('Could not load cython functions')
    print(e)
    USE_CYTHON = False


class Partition(object):
    """ A node partition that can be described as a list of node sets 
        and a node to cluster dict.
        
    """
    def __init__(self, num_nodes,
                 cluster_list=None, 
                 node_to_cluster_dict=None,
                 check_integrity=False):
        
        self.num_nodes = num_nodes
        
        if cluster_list is None and node_to_cluster_dict is None:
            # default clustering is one node per cluster
            self.cluster_list = [set([n]) for n in range(self.num_nodes)]
            self.node_to_cluster_dict = {n : n for n in range(self.num_nodes)}
            
        elif cluster_list is not None and node_to_cluster_dict is None:
            self.cluster_list = cluster_list
            self.node_to_cluster_dict = {}
            for i, clust in enumerate(self.cluster_list):
                for node in clust:
                    self.node_to_cluster_dict[node] = i
    
        elif cluster_list is None and node_to_cluster_dict is not None:
            self.node_to_cluster_dict = node_to_cluster_dict
            self.cluster_list = [set() for _ in \
                                 range(max(node_to_cluster_dict.values()) + 1)]            
            for node, clust in node_to_cluster_dict.items():
                self.cluster_list[clust].add(node)
                
        elif cluster_list is not None and node_to_cluster_dict is not None:
            raise ValueError('cluster_list and node_to_cluster_dict ' +\
                             'cannot be provided together')
        
        self.remove_empty_clusters()
        if check_integrity:
            self.check_integrity()
        
        
    def move_node(self,node, c_f):
        """ moves a node to cluster `c_f` """
        
        # initial cluster
        c_i = self.node_to_cluster_dict[node]
        
        if c_i != c_f:
            self.node_to_cluster_dict[node] = c_f
            self.cluster_list[c_i].remove(node)
            self.cluster_list[c_f].add(node)
        else:
            print('Warning, node is already in {0}'.format(c_f))
            
            
    def remove_empty_clusters(self):
        """ removes empty clusters from `cluster_list` and  `node_to_cluster_dict`
            and reindexes clusters."""
            
        self.cluster_list = [c for c in self.cluster_list if len(c) > 0 ]
        self.node_to_cluster_dict = {}
        for i, clust in enumerate(self.cluster_list):
            for node in clust:
                self.node_to_cluster_dict[node] = i
    
    def get_num_clusters(self):
        
        return len(self.cluster_list)
    
    def get_indicator(self):
        """returns an `N x c` indicator matrix `H` such that each row of `H`
            is all zeros except for a one indicating the cluster to which
            it belongs:
                
                `h_ik = 1` iff node i is in cluster k, and zero otherwise.
        """
        
        H = np.zeros((self.num_nodes,self.get_num_clusters()),
                     dtype=np.int)
        
        for i,c in enumerate(self.cluster_list):
            H[list(c),i] = 1
            
        return H
    
    def iter_cluster_node_index(self):
        """ returns an iterator giving a list of node indices for each cluster"""
        for c in self.cluster_list:
            yield list(c)
            
    def __str__(self):
        
        return str(self.cluster_list)
    
    def check_integrity(self):
        """ check that all nodes are in only one cluster, 
            i.e. non-overlapping clusters whose union is the full set of nodes."""
            
            
        inter = set()
        total_len = 0
        num_clusters = self.get_num_clusters()
        for i in range(num_clusters):
            total_len += len(self.cluster_list[i])
            for j in range(num_clusters):
                if i != j:
                    inter.update(self.cluster_list[i].intersection(self.cluster_list[j]))
        
        if len(inter) > 0:
            raise ValueError('Overlapping clusters.')
        
        if total_len != self.num_nodes:
            raise ValueError('Some nodes have no clusters.')
        

        
class BaseClustering(object):
    """ BaseClustering.
        
        Parameters:
        ----------- 
           
        `T`, `p1`, `p2` and `S` must be numpy.ndarrays.
        
        
        - if `S` is not given, it is computed as diag(p1) @ T - outer(p1,p2).
        
        Clusters can either be initilized with
        
        - `source_cluster_list` : a list of set of nodes or
        - `source_node_to_cluster_dict` : a dictionary with mapping between nodes and cluster number.
        
        - `target_cluster_list` : a list of set of nodes or
        - `target_node_to_cluster_dict` : a dictionary with mapping between nodes and cluster number.
        
        Default is each node in a different cluster.
        
        The random number generator can be specified either with a:
            
        - `rnd_state`, a initialized `np.random.RandomState` object or
        - `rnd_seed`, a random seed for a RandomState object.
        
        Deflaut is to initialize a RandomState with a random seed.
        

    """

    def __init__(self, T=None, p1=None, p2=None, S=None,
                       source_cluster_list=None,
                       source_node_to_cluster_dict=None,
                       target_cluster_list=None, 
                       target_node_to_cluster_dict=None,
                       rnd_state=None, rnd_seed=None,):
  
        if T is None:
            raise ValueError('T must be provided')
        
        self.num_nodes = T.shape[0]
    
        if p1 is None:
            # uniform distribution
            p1 = np.ones(T.shape[0])/T.shape[0]
            
        if p2 is None:
            p2 = p1 @ T
    
        if not (isinstance(T,np.ndarray) and \
                isinstance(p1, np.ndarray) and \
                isinstance(p2, np.ndarray)):
            raise ValueError('p1, p2 and T must be numpy arrays.')

        if isinstance(T,np.matrix) or \
                isinstance(p1, np.matrix) or \
                isinstance(p2, np.matrix):
            raise ValueError('p1, p2 and T must be numpy arrays, not numpy matrices.')            
        
      
        self.p1 = p1
        
        self.T = T.copy()
        
        self.p2 = p2
        
        if S is None:
            # compute stability matrix
            self._compute_S()
        else:
            if not isinstance(S, np.ndarray):
                raise ValueError('S must be numpy arrays.')
            if isinstance(S,np.matrix):
                raise ValueError('S must be numpy arrays, not numpy matrices.')           
                
            assert S.shape == self.T.shape, 'T and S must have the same shape.'
            self._S = S.copy()
            

        # initialize clusters        
        self.source_part = Partition(self.num_nodes, 
                                     cluster_list=source_cluster_list,
                                     node_to_cluster_dict=source_node_to_cluster_dict)
        
        self.target_part = Partition(self.num_nodes, 
                                     cluster_list=target_cluster_list,
                                     node_to_cluster_dict=target_node_to_cluster_dict)
                        
            
        # random number generator
        if rnd_state is not None:
            self._rnd_state = rnd_state
        else:
            self._rnd_state = np.random.RandomState(rnd_seed)
            
            
        # list of out and in neighbors arrays, include potential self loops
        self._out_neighs = []
        self._in_neighs = []
        self._neighs = []
        for node in range(self.num_nodes):
            self._out_neighs.append(np.nonzero(self.T[node,:] > 0)[0].tolist())
            self._in_neighs.append(np.nonzero(self.T[:,node] > 0)[0].tolist())
            self._neighs.append(list(set(self._out_neighs[node] + self._in_neighs[node])))        

         
        

        
    def _compute_S(self):
        
        """ Computes the internal matrix comparing probabilities for each
            node
                    S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
                
            Saves the matrix in `self._S`.
        """
        
        if USE_CYTHON:
            self._S = compute_S(self.p1,self.p2,self.T)
        else:
            self._S = np.diag(self.p1) @ self.T - np.outer(self.p1,self.p2)
        
        return self._S


    
    def _compute_clustered_autocov(self, partition=None):
        """ compute the clustered autocovariance matrix based on `source_part`
            and `target_part`.
            
            `partition` is a tuple with `(source_part, target_part)`.
            
            Default partitions are `self.source_part` and `self.target_part`."""
        
        
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition
        
        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()

        R = np.zeros((num_s_clusters,num_t_clusters))
        
        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}
        
        for s in range(num_s_clusters):
            if len(source_part.cluster_list[s]) > 0:
                for t in range(num_t_clusters):
                    if len(target_part.cluster_list[t]) > 0 :
                        # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                        R[s,t] = self._S[s_cluster_to_node_list[s],
                                         t_cluster_to_node_list[t]].sum()
                        
        return R
    
    @staticmethod
    def _find_optimal_flow(R):
        """ for a given clustered autocovariance matrix `R`, returns the optimal
            flow, i.e. the perfect matching between clusters that give the maximum
            stability
            
            Returns
            -------
            
            flow_stab, flow_map, flow_map_inv
            
        """
        
        row_ind, col_ind = linear_sum_assignment(1-R)
        
        return (R[row_ind, col_ind].sum(),
                {s:t for s,t in zip(row_ind,col_ind)},
                {t:s for s,t in zip(row_ind,col_ind)})
        

    def compute_stability(self, R=None):
        """ returns the stability 
            
        """
        
        raise NotImplementedError

    
    def _compute_new_R_moveto(self, k, c_i, 
                                       c_f,
                                       Rold,
                                       partition=None):
        
        """ return the new clustered autocov matrix obtained from `Rold` 
            by moving node `k` from `c_i = (c_i_s,c_i_t)` into bi-cluster 
            `c_f = (c_f_s,c_f_t)`.
            
            If given, the list of original clusters is given by 
            `partition = (source_part, target_part)`.
            Otherwise is taken from `self.source_part` and `self.target_part`.
            
            `Rold` should be the output of `_compute_delta_R_moveout`.
            
            `c_f` may be an empty cluster
        """
        
        c_i_s, c_i_t = c_i
        c_f_s, c_f_t = c_f
        
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition
            
        Rnew = Rold.copy()
        
        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()    

        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}
        
        for s in range(num_s_clusters):
            Rnew[s,c_f_t] +=  self._S[s_cluster_to_node_list[s],k].sum()
        
        for t in range(num_t_clusters):
            Rnew[c_f_s,t] +=  self._S[k,t_cluster_to_node_list[t]].sum()
        
        # we remove Skk from the c_i because it's not there anymore
        Rnew[c_i_s,c_f_t] -= self._S[k,k]
        Rnew[c_f_s,c_i_t] -= self._S[k,k]
        
        # we add the diagonal because it was not in the sums
        Rnew[c_f_s,c_f_t] += self._S[k,k]
        
        return Rnew
    
    
    def _compute_new_R_moveout(self, k, c_i,
                                 partition=None,
                                 Rold=None):
        
        """ return the  new clustered autocov matrix obtained from `Rold` 
            by moving node k out of the bi-cluster `c_i = (c_i_s, c_i_t)`.
            
            If given, the list of original clusters is given by 
            `partition = (source_part, target_part)`.
            Otherwise is taken from `self.source_part` and `self.target_part`.
            
            If `Rold` is not given, it will be recomputed.
            
            c_i is assumed to be non-empty!
            
        """

        c_i_s, c_i_t = c_i
        
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition
            
        if Rold is None:
            Rold = self._compute_clustered_autocov(partition=(source_part,target_part))

        Rnew = Rold.copy()
            
        if k not in source_part.cluster_list[c_i_s] or \
           k not in target_part.cluster_list[c_i_t]:
            raise ValueError('node k must be in bicluster (c_i_s, c_i_t)')

        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()            


        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                            enumerate(target_part.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}

        
        for s in range(num_s_clusters):
            Rnew[s,c_i_t] -=  self._S[s_cluster_to_node_list[s],k].sum()
        
        for t in range(num_t_clusters):
            Rnew[c_i_s,t] -=  self._S[k,t_cluster_to_node_list[t]].sum()
            
        # we add S[k,k] because it was counted twice in the sums            
        Rnew[c_i_s,c_i_t] += self._S[k,k]
                   
                   
        return Rnew    
    
    def _potential_new_clusters(self, node):
        """ returns a set of potential source and target clusters where to move
            `node`.
        """
        
        raise NotImplementedError
        
        
    
    def _compute_subsets_connectedness(self, s1, s2):
        
        """ `s1` and `s2` must be two lists of nodes.
        
            returns
            
            .. math::
                \sum_{i\in s_1}\sum_{j\in s_2} S_{i,j}
                
            The fraction of walkers going from s1 to s2 minus the expected
            value of the same quantity.                
            
        """
        #reshape for correct indexing
        s1 = [[i] for i in s1]
            
        return self._S[s1,s2].sum()


    def _louvain_move_nodes(self, 
                           delta_r_threshold=np.finfo(float).eps,
                           verbose=False,):
        """ return delta_r_tot, n_loop
        
        """
        
        delta_r_tot = 0
        delta_r_loop = 1
        n_loop = 1
        

        R = self._compute_clustered_autocov()
        
        stability = self.compute_stability(R)
        
        while delta_r_loop > delta_r_threshold:
            
            delta_r_loop = 0
            
            if verbose:
                print('\n-------------')
                print('Louvain sub loop number ' + str(n_loop))            
            
            # shuffle order to process the nodes
            node_ids = np.arange(self.num_nodes)
            self._rnd_state.shuffle(node_ids)
                                    
            for node in node_ids:
                # test gain of stability if we move node to neighbours communities
                
                # initial cluster of node
                c_i = self._get_node_cluster(node)

                if verbose >1:
                    print('++ treating node {0} from cluster {1}'.format(node,
                                                                          c_i))
                
                # new R if we move node out of (c_i_s,c_i_t)
                R_out = self._compute_new_R_moveout(node, c_i,
                                                    Rold=R)
                
                # find potential communities where to move node
                comms = self._potential_new_clusters(node)
                
                
                delta_r_best = 0
                c_f_best = c_i
                for c_f in comms:
                    if c_f != c_i:
                        # new R if we move node there
                        Rnew = self._compute_new_R_moveto(node, c_i,
                                                              c_f,
                                                              R_out)
                        # total gain of moving node
                        delta_r = self.compute_stability(Rnew) - stability
                            
                        if verbose >= 10:
                            print(c_f)
                            print(delta_r)
                        # we use `>=` to allow for more mixing (can be useful)
                        if delta_r >= delta_r_best:
                            delta_r_best = delta_r
                            c_f_best = c_f
                            Rnew_best = Rnew
                        
                if c_f_best != c_i:
                    #move node to best_source cluster
                    self._move_node_to_cluster(node,c_f_best)
                    
               
                    delta_r_loop += delta_r_best
                    stability += delta_r_best
                    R = Rnew_best
                    
                    if verbose > 1:
                        print('moved node {0} from cluster {1} to cluster {2}'.format(node,
                                                          c_i,c_f_best)) 

                # else do nothing
                else:
                    if verbose > 1:
                            print('node {0} in clusters ({1}) has not moved'.format(node,
                                                              c_i)) 
                    
                

                
            if verbose:
                print('\ndelta r loop : ' + str(delta_r_loop))
                print('delta r total : ' + str(delta_r_tot))
                print('number of clusters : ' + \
                      str(self._get_num_clusters()))
                if verbose>1:
                    print('** clusters : ')
                    for cl in self._get_cluster_list().items():
                        print('** ', cl)
                    
                if delta_r_loop == 0:
                    print('No changes, exiting.')

            delta_r_tot += delta_r_loop
        
            n_loop += 1
        
        # remove empty clusters
        self._remove_empty_clusters()
        
        return delta_r_tot, n_loop
    
        

    def _aggregate_clusters(self, partition=None):
        """ For each of the c_s source clusters given by `source_part`
            and c_t clusters given by `target_part`, aggregates 
            the corresponding nodes in a new meta node.
            
            `partition=(source_part, target_part)`.
            
            Returns `T`, `p1` and `p2` the corresponding c_sxc_t transition matrix
            a 1xc_s and a 1xc_t probability vector.
         
        """
        
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition
            
        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()

        p1 = np.zeros(num_s_clusters)    
        p2 = np.zeros(num_t_clusters)
        T = np.zeros((num_s_clusters,num_t_clusters))
        S = np.zeros((num_s_clusters,num_t_clusters))
        
        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}
        
        for s in range(num_s_clusters):
            p1[s] = self.p1[s_cluster_to_node_list[s]].sum()
            for t in range(num_t_clusters):
                p2[t] = self.p2[t_cluster_to_node_list[t]].sum()
                
                # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                T[s,t] = self.T[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()    
                
                S[s,t] = self._S[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()    
        
        # renormalize T
        T = T/T.sum(axis=1)[:, np.newaxis]
        #also normalize p1 and p2 for rounding errors
        p1 = p1/p1.sum()
        p2 = p2/p2.sum()
  
        return T, p1, p2, S
            
    def _get_updated_partition(self, og_old_part, meta_new_part):
        """ returns an updated version of original og_old_part corresponding 
            to the meta_new_part"""
            
        og_old_part.remove_empty_clusters()
        meta_new_part.remove_empty_clusters()
        
        new_node_to_cluster_dict = dict()
        # this is the state before moving nodes
        for meta_node, original_nodes in enumerate(og_old_part.cluster_list):
            
            for node in original_nodes:
                
                # this is the state after moving nodes
                new_node_to_cluster_dict[node] = \
                    meta_new_part.node_to_cluster_dict[meta_node]


        return Partition(og_old_part.num_nodes,
                         node_to_cluster_dict=new_node_to_cluster_dict)               
        
    def find_louvain_clustering(self, delta_r_threshold=np.finfo(float).eps,
                                n_iter_max=1000, 
                                verbose=False, 
                                rnd_seed=None,
                                save_progress=False,):
        """returns n_meta_loop
        
           `rnd_seed` is used to set to state of the random number generator.
           Default is to keep the current state
           
        """
        # implement loop control based on delta r difference
        
        # random numbers from seed rnd_seed
        if rnd_seed is not None:
            self._rnd_state.seed(rnd_seed)
                    
        if save_progress:
            self._save_progress()
            
        
        # initial clustering
        cluster_dict = deepcopy(self._get_cluster_list())
        
        meta_clustering = self.__class__(T=self.T, p1=self.p1, p2=self.p2,
                                         rnd_state=self._rnd_state,
                                         **cluster_dict)
                                       
        delta_r_meta_loop = 1
        n_meta_loop = 0
        stop = False
        while not ((delta_r_meta_loop <= delta_r_threshold) or stop or \
                                                   (n_meta_loop > n_iter_max)):
        
            n_meta_loop += 1
            
            if verbose:
                print('\n**************')
                print('* Louvain meta loop number ' + str(n_meta_loop))
                
            # sub loop
            delta_r_meta_loop, _= meta_clustering._louvain_move_nodes(
                                                delta_r_threshold, verbose)
            
            
            #update original node partition
            self._update_original_partition(meta_clustering)
                        

            if (all([len(c) == 1 for clust_list in self._get_cluster_list().values() \
                     for c in clust_list])) or \
                   n_meta_loop > n_iter_max or \
                   delta_r_meta_loop <= delta_r_threshold:
                stop = True # we've reached the best partition
            else:
                # cluster aggregation
                
                T, p1, p2, S = self._aggregate_clusters()
                
              
                meta_clustering = self.__class__(T=T, p1=p1, p2=p2, S=S,
                                             rnd_state=self._rnd_state)

            if save_progress:
                self._save_progress()
            
                
            if verbose:
                print('\n*  delta r meta loop : ' + str(delta_r_meta_loop))
                print('*  number of clusters : ' + \
                      str(self._get_num_clusters()))

                if verbose>1:
                    print('** clusters : ')
                    for cl in self._get_cluster_list().items():
                        print('** ', cl)
                print('*  end of meta loop num : ', n_meta_loop)
                
            
        return n_meta_loop
    
    
    def _save_progress(self):
        if not hasattr(self,'source_part_progress'):
            self.source_part_progress = []
        if not hasattr(self,'target_part_progress'):
            self.target_part_progress = []
            
        if not hasattr(self, 'stability_progress'):
            self.stability_progress = []
            
        self.source_part_progress.append(deepcopy(self.source_part))
        self.target_part_progress.append(deepcopy(self.target_part))
        
        self.stability_progress.append(self.compute_stability())

    def _get_node_cluster(self, node):
        """ returns the bi-cluster `(c_s, c_t)` in which `node` is. """
        
        raise NotImplementedError
        
        
    def _move_node_to_cluster(self, node, c):
        """ move `node` to the bi-cluster `c`. """
        
        raise NotImplementedError
        
    def _get_cluster_list(self):
        """ returns a dictionary of source and target clusters lists. """
                
        raise NotImplementedError
    
    def _get_num_clusters(self):
        """ returns a tuple with the number of source and target clusters. """
                
        raise NotImplementedError

    def _remove_empty_clusters(self):
        """ remove empty source and target clusters. """
        
        self.source_part.remove_empty_clusters()
        self.target_part.remove_empty_clusters()
        
    def _update_original_partition(self, meta_clustering):
        """ update the partion of `self` based on `meta_clustering` """
            
        self.target_part = self._get_updated_partition(self.target_part,
                                                     meta_clustering.target_part)
                       
        self.source_part = self._get_updated_partition(self.source_part,
                                                     meta_clustering.source_part)        
        
        

class Clustering(BaseClustering):
    """ Symmetric Clustering.
    
        Finds the best partition that optimizes the stability 
        defined as the trace of the clustered autocovariance matrix.
        
        
        Parameters:
        ----------- 
           
        `T`, `p1`, `p2` and `S` must be numpy.ndarrays.
        
        Clusters can either be initilized with
        
        - `cluster_list` : a list of set of nodes or
        - `node_to_cluster_dict` : a dictionary with mapping between nodes and cluster number.
        
        Default is each node in a different cluster.
        
        The random number generator can be specified either with a:
            
        - `rnd_state`, a initialized `np.random.RandomState` object or
        - `rnd_seed`, a random seed for a RandomState object.
        
        Deflaut is to initialize a RandomState with a random seed.
        

    """
    
    def __init__(self, p1=None, p2=None,T=None, S=None,
                       cluster_list=None, 
                       node_to_cluster_dict=None,
                       rnd_state=None, rnd_seed=None,):
        
        super().__init__(p1=p1, p2=p2,T=T, S=S,
                       source_cluster_list=cluster_list, 
                       source_node_to_cluster_dict=node_to_cluster_dict,
                       target_cluster_list=None, 
                       target_node_to_cluster_dict=None,
                       rnd_state=rnd_state,
                       rnd_seed=rnd_seed,)
        
        # create an alias for source partition since we only need one partition
        self.partition = self.source_part
        self.target_part = None
        
            
    def compute_stability(self, R=None):
        """ returns the stability of the clusters given in `cluster_list` 
            computed between times `t1` and `t2`
            
        """
        
        if R is None:
            R = self._compute_clustered_autocov()
            
        return R.trace()
    
    
    def _compute_clustered_autocov(self, partition=None):
        """ compute the clustered autocovariance matrix based on `partition`.
            
            
            Default partition is `self.source_part`."""
        
        
        if partition is None:
            partition = self.source_part
        
        num_clusters = partition.get_num_clusters()

        R = np.zeros((num_clusters,num_clusters))
        
        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(partition.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}
        
        for s in range(num_clusters):
            if len(partition.cluster_list[s]) > 0:
                for t in range(num_clusters):
                    if len(partition.cluster_list[t]) > 0 :
                        # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                        R[s,t] = self._S[s_cluster_to_node_list[s],
                                         t_cluster_to_node_list[t]].sum()
                        
        return R
    
    def _compute_delta_stab_moveto(self, k, c_f,
                                 partition=None,):
        
        """ return the gain in stability obtained by moving node
            k into community c_f.
            
            If given, the list of original clusters is given by `partition` and
            otherwise is taken from `self.source_part`.
            
            c_f may be an empty cluster
        """
        
        if partition is None:
            partition = self.partition
            
        if k in partition.cluster_list[c_f]:
            raise ValueError('node k must not be in cluster c_f')
            
        # indexes of nodes in c_f
        ix_cf = list(partition.cluster_list[c_f])
        
        # gain in stability from moving node k to community c_f
        if USE_CYTHON:
            delta_r1 = sum_Sto(self._S, k, ix_cf)
        else:
            delta_r1 = self._S[k,ix_cf].sum() \
                        + self._S[ix_cf,k].sum() \
                        + self._S[k,k]
                               
        return delta_r1


    def _compute_delta_stab_moveout(self, k, c_i,
                                 partition=None,):
        
        """ return the gain in stability obtained by moving node
            k out of community c_i.
            
            If given, the list of clusters is given by `partition` and
            otherwise is taken from `self.partition`.
            
            c_i is assumed to be non-empty!
            
        """
        
        if partition is None:
            partition = self.partition
            
        if k not in partition.cluster_list[c_i]:
            raise ValueError('node k must be in cluster c_i')


        # indexes of nodes in c_i
        ix_ci = list(partition.cluster_list[c_i])
        
        # gain in stability from moving node k out of community c_i
        if USE_CYTHON:
            delta_r2 = sum_Sout(self._S, k, ix_ci)
        else:
            delta_r2 = - self._S[k,ix_ci].sum() \
                       - self._S[ix_ci,k].sum() \
                       + self._S[k,k]
                   # we add S[k,k] because it was counted twice in the sums
                   
        return delta_r2        
    
    
    def _louvain_move_nodes(self, 
                           delta_r_threshold=np.finfo(float).eps,
                           verbose=False,):
        """ return delta_r_tot, n_loop
        
        """
        
        delta_r_tot = 0
        delta_r_loop = 1
        n_loop = 1
        
        
        while delta_r_loop > delta_r_threshold:
            
            delta_r_loop = 0
            
            if verbose:
                print('\n-------------')
                print('Louvain sub loop number ' + str(n_loop))            
            
            # shuffle order to process the nodes
            node_ids = np.arange(self.num_nodes)
            self._rnd_state.shuffle(node_ids)
                                    
            for node in node_ids:
                # test gain of stability if we move node to neighbours communities
                
                # initial cluster of node
                c_i = self._get_node_cluster(node)

                if verbose >1:
                    print('++ treating node {0} from cluster {1}'.format(node,
                                                                          c_i))
                
                # delta stab if we move node out of (c_i_s,c_i_t)
                r_out = self._compute_delta_stab_moveout(node, c_i)
                
                # find potential communities where to move node
                comms = self._potential_new_clusters(node)
                
                
                delta_r_best = 0
                c_f_best = c_i
                for c_f in comms:
                    if c_f != c_i:
                        # new delta r if we move node there
                        r_in = self._compute_delta_stab_moveto(node, c_f)
                        # total gain of moving node
                        delta_r = r_out + r_in
                            
                        if verbose >= 10:
                            print(c_f)
                            print(delta_r)
                        # we use `>=` to allow for more mixing (can be useful)
                        if delta_r >= delta_r_best:
                            delta_r_best = delta_r
                            c_f_best = c_f
                        
                if c_f_best != c_i:
                    #move node to best_source cluster
                    self._move_node_to_cluster(node,c_f_best)
                    
               
                    delta_r_loop += delta_r_best
                    
                    if verbose > 1:
                        print('moved node {0} from cluster {1} to cluster {2}'.format(node,
                                                          c_i,c_f_best)) 

                # else do nothing
                else:
                    if verbose > 1:
                            print('node {0} in clusters ({1}) has not moved'.format(node,
                                                              c_i)) 
                    
                

                
            if verbose:
                print('\ndelta r loop : ' + str(delta_r_loop))
                print('delta r total : ' + str(delta_r_tot))
                print('number of clusters : ' + \
                      str(self._get_num_clusters()))
                if verbose>1:
                    print('** clusters : ')
                    for cl in self._get_cluster_list().items():
                        print('** ', cl)
                    
                if delta_r_loop == 0:
                    print('No changes, exiting.')

            delta_r_tot += delta_r_loop
        
            n_loop += 1
        
        # remove empty clusters
        self._remove_empty_clusters()
        
        return delta_r_tot, n_loop    
    
    def _potential_new_clusters(self, node):
        """ returns a set of potential source and target clusters where to move
            `node`.
            
            The current cluster of node is not included.
        """
        
        comms = {self.partition.node_to_cluster_dict[neigh] for \
                                                neigh in self._neighs[node]}
        
        # remove initial community if it is there
        comms.discard(self.partition.node_to_cluster_dict[node])
        
        return comms
    
    def _compute_new_R_moveto(self, k, c_i, 
                                       c_f,
                                       Rold,
                                       partition=None):
        
        """ return the new clustered autocov matrix obtained from `Rold` 
            by moving node `k` from `c_i` into cluster 
            `c_f `.
            
            If given, the list of original clusters is given by 
            `partition`.
            Otherwise is taken from `self.source_part`.
            
            `Rold` should be the output of `_compute_delta_R_moveout`.
            
            `c_f` may be an empty cluster
        """
        
        
        if partition is None:
            partition = self.source_part
            
        Rnew = Rold.copy()
        
        num_clusters = partition.get_num_clusters() 

        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                                enumerate(partition.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}
        
        for s in range(num_clusters):
            Rnew[s,c_f] +=  self._S[s_cluster_to_node_list[s],k].sum()
        
        for t in range(num_clusters):
            Rnew[c_f,t] +=  self._S[k,t_cluster_to_node_list[t]].sum()
        
        # we remove Skk from the c_i because it's not there anymore
        Rnew[c_i,c_f] -= self._S[k,k]
        Rnew[c_f,c_i] -= self._S[k,k]
        
        # we add the diagonal because it was not in the sums
        Rnew[c_f,c_f] += self._S[k,k]
        
        return Rnew
    
    def _compute_new_R_moveout(self, k, c_i,
                                 partition=None,
                                 Rold=None):
        
        """ return the  new clustered autocov matrix obtained from `Rold` 
            by moving node k out of the cluster `c_i`.
            
            If given, the list of original clusters is given by 
            `partition`.
            Otherwise is taken from `self.source_part`.
            
            If `Rold` is not given, it will be recomputed.
            
            c_i is assumed to be non-empty!
            
        """

        
        if partition is None:
            partition = self.source_part

            
        if Rold is None:
            Rold = self._compute_clustered_autocov(partition=partition)

        Rnew = Rold.copy()
            
        if k not in partition.cluster_list[c_i]:
            raise ValueError('node k must be in cluster c_i')

        num_clusters = partition.get_num_clusters()          


        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}

        
        for s in range(num_clusters):
            Rnew[s,c_i] -=  self._S[s_cluster_to_node_list[s],k].sum()
        
        for t in range(num_clusters):
            Rnew[c_i,t] -=  self._S[k,t_cluster_to_node_list[t]].sum()
            
        # we add S[k,k] because it was counted twice in the sums            
        Rnew[c_i,c_i] += self._S[k,k]
                   
                   
        return Rnew        
    
    def _aggregate_clusters(self, partition=None):
        """ For each of the c clusters given by `partition`, aggregates 
            the corresponding nodes in a new meta node.
            
            Default `partition` is `self.source_part`.
            
            Returns `T`, `p1`, `p2` and `S` the corresponding cxc transition matrix
            a 1xc, a 1xc probability vector and the cxc covariance matrix.
         
        """
        
        if partition is None:
            partition = self.source_part

        num_clusters = partition.get_num_clusters()

        p1 = np.zeros(num_clusters)    
        p2 = np.zeros(num_clusters)
        T = np.zeros((num_clusters,num_clusters))
        S = np.zeros((num_clusters,num_clusters))
        
        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(partition.iter_cluster_node_index())}
    
        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}
        
        for s in range(num_clusters):
            p1[s] = self.p1[s_cluster_to_node_list[s]].sum()
            for t in range(num_clusters):
                p2[t] = self.p2[t_cluster_to_node_list[t]].sum()
                
                # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                T[s,t] = self.T[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()        
                S[s,t] = self._S[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()        
        # renormalize T
        T = T/T.sum(axis=1)[:, np.newaxis]
        #also normalize p1 and p2 for rounding errors
        p1 = p1/p1.sum()
        p2 = p2/p2.sum()
  
        return T, p1, p2, S
    
    def _save_progress(self):
        if not hasattr(self,'partition_progress'):
            self.partition_progress = []
            
        if not hasattr(self, 'stability_progress'):
            self.stability_progress = []
            
        self.partition_progress.append(deepcopy(self.source_part))
        
        self.stability_progress.append(self.compute_stability())

    def _get_node_cluster(self, node):
        """ returns the bi-cluster `(c_s, c_t)` in which `node` is. """
        
        return self.source_part.node_to_cluster_dict[node]

        
    def _move_node_to_cluster(self, node, c):
        """ move `node` to the bi-cluster `c`. """
        
        self.source_part.move_node(node, c)
        
    def _get_cluster_list(self):
        """ returns a dictionary of source and target clusters lists. """
                
        return {'cluster_list' : self.source_part.cluster_list}
    
    def _get_num_clusters(self):
        """ returns a tuple with the number of source and target clusters. """
                
        return self.source_part.get_num_clusters()

    def _remove_empty_clusters(self):
        """ remove empty source and target clusters. """
        
        self.source_part.remove_empty_clusters()
        
    def _update_original_partition(self, meta_clustering):
        """ update the partion of `self` based on `meta_clustering` """
            
        self.source_part = self._get_updated_partition(self.source_part,
                                                     meta_clustering.source_part)
        
    def find_louvain_clustering(self, delta_r_threshold=np.finfo(float).eps,
                                n_iter_max=1000, 
                                verbose=False, 
                                rnd_seed=None,
                                save_progress=False,):
        """returns n_meta_loop
        
           `rnd_seed` is used to set to state of the random number generator.
           Default is to keep the current state
           
        """
        
        n_loop = super().find_louvain_clustering(delta_r_threshold=delta_r_threshold,
                                n_iter_max=n_iter_max, 
                                verbose=verbose, 
                                rnd_seed=rnd_seed,
                                save_progress=save_progress,)
        
        self.partition = self.source_part
        self.target_part = None
        
        return n_loop
        
        

            
def norm_mutual_information(clusters1, clusters2):
    """ returns the normalized mutial information between two
        non-overlapping clustering.
        
        The mutual information is normalized by the max of the 
        two individual entropies.
        
        .. math::
            NMI = (H(C1)+H(C2)-H(C1,C2))/max(H(C1),H(C2))
    
        inputs can be node_to_cluster dictionaries, cluster lists of node sets
        or instances of Partition.
    """
    # convert to list of sets
    if isinstance(clusters1, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters1.values()) + 1)]            
        for node, clust in clusters1.items():
                cluster_list[clust].add(node)
        clusters1 = cluster_list
         
    if isinstance(clusters2, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters2.values()) + 1)]            
        for node, clust in clusters2.items():
                cluster_list[clust].add(node)
        clusters2 = cluster_list      
        
    if isinstance(clusters1, Partition):
        clusters1 = clusters1.cluster_list
    
    if isinstance(clusters2, Partition):
        clusters2 = clusters2.cluster_list
        
    # num nodes
    N = sum(len(clust) for clust in clusters1)
    n1 = len(clusters1)
    n2 = len(clusters2)
    
    if USE_CYTHON:
        return cython_nmi(clusters1, clusters2, N, n1, n2)
    else:
        # loop over pairs of clusters
        p1 = np.zeros(n1) # probs to belong to clust1
        p12 = np.zeros(n1*n2) # probs to belong to clust1 & clust2
        k = 0
        for i,clust1 in enumerate(clusters1):
            p1[i] = len(clust1)/N
            for j, clust2 in enumerate(clusters2):
                p12[k] = len(clust1.intersection(clust2))/N
                k += 1
        
        p2 = np.array([len(clust2)/N for clust2 in clusters2])
        
        # Shannon entropies
        H1 = - np.sum(p1[p1 !=0]*np.log2(p1[p1 !=0]))
        H2 = - np.sum(p2[p2 !=0]*np.log2(p2[p2 !=0]))
        H12 = - np.sum(p12[p12 !=0]*np.log2(p12[p12 != 0]))
        
        # Mutual information
        MI = H1 + H2 - H12
        
        return MI/max((H1,H2))
    
    
def norm_var_information(clusters1, clusters2):
    """ returns the normalized variation of information between two
        non-overlapping clustering.
        
        .. math::

            \hat{V}(C_1,C_2) = ({H(C1|C2)+H(C2|C1)})/{log_2 N}
    
        inputs can be node_to_cluster dictionaries, cluster lists of node sets
        or instances of Partition.
    """
    # convert to list of sets
    if isinstance(clusters1, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters1.values()) + 1)]            
        for node, clust in clusters1.items():
                cluster_list[clust].add(node)
        clusters1 = cluster_list
         
    if isinstance(clusters2, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters2.values()) + 1)]            
        for node, clust in clusters2.items():
                cluster_list[clust].add(node)
        clusters2 = cluster_list        
        
    if isinstance(clusters1, Partition):
        clusters1 = clusters1.cluster_list
    
    if isinstance(clusters2, Partition):
        clusters2 = clusters2.cluster_list
        
    # num nodes
    N = sum(len(clust) for clust in clusters1)
    n1 = len(clusters1)
    n2 = len(clusters2)
    
    if USE_CYTHON:
        return cython_nvi(clusters1, clusters2, N, n1, n2)
    else:
        # loop over pairs of clusters
        p1 = np.zeros(n1) # probs to belong to clust1
        p12 = np.zeros((n1,n2)) # probs to belong to clust1 & clust2
        for i,clust1 in enumerate(clusters1):
            p1[i] = len(clust1)/N
            for j, clust2 in enumerate(clusters2):
                p12[i,j] = len(clust1.intersection(clust2))/N
        
        p2 = np.array([len(clust2)/N for clust2 in clusters2])
        
        # Conditional Shannon entropies
        H1c2 = 0
        H2c1 = 0
        for i in range(n1):
            for j in range(n2):
                if p12[i,j] > 0:
                    H1c2 -= p12[i,j]*np.log2(p12[i,j]/p2[j])
                    H2c1 -= p12[i,j]*np.log2(p12[i,j]/p1[i])
            
        
        return (H1c2+H2c1)/np.log2(N)


def static_clustering(A, t=1, rnd_seed=None):
    """ create a static clustering class to optimize the continuous time 
        stability from the graph given by the adjacency matrix `A`.
        
        Defined for undirected and connected graphs.
    
        `t` is the time associated continuous time random walk and serves as
        the resolution paramter.
    """
    
    from scipy.linalg import expm
    
    # degrees vector
    degs = A.sum(axis=1)

    degs_m1 = degs.copy()
    degs_m1[degs_m1==0] = 1
    degs_m1 = 1/degs_m1
    # DTRW Transition matrix
    Tstat = np.diag(degs_m1) @ A

    
    # stationary solution
    if degs.sum() == 0:
        pi = degs
    else:
        pi = degs/degs.sum()
    
    # transition matrix at time t
    if t != 1:
        # Random walk laplacian
        Lrw = np.eye(degs.size) - Tstat
    
        T = expm(-t*Lrw)
    else:
        T = Tstat
        
    return Clustering(p1=pi,p2=pi,T=T, rnd_seed=rnd_seed)
    
    
    