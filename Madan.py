from pygsp import graphs, filters, plotting
import numpy as np
from sklearn import preprocessing
import networkx as nx
from Plotters import *
import pandas as pd
from collections import Counter
from LouvainClustering_fast import Clustering, norm_var_information
from joblib import Parallel, delayed
import pdb	

class Madan(object):


	def __init__(self, net, attributes=None, sigma=None):

		self.N          = net.order()
		self.network    = net
		self.attributes = attributes
		self.pl         = Plotters()

		f_matrix = np.zeros((self.N,len(attributes)))
		
		for i, att in enumerate(attributes):
			attribs       =  nx.get_node_attributes(net,att)
			f_matrix[:,i] =  list(attribs.values())


		f_scaled = preprocessing.MinMaxScaler().fit_transform(f_matrix)

		if sigma is None:
		   self.sigma = f_scaled.std()
		else:
		   self.sigma = sigma   

		#--------------------------------------------------------------------
		self.A = nx.adjacency_matrix(net).toarray()
		self.W = self._get_weigth_matrix_vectors(self.A,f_scaled,self.sigma)
		self.G = graphs.Graph(self.W)
		self.computing_fourier_basis()
		#------------------------------------------------------------------------
		# Random walk components
		#------------------------------------------------------------------------
		v_ones      =  np.matrix(np.ones((self.N,1)))
		degree_vect =  self.W.sum(axis=1)                        # strengths vector
		D           =  np.matrix(np.diag(degree_vect))    
		self.avg_d  =  (v_ones.T*D*v_ones)[0,0]/self.N           # average strength

		#------------------------------------------------------------------------
		# stationary distribution
		#------------------------------------------------------------------------
		self.pi     =  v_ones.A.T.reshape(self.N)/self.N
		#------------------------------------------------------------------------

	def computing_fourier_basis(self, chebychev=False):

		self.G.compute_fourier_basis() 

		
	def evaluating_heat_kernel(self, tau=0):	
		#------------------------------------------------------------------------
		# Computing Fourier basis evaluated at tau
		#------------------------------------------------------------------------
								
		kernel     =  np.exp(-tau/self.avg_d*(self.G.e/self.G.lmax)) 
		exp_sigma  =  np.diag(kernel)
		self.Ht    =  np.dot(np.dot(self.G.U,exp_sigma),self.G.U.T)
		self.tau   =  tau
		
		

	def compute_concentration(self, tau=0):		
		
		self.evaluating_heat_kernel(tau)
		self.concentration =  np.linalg.norm(self.Ht, axis=0, ord=2)

		self.concentration + 2*self.concentration.std()
	   
		thre = self.concentration.mean() + 2*self.concentration.std()
		self.anomalies_labels = (self.concentration>=thre)*1
		self.anomalous_nodes  = [i for i in range(0,len(self.anomalies_labels)) if self.anomalies_labels[i]==1]

	
	def compute_context_for_anomalies(self, random_seed=2):

		#------------------------------------------------------------------------
		#  Find clusters with Louvain algorithm 
		#------------------------------------------------------------------------
		clustering   =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)    
		
		clustering.find_louvain_clustering(rnd_seed=random_seed)
		self.partition = clustering.partition.node_to_cluster_dict

		clust_labels      = np.array([self.partition[i] for i in range(len(self.partition))]) # Re-order

		self.interp_com   = self._interpolate_comm(self.network, clust_labels, self.anomalous_nodes)
		self.num_clusters = len(set(self.interp_com))
	

		
	def plot_graph_concentration(self, coord=None):

		if coord is None:
			self.G.set_coordinates(iterations=1000, seed=100)
		else:	
			self.G.set_coordinates(coord)

		self.pl.visualize_graph_signal(self.G,self.concentration,title='t='+str(self.tau))
		#self.pl.plot_matrix(W, inter=0, title="W")
		
	def plot_graph_context(self, coord=None):
		
		if coord is None:
			self.G.set_coordinates(iterations=1000, seed=100)
		else:	
			self.G.set_coordinates(coord)

		#node_labels = dict(zip(self.network.nodes(), np.array(self.interp_com, dtype='int')))
		#self.pl.visualize_graph_signal(self.G, self.interp_com, node_labels=node_labels)    
		self.pl.visualize_graph_signal(self.G, self.interp_com)    


	def plot_concentration(self):
		#------------------------------------------------------------------------
		# Plotting concentration
		#------------------------------------------------------------------------
		comm_concent = np.zeros((self.N,2))
		comm_concent[:,0] = self.interp_com
		comm_concent[:,1] = self.concentration
		
		df_comm_concent   = pd.DataFrame(comm_concent, columns=['groups','concentration'])
		
		ax = plt.subplot(1, 1, 1)
		std_val = df_comm_concent['concentration'].std()

		df_comm_concent.plot(kind='bar', title='Node concentration', grid=False, y='concentration',rot=0, ax=ax, cmap='viridis', fontsize=8, legend=False)
		plt.hlines(df_comm_concent['concentration'].mean() + 2.0*std_val, xmin=-1, xmax=170, linestyles='dashed', alpha=1.0, color='blue')
		ax.set_facecolor((1.0, 1.0, 1.0))
		plt.show()
	

	# Only interpolate anomalies
	def _interpolate_comm(self, G,part, true_nodes):

		new_part = part
		for node in true_nodes:
			neigh_n =  list(G.neighbors(node))
			
			flag = len(set(part[neigh_n])) == len(part[neigh_n])
			if not(flag): 
				res            =  dict(Counter(part[neigh_n]))
				max_key        =  max(res, key=lambda k: res[k])
				new_part[node] =  max_key

		keys_comm = list(set(new_part))
		vals_comm = range(0,len(keys_comm))
		mapping = dict(zip(keys_comm, vals_comm))
		interp_comm_ok = np.array([mapping[val] for val in new_part])
		
		return interp_comm_ok	

	def _get_weigth_matrix_vectors(self, A,attrib_mat,sigma=1):
	
		N = A.shape[0]
		W = np.zeros((N,N))

		for i in range(0,N):
			for j in range(i,N):
				if A[i,j] == 1:	
								
					W[i,j] = np.linalg.norm(attrib_mat[i] - attrib_mat[j])
					W[j,i] = W[i,j]
			
		W_res  = np.exp(-0.5*np.square(W)/(sigma**2))
		
		return np.multiply(W_res,A)		

	def compute_voi(self, list_partitions): #M x N
	
		res = []
		for i in range(len(list_partitions)):
			for j in range(i, len(list_partitions)):
				if i!=j:
					voi = norm_var_information(list_partitions[i],list_partitions[j])
					res.append(voi)
				
		return np.mean(res)	


	def __processInput(self, i):
		clustering =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)      
		clustering.find_louvain_clustering(rnd_seed=i)
		return clustering.partition.node_to_cluster_dict

	def scanning_relevant_context(self, time, n_jobs=1):
	
		self.num_com   = []
		self.voi_list  = []
		self.time      = time

		for inx, t in enumerate(time):
			
			if inx%50==0:
				print("Processed %d/%d"%(inx,len(time)))
			
			#------------------------------------------------------------------         
			self.evaluating_heat_kernel(t)
			#------------------------------------------------------------------         
			list_partitoins = Parallel(n_jobs=4)(delayed(self.__processInput)(i) for i in range(20))      
			self.voi_list.append(self.compute_voi(list_partitoins))

			self.compute_context_for_anomalies()
			self.num_com.append(self.num_clusters)

		#-----------------------------------------------------------------------                
		self.plot_relevant_context()
		#-----------------------------------------------------------------------            
		
		
	def plot_relevant_context(self):

		fig, ax1 = plt.subplots(figsize=(9,5))

		color = 'tab:blue'
		ax1.set_xlabel('time (s)', size=16)
		ax1.set_ylabel('Num communities', color=color, size=16)
		ax1.set_yscale('log')
		ax1.set_ylim([1,np.max(self.num_com)+10])
		ax1.set_xscale('log')
		ax1.plot(self.time, self.num_com, color=color, linewidth=2)
		ax1.tick_params(axis='y', labelcolor=color)
		ax1.grid(False)
		#ax1.set_facecolor((1.0, 1.0, 1.0))

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

		color = 'tab:red'
		ax2.set_ylabel('Variaiton of Information', color=color, size=16)  # we already handled the x-label with ax1
		ax2.plot(self.time, self.voi_list, color=color, linewidth=2, alpha=0.5)
		ax2.tick_params(axis='y', labelcolor=color)
		ax2.grid(False)

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.show()


	def scanning_relevant_context_time(self,time):
		#--------------------------------------------
		# Compute temporal partitions
		#--------------------------------------------
		mat_parts  = []    
		for inx, t in enumerate(range(len(time))):
			
			if inx%50==0:
				print("Processed %d/%d"%(inx,len(time)))
					
			#------------------------------------------------------------------         
			self.evaluating_heat_kernel(t)
			#------------------------------------------------------------------         
			clustering =  Clustering(p1=self.pi, p2=self.pi, T=self.Ht)      
			
			clustering.find_louvain_clustering(rnd_seed=42)
			partition  = clustering.partition.node_to_cluster_dict 
			mat_parts.append(partition)

		#-----------------------------------------------------------------------
		self.voi_mat = np.zeros((len(time),len(time)))
		
		print("Computing variation of information between V(t,t')...")           
		for i in range(len(mat_parts)):                       
			
			for j in range(i, len(mat_parts)):
				if i!=j:
					voi = norm_var_information(mat_parts[i],mat_parts[j])
					self.voi_mat[i,j] = voi
					self.voi_mat[j,i] = voi       

		#-----------------------------------------------------------------------            
		self.plot_voi_matrix()            
		#-----------------------------------------------------------------------            


	def plot_voi_matrix(self):
		fig = plt.figure(figsize=(8,8))
		plt.imshow(self.voi_mat, origin='low')
		plt.grid(False)
		plt.colorbar()	
		plt.show()