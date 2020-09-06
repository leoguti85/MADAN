import pandas as pd
import numpy as np
import networkx as nx

"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""

# This class load and create node attributes plus anomalies on synthetic LFR networks
class LFR_nets():



	def __init__(self):

		self.global_std  =   0.1
		self.anomal_std  =   1.0
		self.global_seed =   42
		self.data        =   self.load_LFR_graphs()
		self.attributes_names = []
	
	def load_LFR_graphs(self):


		self.num_nets = 50 

		nets = dict()
		for index, i in enumerate(range(0,self.num_nets)): 

			edges  = pd.read_csv('data/LFR/nets/network'+str(i)+'.dat', sep='\t', header=None)
			groups = pd.read_csv('data/LFR/nets/community'+str(i)+'.dat', sep='\t', header=None)
			
			G = nx.from_edgelist(edges.values)
			mapping = dict(zip(groups[0].values, groups[1].values))
			
			nx.set_node_attributes(G, mapping, 'group')
			G = nx.convert_node_labels_to_integers(G, first_label=0)
			
		
			nets[index] = {'network': G, 'num_com': len(groups[1].unique()), 'mu': '0.1'}

		return nets	


	def select_net(self, net_id=0):
		self.net_id  =  net_id
		self.network =  self.data[net_id]
		self.net     = self.network['network']
		self.true_communities     = nx.get_node_attributes(self.net,"group")
		self.true_attrib_clusters = self.creating_attrib_clusters()


	def creating_attrib_clusters(self):
			
		groups    = set(list(self.true_communities.values()))
		num_com   = len(groups)

		attrib_group = []
		for c in groups:
			
			for count in range(0,c):
				if len(attrib_group)<=num_com:
					attrib_group.append(c)
				else:	
					break;	

		mapping = dict(zip(groups,attrib_group))	

		values  = [(n,mapping[self.net.nodes[n]['group']]) for n in self.net.nodes()] # (node, attribs_groups)
		nx.set_node_attributes(self.net, dict(values), 'attrib_clusters')
		return dict(values)		



	def creating_node_attributes(self, partition, attrib_dim=1):
		'''
		input: attrib_dim, Number of attributes per node
		Generate attributes for each ground truth communities

		'''
		N                =  self.net.order()
		partition_labels =  np.array(list(partition.values())).reshape(N,1)

		node_attribs     = np.zeros((N,attrib_dim))
		comm_labels      = list(set(partition_labels.ravel()))

		data = np.concatenate([partition_labels,node_attribs], axis=1)
		df_partition = pd.DataFrame(data, index = list(partition.keys())) # index: nodes, node group, then zeros Nxd

		for at in range(1,attrib_dim + 1):

			# Iterate over groups assiging attributes
			num_nodes_comms = []
			for comm in comm_labels: # comm_labels
				comm_i    = df_partition[df_partition[0]==comm]
				num_nodes = comm_i.shape[0] # num nodes in comm i

				kkey = str(at)+str(comm)
				np.random.seed(int(kkey))

				if at%3 == 0:  #a3
					node_vals = np.random.normal(loc=comm, scale=self.global_std, size= num_nodes) # mu=comm_index, std=global
				elif at%3 == 1:	#a1
					node_vals = np.random.uniform(low=(comm-self.global_std), high=(comm+self.global_std), size=num_nodes)	
				elif at%3 == 2: #a2
					node_vals =	np.random.logistic(loc=comm, scale=self.global_std, size=num_nodes)
				
				
				df_partition.loc[comm_i.index,at] = node_vals
				num_nodes_comms.append(num_nodes)
				
			mapping = dict(zip(df_partition.index.values, df_partition[at].values))    
			at_name = 'a'+str(at)
			nx.set_node_attributes(self.net, mapping, at_name)  

		self.attributes_names = ['a'+str(i) for i in range(1,attrib_dim + 1)]



	# Creating node attributes iterating over communities	
	def creating_node_attributes2(self, attrib_dim=1):
		'''
		input: attrib_dim, Number of attributes per node
		Generate attributes for each ground truth communities

		'''
		N                =  self.net.order()
		partition        =  self.true_communities
		partition_labels =  np.array(list(partition.values())).reshape(N,1)

		node_attribs     = np.zeros((N,attrib_dim))
		comm_labels      = list(set(partition_labels.ravel()))

		data = np.concatenate([partition_labels,node_attribs], axis=1)
		df_partition = pd.DataFrame(data, index = list(partition.keys()))

		for at in range(1,attrib_dim + 1):

			np.random.seed(at)
			# Iterate over communities assiging attributes
			num_nodes_comms = []
			for comm in comm_labels: 								# ground truth comm labels

				comm_i    = df_partition[df_partition[0]==comm]
				num_nodes = comm_i.shape[0] 						# num nodes within comm i

				node_vals = np.random.normal(loc=comm, scale=self.global_std, size= num_nodes)  # generating nattributes for at_i within current comm
				
				df_partition.loc[comm_i.index,at] = node_vals
				num_nodes_comms.append(num_nodes)
				
			mapping = dict(zip(df_partition.index.values, df_partition[at].values))    
			at_name = 'a'+str(at)
			nx.set_node_attributes(self.net, mapping, at_name)  

		self.attributes_names = ['a'+str(i) for i in range(1,attrib_dim + 1)]
	

	# Must be used after injecting anomalies
	# attr_labels must be a list	
	def set_new_attributes(self, net, attr_labels):
		# Aditional node attribus
		N = net.order()
		for at in attr_labels:

			np.random.seed(self.global_seed)
			attribs     =   dict(zip(net.nodes(),np.random.uniform(low=0.0, high=1.0, size=N)))
			nx.set_node_attributes(net, attribs, at)

		self.attributes_names = self.attributes_names + attr_labels	
		self.net = net

	def get_true_communities(self):
		return self.true_communities 

	def get_attrib_clusters(self):
		return self.true_attrib_clusters

	def injecting_anomalies(self,percet_anomalous_nodes = 10, number_anomalous_attribs = 1):
		'''
		Inject certain percentaje of anomalies
		Inject an anomaly by modifying some percentage of attributes values
		'''
		
		N = self.net.order()	
		partition = self.get_attrib_clusters()
		partition_labels =  np.array(list(partition.values())).reshape(N,1)
		comm_labels = list(set(partition_labels.ravel()))

		#defining anomalous nodes
		number_anomalous_nodes = int(1.0*percet_anomalous_nodes/100*N)

		np.random.seed(self.net_id)
		anomalous_nodes = np.random.choice(self.net.nodes(), number_anomalous_nodes, replace=False)

		#-- defining anomalous attributes ----
		#-- Same anomalous attributes are used in the whole network
		#num_attribs = len(self.attributes_names)
		#number_anomalous_attribs = int(1.0*percet_anomalous_attribs/100*num_attribs)
		anomalous_attribs = np.random.choice(self.attributes_names, number_anomalous_attribs, replace=False)


		for node_i in anomalous_nodes:
			
			comm_i = partition[node_i]
			comm_lables_without_i = [c for c in comm_labels if c!=comm_i]
			mu_anomalous = np.random.choice(comm_lables_without_i)

			# anomalous attributes are drawn from a normal with the same mean and std
			for inx, at in enumerate(anomalous_attribs):

				if (inx+1)%3 == 0:   # a3
					node_vals = np.random.normal(loc=mu_anomalous, scale=self.global_std) # mu=comm_index, std=global
				elif (inx+1)%3 == 1: # a1	
					node_vals = np.random.uniform(low=(mu_anomalous-self.global_std), high=(mu_anomalous+self.global_std))	
				elif (inx+1)%3 == 2: # a2
					node_vals =	np.random.logistic(loc=mu_anomalous, scale=self.global_std)
				
				self.net.nodes[node_i][at] = node_vals


		# generating true binary vector of anomalies    
		anomalies = np.zeros(N, dtype=int)
		anomalies[anomalous_nodes] = 1
		mapping = dict(zip(self.net.nodes(), anomalies))
		nx.set_node_attributes(self.net, mapping, 'anomaly')
		self.y_true            =  anomalies
		self.anomalous_nodes   =  anomalous_nodes
		self.anomalous_attribs =  anomalous_attribs


	def get_node_attributes(self):	

		num_attribs = len(self.attributes_names)
		N = self.net.order()

		f_matrix = np.zeros((N,num_attribs))
		for i, att in enumerate(self.attributes_names):
			attribs  =  nx.get_node_attributes(self.net,att)
			f_matrix[:,i] =  list(attribs.values())

		return f_matrix	

	def set_global_seed(self, new_seed):
		self.global_seed = new_seed
		np.random.seed(self.global_seed)
