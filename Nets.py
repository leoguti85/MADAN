# -*- coding: utf-8 -*-
"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""


import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite
from sklearn import preprocessing
import numpy as np
import scipy.io
import json
from sklearn import metrics
from networkx.readwrite import json_graph
import os
import glob
import community
import ast


seed = 100			
np.random.seed(seed)


class Nets(object):


	def __init__(self, num_net, attrib_name):
		self.net, self.name_net = self.choose_net(num_net)
		self.attribs_vals = list(nx.get_node_attributes(self.net,attrib_name).values())
		self.attribs_encoded = self.encode_labels()
		self.adding_weights()


	def choose_net(self, num_net):
		
		if num_net == 1:
			network = self.load_disney()
			name = "disney_"	
  			
		elif num_net == 2:	
			network = self.load_toy_example()
			name = "toy_example"	

		elif num_net == 3:	
			network = self.load_books()
			name = "books"		

		elif num_net == 4:	
			network = self.load_enron()
			name = "enron"			

	
		return (network,name)		
	
	
	def load_enron(self):

		net = nx.read_graphml('data/Enron/Enron.graphml')

		anomalies = pd.read_csv('data/Enron/Enron.true', index_col=0, sep=';', header=None)

		net = net.to_undirected()

		for n in net.nodes():
			net.nodes[n]['anomaly'] = anomalies.iloc[int(n)][1]

		net = nx.convert_node_labels_to_integers(net, first_label=0, ordering='sorted')
					
		return net	
	
			
	def load_disney(self):
		
		
		net = nx.read_graphml('data/Disney/DisneyWithClusters.graphml')
		net = nx.convert_node_labels_to_integers(net, ordering='sorted', first_label=0)
		anomalies = pd.read_csv('data/Disney/OutliersLabelsDisney_6.csv', index_col=0, sep=';', header=None)

		code_titles = pd.read_csv('data/Disney/codes_titles.csv', index_col=0, sep=',')

		net = net.to_undirected()
		
		for n in net.nodes():
			cod = net.nodes[n]['label'] # take the label from node
			net.nodes[n]['anomaly'] = anomalies.loc[cod][1]
			net.nodes[n]['title'] = code_titles[code_titles['code']==cod]['title'].values[0]
		
		
		return net


	def load_toy_example(self):
	
	            #rich, medium, poor, very-poor
		sizes = [30, 70, 35, 25] 
		
		probs = [[0.4,   0.04,  0.01, 0.009],
				[0.04,    0.3,  0.009,  0.009],
				[0.01,    0.009, 0.3,  0.02],
				[0.009,   0.009,  0.02,  0.4]]
		
		probs   = np.array(probs)	 
		G       = nx.stochastic_block_model(sizes, probs, seed=1)
		
		partition = nx.get_node_attributes(G,"block")
		
		df        = pd.DataFrame(data=list(partition.values()), index=list(partition.keys()))


		# very-rich, medium, poor, very-poor
		#params = [(10000,1000),(2500,250),(1200,120),(600,60)]
		params = [(10000,10),(2500,10),(1200,10),(600,10)]
		attribs = []
		for comm, num_nodes in enumerate(sizes):
			nodes     =  df[df[0]==comm].index.values
			mu, sigma =  params[comm]
			attribs   =  attribs + list(zip(nodes, np.random.normal(loc=mu,scale=sigma, size=len(nodes))))
				

		attribs2 = zip(G.nodes(), np.zeros(G.order()))
		nx.set_node_attributes(G,dict(attribs),'income')
		nx.set_node_attributes(G,dict(attribs2),'anomaly')

		# Anomalies
		G.nodes[8]['income']  = 2600   # Medium in rich
		G.nodes[135]['income'] = 9650  # Rich in very-poor
		G.nodes[50]['income'] =  600   # very-poor in medium

		G.nodes[8]['anomaly']  = 1  
		G.nodes[135]['anomaly'] = 1
		G.nodes[50]['anomaly']  = 1  

		return G


	def load_books(self):
		net = nx.read_graphml('data/AmazonFail/AmazonFailNumeric.graphml')
		anomalies = pd.read_csv('data/AmazonFail/AmazonFailNumeric.true', index_col=0, sep=';', header=None)

		net = net.to_undirected()

		for n in net.nodes():
			net.nodes[n]['anomaly'] = anomalies.loc[int(n)][1]
		
		net = nx.convert_node_labels_to_integers(net, ordering='sorted', first_label=0)

		
		return net

	def encode_labels(self):

		if isinstance(self.attribs_vals[0], np.ndarray):
			return np.array(self.attribs_vals)
		else:	
			le = preprocessing.LabelEncoder()
			return le.fit_transform(self.attribs_vals)
		

	def adding_weights(self):
		for edge in self.net.edges():
			self.net[edge[0]][edge[1]]['weight'] = 1
