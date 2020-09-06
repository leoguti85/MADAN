	# -*- coding: utf-8 -*-

"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import scipy.io as sio
from scipy.linalg import eig
from scipy.linalg import norm
from sklearn import preprocessing
from matplotlib.colors import Normalize
from scipy.spatial.distance import squareform,pdist  
import pandas as pd
from sklearn.metrics import roc_curve


plt.style.use('ggplot')

class Plotters(object):

	
	def plot_matrix(self, M, inter=1, title=None):
		fig, ax = plt.subplots()

		if inter:
			cax = ax.imshow(M,interpolation='spline16')
		else:
			cax = ax.imshow(M)	
					
		plt.grid(False)
		if not(title is None):
			plt.title(title)					
		cbar = fig.colorbar(cax)
		plt.show() 
	

	def plot_matrix_enumeroted(self, M, numeroted,labs, inter=1):
		fig, ax = plt.subplots()

		if inter:
			cax = ax.imshow(M,interpolation='nearest')
		else:
			cax = ax.imshow(M)	
		if numeroted==1:	
			ax.set_xticklabels(['', labs])
							
		cbar = fig.colorbar(cax)
		plt.show() 

	def visualize_graph_signal(self,G, graph_signal,title='Graph signal', node_labels=None):
		'''
		G: PyGSP graph
		W: graph_signal to plot on the graph nodes
		'''

		#fig 		 =  plt.figure()
		fig, ax      =  plt.subplots() 
		W            =  G.W.toarray()
		net          =  nx.Graph(W)
		pos 		 =  G.coords
		degree		 =  dict(nx.degree(net))
		edges_weight =  nx.get_edge_attributes(net,'weight')
		node_size	 =  [v*100 for v in degree.values()]

		if node_labels is None:
			node_labels = {}
			for nod in net.nodes():			
				node_labels.setdefault(nod, nod)		

		if graph_signal.max() > 1:
			valor_max = graph_signal.max()
		else:
			valor_max = 1

		if graph_signal.min() < 0:
			valor_min = graph_signal.min()
		else:
			valor_min = 0		

		edges_vals     = np.array(list(edges_weight.values()))
		scaled_weights = (edges_vals - W.min())/(W.max() - W.min())*(5.0 - 0.3) + 0.3

		ax.grid(False)
		ax.set_facecolor((1.0, 1.0, 1.0))
		nx.draw_networkx_labels(net, pos, labels = node_labels,font_size=10, font_color = 'white', ax=ax)
		nx.draw_networkx_edges(net, pos, edgelist=edges_weight.keys(), width=scaled_weights, ax=ax, alpha=0.6)
		
		ax1 = nx.draw_networkx_nodes(net, pos, node_size=190, node_color=graph_signal, ax=ax, cmap=plt.cm.jet, vmin=valor_min,vmax=valor_max)
		plt.colorbar(ax1)
		plt.title(title, fontsize=16)

		plt.show()


	def visualize_net_partition(self,G, partition,it=50):

		mplt.rcParams['figure.figsize'] = 12, 8

		pos = nx.spring_layout(G, iterations=it)
		colors = 'bgrcmykw'
		for i, com in enumerate(set(partition)):
			list_nodes = [nodes for nodes in G.nodes()
										if partition[nodes] == com]
			nx.draw_networkx_nodes(G, pos, list_nodes,
									node_size=100, node_color=colors[i])

		nx.draw_networkx_edges(G, pos, alpha=0.5);
		plt.show()


	def plot_mean_std(self,means, stds,tit):
		mins = min(means)
		maxs = max(means)
		plt.title(tit)
		plt.ylim([mins-0.3,maxs+0.3])
		plt.ylabel('cv acc')
		plt.xticks(arange(0,len(means)+1))
		plt.xlabel('Number of experiments')
		plt.errorbar(range(1,len(means)+1), means, stds, linestyle='None', marker='^')

		plt.show()	



	def plot_auc_pr_rc(self, recall, precision, title='PR-RC curve'):
		
		fig = plt.figure()
		#step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
		plt.step(recall, precision, color='b', alpha=0.2,  where='post')
		plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
		plt.title(title)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])	
		plt.show()


	def plot_roc_auc(self,y_true,y_scores, title='ROC-AUC curve'):
		
		fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

		plt.plot([0, 1], [0, 1], linestyle='--')
		plt.plot(fpr, tpr, marker='o', linewidth=2.0)
		plt.xlabel("False positive rate")
		plt.ylabel("True positive rate")
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])	
		plt.show()	
