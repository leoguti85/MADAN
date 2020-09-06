import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import average_precision_score, roc_auc_score
from LFR_nets import *
import Madan as md

"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""

#------------------------------------------------------------------------------------------------
# Parameters of synthetic networks
#------------------------------------------------------------------------------------------------
list_percentage = [1,5,10,15,20,25,30]
num_attribs              =  20  
num_anomalous_attribs    =  6

#-------------------------------------------------------------------------------------------------
LFR         =   LFR_nets()
time        =   10**np.linspace(0,5,200)
time        =   np.concatenate([np.array([0]),time])
t           =   time[90]
sigma       =   0.1
#-------------------------------------------------------------------------------------------------

df_results = pd.DataFrame(columns=['avg_auc','avg_avg_prec','avg_roc_auc','mu','net_id','perturbation','std_auc','std_avg_prec','std_roc_auc'])

print("Benchmarking on synthetic networks\n")

for perc_anomalous_nodes in list_percentage:

	print("Attributes dimension: %d"%num_attribs); 
	print("Anomalous attributes: %d"%num_anomalous_attribs);
	print("Percentage of anomalous nodes: %d"%perc_anomalous_nodes);
	print("Generating artificial networks with node attributes and anomalies...")
	print("\n")


	# Results for each percentage of anomalous nodes
	pr_auc_values      = []
	roc_auc_values     = []
	avg_precision_values = []

	
	for net_id in range(0,50): #LFR.num_nets
	
		mu = '0.1'		
		LFR.select_net(net_id)
		attrib_partition = LFR.get_attrib_clusters()
		groups_partition = LFR.get_true_communities()

		LFR.creating_node_attributes(attrib_partition,num_attribs) # dimension of anomalies
		LFR.injecting_anomalies(perc_anomalous_nodes, num_anomalous_attribs)

		#-------------------------------------------------------------------------------------------
		# Staring MADAN Algorithm
		#-------------------------------------------------------------------------------------------

		net = LFR.net
		N   = net.order()

		madan = md.Madan(net, LFR.attributes_names, sigma=sigma)	

		#-------------------------------------------------------------------------------------------------
		# Filtering signal with heat kernel 
		#-------------------------------------------------------------------------------------------------
		madan.compute_concentration(t)

		y_scores = madan.concentration

		#-------------------------------------------------------------------------------------------------
		#  Evaluating Precision-Recall AUC 
		#-------------------------------------------------------------------------------------------------
		precision, recall, thresholds = precision_recall_curve(LFR.y_true, y_scores, pos_label=1)
		
		pr_auc = auc(recall,precision)
		pr_auc_values.append(pr_auc)

		#-------------------------------------------------------------------------------------------------
		# Evaluating ROC - AUC
		#--------------------------------------------------------------------------------------------------	
		roc_auc_values.append(roc_auc_score(LFR.y_true, y_scores, average='macro'))	
			
		#-------------------------------------------------------------------------------------------------
		# Evaluating average precision (AP)
		#-------------------------------------------------------------------------------------------------
		avg_precision_values.append(average_precision_score(LFR.y_true, y_scores, average='macro'))	


				
	#---------------------------------------------------------------------------------------

	avg_pr_auc    =  round(np.mean(pr_auc_values),3)
	std_pr_auc    =  round(np.std(pr_auc_values),3)

	avg_roc_auc   =  round(np.mean(roc_auc_values),3)
	std_roc_auc   =  round(np.std(roc_auc_values),3)

	avg_avg_prec   = round(np.mean(avg_precision_values),3)
	std_avg_prec   = round(np.std(avg_precision_values),3)

	df_results = df_results.append({'perturbation': perc_anomalous_nodes, 
									'avg_auc': avg_pr_auc,   'std_auc': std_pr_auc, 
									'avg_roc_auc': avg_roc_auc, 'std_roc_auc': std_roc_auc,
									'avg_avg_prec': avg_avg_prec, 'std_avg_prec': std_avg_prec,
									'mu': mu, 'net_id': 0}, ignore_index=True)


print(df_results)

df_results.to_csv('plot_LFR/MADAN_LFR_pert_20.csv')

print("Now, plot the results: cd plot_LFR/")
print("python plot_scores.py")

