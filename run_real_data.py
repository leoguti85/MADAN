"""
@ Leonardo Gutiérrez-Gómez
leonardo.gutierrez@list.lu
"""

from sklearn.metrics import roc_auc_score
import numpy as np
from Nets import *
import pandas as pd
from exp_chebyshev import *
import Madan as md
import argparse
from params_db import params_db
from utils import tic, toc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-db_name', dest='db_name')
    return parser.parse_args()


#--------------------------------------------------------------------------
# Performing anomalu detection on real life networks (disney, enron, books)
#--------------------------------------------------------------------------
args         = parse_args()

data_params  = params_db[args.db_name]
num_net     =  data_params['num_net']
name        =  args.db_name
attributes  =  data_params['attributes']

#---------------------------------------------------------------------------

data        =   Nets(num_net,attributes[0]) 

net         =   data.net
N           =   net.order()
partition   =   nx.get_node_attributes(net,"ModularityClass")

y_true      =   [net.nodes[n]['anomaly'] for n in net.nodes()] 
true_nodes  =   [i for i in range(0,len(y_true)) if y_true[i]==1]

#---------------------------------------------------------------------------
taus        =   10**np.linspace(0,4,200)
taus        =   np.concatenate([np.array([0]),taus])
sigma_list  =  [data_params['sigma']]
#----------------------------------------------------------------------------

print(num_net)
print(name)
print(attributes)
print(sigma_list)
print('\n')

res  = dict()

tic();
count_itr = 0
for count_sig, sigma in enumerate(sigma_list):

    #-----------------------------------------------------------------------
    madan = md.Madan(net, attributes, sigma=sigma)    
    #-----------------------------------------------------------------------

    auc_values      = []
    roc_auc         = []
    avg_precision   = []
    concentration_times = np.zeros((N,len(taus)))

    print("Looking at the best scale...\n")

    for inx, t in enumerate(taus):

        #-------------------------------------------------------------------
        madan.compute_concentration(t)
        concentration_times[:,inx]  = madan.concentration

        #concentration_times[:,inx]  = compute_fast_exponential(G,t/avg_d) # Chebychev
        #concentration_times[:,inx]   = chebychev_sequential(G,t/avg_d)
        #-------------------------------------------------------------------
        y_scores  = concentration_times[:,inx]

        
        #--------- Evaluate roc_auc -----------------------------------------
        roc_auc.append(roc_auc_score(y_true, y_scores, average='macro'))    
        
        if count_itr%10 == 0:
            print('Itr: %f / %f'%(count_itr, len(sigma_list)*len(taus)))
        count_itr+=1    

    #---------------------------------------------------------------------            
    # ROC-auc
    #---------------------------------------------------------------------            
    inx = np.argmax(roc_auc)
    res[(sigma,inx)] = roc_auc[inx]
    print('\n')
    print('Best scale, time= %f' % (taus[inx]/(madan.avg_d)))
    print('ROC/AUC: %f'% (roc_auc[inx]))
    print('\n')


toc();
