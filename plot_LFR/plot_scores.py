import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.interactive(True)
plt.close('all')


# Change metric by: avg_roc_auc or  avg_auc
# Change std by:    std_roc_auc or  std_auc

metric = 'avg_roc_auc'
std = 'std_roc_auc'

df_gsp   = pd.read_csv('MADAN_LFR_pert_20.csv', index_col=0)
df_lof   = pd.read_csv('LOF_LFR_pert_20.csv', index_col=0)
df_alad  = pd.read_csv('ALAD_LFR_pert_20.csv', index_col=0)
df_amen  = pd.read_csv('amen_LFR_pert_20.csv', index_col=0)
df_radar = pd.read_csv('RADAR_LFR_pert_20.csv', index_col=0)
df_rnd   = pd.read_csv('RND_LFR_pert_20.csv', index_col=0)



colors = plt.cm.viridis(np.linspace(0,1,6))
lw = 2.5
x_score = [0,1,2,3,4,5,6]

plt.errorbar(x=x_score, y=df_gsp[metric].values,   yerr =df_gsp[std].values, marker='.', markersize='12', linewidth=lw, label='MADAN', color=colors[0])
plt.errorbar(x=x_score, y=df_lof[metric].values,   yerr =df_lof[std].values,  marker='o',markersize='11', linewidth=lw, label='LOF', color=colors[1])
plt.errorbar(x=x_score, y=df_amen[metric].values,  yerr =df_amen[std].values, marker='2', markersize='12',linewidth=lw, label='AMEN', color=colors[2])
plt.errorbar(x=x_score, y=df_alad[metric].values,  yerr =df_alad[std].values, marker='*', markersize='12',linewidth=lw, label='ALAD', color=colors[3])
plt.errorbar(x=x_score, y=df_rnd[metric].values,   yerr =df_rnd[std].values, marker='1',  markersize='12', linewidth=lw, label='Random', color=colors[4])
plt.errorbar(x=x_score, y=df_radar[metric].values, yerr =df_radar[std].values, marker='^',markersize='12', linewidth=lw, label='RADAR', color=colors[5])
plt.ylabel(metric)
plt.ylim([0,1.05])
plt.yticks(fontsize='25')
plt.xlabel('percentage of anomalous nodes', fontsize='14')
plt.xlim([0,6.01])
plt.xticks(x_score,[1,5,10,15,20,25,30], fontsize='25')
plt.legend(prop={'size': 20})
plt.set_cmap('Set1')
plt.show()
