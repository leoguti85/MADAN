# MADAN
> Leonardo Gutiérrez-Gómez, Alexandre Bovet and Jean-Charles Delvenne<br>

MADAN is the acronym of **Multi-scale Anomaly Detection on Attributed Networks**.
This is an unsupervised algorithm to compute anomalies on attributed networks, i.e., graphs with node attributes, and the context of node anomalies, i.e., the clusters.
MADAN allows us to detect anomalous nodes with respect to the node attributes and network structure at *all scales* of the network.

<p align="center">
<img src="figures/office.png">
</p>

<font size="+1">Figure 1. A toy example of work relation network. Nodes have  attributes  describing  individual  features.  Node  at-tributes define structural clusters in multiple scales. At the 1st scale outlier nodes (O1,O2,O3) lie within a local con-text, i.e, offices. In a 2nd scale, departments emerge as new contexts where O2 is not defined. Finally, at a larger scale O3 remains as a global anomaly in context of the whole company.</font>


Here you can find the code of the algorith with some examples implemented on our paper:
**_Multi-scale Anomaly Detection For Attributed Networks (MADAN algorithm), published at AAAI-20 conference.
[Preprint.](https://arxiv.org/abs/1912.04144)_**

--------------------------------------------------------------------------------------------------------------------
Tested on Jupyter notebook 5.7.0 with Python 3.7.3, networkx 2.5, sklearn 0.21.2, pygsp 0.5.1

**Note:** For efficiency reasons some functions were written in cython. We recommend you to compile them before, running the following script:
```
python setup.py build_ext --inplace 
```

## Example: Income network
Suppose you have a social network where we know the income of each person. 

<p align="center">
<img src="figures/income_net.png">
</p>

We aim to find anomalous people, i.e., persons with an unexpected income according to the network structure.

#### Initiating MADAN ####
```
import Madan as md
madan = md.Madan(net, attributes=['income'], sigma=0.08)
```
Where net is a networkx graph with node attributes. 

#### Scanning relevant scales ####
Before to look at anomalies, we should scan the relevant context (scales) where potential anomalies lie.

```
time_scales   =   np.concatenate([np.array([0]), 10**np.linspace(0,5,500)])
madan.scanning_relevant_context(time_scales, n_jobs=4)
```

<p align="center">
<img src="figures/scanning_context.png">
</p>

We can also scanning relevant contexts for different times, i.e., V(t,t'):

```
madan.scanning_relevant_context_time(time_scales)
```
<p align="center">
<img src="figures/scanning_context_time.png">
</p>

#### Uncovering anomalous nodes and context ####
We compute the concentration for all nodes at time t:
```
madan.compute_concentration(t)
madan.concentration
```
and the anomalous nodes:
```
madan.anomalous_nodes
[50, 135]
```
The context for anomalies at the given scale:
```
madan.compute_context_for_anomalies()
madan.interp_com
```

<p align="center">
<img src="figures/context.png">
</p>


## Jupyter notebooks ##

* Running MADAN algorithn on a toy example network. (Figure 2 of the paper).

```
toy_example.ipynb
```

* MADAN algorithn on a real life dataset, the Disney copurchasing network (Figure 5 of the paper)

```
Case_of_study_Disney.ipynb
```

#### Benchmarking on synthetic networks ######

* MADAN algorith on synthetic networks
We generate artificial attributed networks with ground truth anomalies and evaluate the performance of recovering anomalous nodes varing the percentage of anmalies in the network (Figure 3 of the paper).

The following script will compute the ROC/AUC and PR/AUC metrics for the synthetic networks:

```
python LFR_MADAN.py
```

* Then plotting the scores as the Figure 3 of the paper:

```
cd plot_LFR/
python plot_scores.py
```

#### Real life examples ######

* Running anomaly detection on the Disney copurchasing network data and computes ROC-AUC score, (Table 2 of the paper).
```
python run_real_data.py
```
    
* Normalized variaiton of information  
This script computes the variation of information V(t,t') (background of Figure 2 and 4), between optimal partitions at times t and t'.
It allows to uncover the intrinsic scales having into account the graph structure and the node attributes.

```
variation_of_information/compare_partitions_over_time.m
```

#### Others ######

This folder contains some *Matlab* figures of the scanning of relevant partitions in the toy example and the Disney network.
```
figures/
```


### Citing
If you find *MADAN* useful for your research, please consider citing the following paper:

<!--
	@inproceedings{madan-aaai20,
	author = {Gutiérrez-Gómez Leonardo, Bovet Alexandre and Delvenne Jean-Charles},
	 title = {Multi-scale Anomaly Detection on Attributed Networks},
	 booktitle = {Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI-20)},
	 year = {2020}
	}
-->

### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <leonardo.gutierrez@list.lu>.
