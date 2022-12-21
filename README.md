# Estimation of denied boarding probabilities from AFC and AVL data.

This repository contains a project of estimation of denied boarding probabilities by train run along an urban transit line. A Maximum Likelihood Estimation method is performed from a stochastic user-centric model of egress time conditionnal to the access time. It relies on a set of observed trips (access and egress instant, ie AFC data) and observed runs (arrival and departure instants, ie AVL data). 
 
## Installation

- Creation of virtual environmnent to avoid library version conflicts:
```bash
python3 -m venv venv
source venv/bin/activate
```
- Dependencies installation: 
```bash
pip install -r requirements.txt
```
and to see itself as a package:
```bash
pip install -e .
```

- If `matplotlib.pyplot.plot()` fails:
```
sudo apt-get install python3-tk
```

## Preprocessing.

The estimation of delayed boarding probabilities requires known access and egress time distributions in the access and egress stations of the observed trips. Functions and classes to fit the best probability distribution on a set of observed times are available in the folder [preprocessing](f2b/preprocessing/). 

## Estimation of delayed boarding probabilities.

The optimization of the Maximum Likelihood on the set of observations may be conducted with different optimization method implemented in [maximum_likelihood_estimation](f2b/f2b_estimation/maximum_likelihood_estimation.py).


## Postprocessing.
Ex-post analysis on the estimated probabilities (differential information concerning the likelihood, asymptotic confidence interval) are available in [postprocessing](f2b/postprocessing/).