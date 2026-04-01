# MCMC_chain_code

This repo is where I keep codes for both running MCMC analyses (specifically for configuration space PNG analyses), as well as code to visualize the results with corner plots. Much of this code was initially written by either Zack Brown or Ben Levi, and my contributions have been with the goal of increasing efficiency or ease of use. 

All major functions have their own docstrings describing them, and an example of current functionality is available in example.ipynb.

Everything in codes/PNGModel.py (the code that is used to run the MCMC) is done at the level of the observation vector. The theoretical observation vector is defined by the chosen MathModel. To adapt to a new type of observation vector, all that is needed is to define the corresponding theoretical observation vector which is described in detail in codes/MathModels.py. 
