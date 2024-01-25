#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 08:10:03 2020

@author: christopherhunter
"""

import AdvancedDerivatives1 as ad
import AdvancedDerivativesUtil as adu
import numpy as np
import pandas as pd

import timeit

#####################
# Runner
#S = 100
#K = 100
#today = 0
#Expiry = 1
#r = 0.02
#q = 0.01
#sigma = 0.20

S = 100
K = 100
today = 0
Expiry = 1/12
r = 0.0
q = 0.0
sigma = 0.2



model = ad.BlackScholesModel(S=S, r=r, q=q, sigma=sigma, t=today)
#deriv = ad.Vanilla( T=Expiry, K=K, Type='Call')
deriv = ad.Digital( T=Expiry, K=K, Type='Call')
p = ad.BlackScholesPricer(model,deriv)


#####################
# Function to calculate BS prices by MC and check the errors
#sens_key = [ ['Rho',0.01,False] ]
#sens_key = [ ['Delta',1.0001,True], ['Vega', 1.0001, True], ['Theta', 0.01, False], ['Rho', 0.0001, False] ]
sens_key = []
num_samples = 1
num_paths = 100000
#adu.black_scholes_error(p, sens_key, num_paths, save_plot=False, filename='P'+str(num_paths),antithetic=False)
#adu.sample_std(p, sens_key, num_paths, num_samples, save_plot=True, filename='S'+str(num_samples)+'P'+str(num_paths))
#adu.importance_sampling(p, sens_key, num_paths, num_samples)
#adu.black_scholes_control_variate(p, sens_key, 1, num_paths)

#####################
# Function to check formulas by numerically calculating the derivatives
#adu.discrete_greeks()     

#####################
# Function to simulate impact of hedging on derivatives pricing
real_world_drift = 0.05
real_world_vol = 0.20
real_world_model = ad.BlackScholesModel(S=S, r=real_world_drift, q=0, sigma=real_world_vol, t=today)
cost = 0.00
num_steps = 21
filename='Test'
num_paths = 10000
num_samples = 1
#ad.delta_hedging( pricer=p, real_world_model=real_world_model, start_time=today, end_time=Expiry, cost=cost, num_steps=num_steps, num_paths=num_paths, num_samples=num_samples, write_file=False, filename=filename )

#####################
# Function to simulate impact of hedging on derivatives pricing
#delta_hedging_sigma_dependency( pricer=p, real_world_model=real_world_model, start_time=today, end_time=Expiry, cost=cost, num_steps=num_steps, num_paths=num_paths, num_samples=num_samples, write_file=True, filename=filename )

#####################
# Function to simulate the impact of transactions cost on hedging 
real_world_drift = 0.05
real_world_vol = 0.20
real_world_model = ad.BlackScholesModel(S=S, r=real_world_drift, q=0, sigma=real_world_vol, t=today)
cost = 0.005
num_steps = 84
filename='Digital84'
num_paths = 10000
 
num_samples = 1
ad.delta_hedging( pricer=p, real_world_model=real_world_model, start_time=today, end_time=Expiry, cost=cost, num_steps=num_steps, num_paths=num_paths, num_samples=num_samples, write_file=True, filename=filename )

#####################
# Function to simulate hedging with the TC volatility (higher than market value)
plus_minus = 1 #plosition in the option:  1 or -1
hedging_vol = ad.get_transaction_cost_vol(sigma=real_world_vol, cost=cost, delta_t=(Expiry-today)/num_steps, plus_minus=plus_minus)
model = ad.BlackScholesModel(S=S, r=r, q=q, sigma=hedging_vol, t=today)
p = ad.BlackScholesPricer(model,deriv)
#ad.delta_hedging( pricer=p, real_world_model=real_world_model, start_time=today, end_time=Expiry, cost=cost, num_steps=num_steps, num_paths=num_paths, num_samples=num_samples, write_file=True, filename=filename, plus_minus=plus_minus )

    