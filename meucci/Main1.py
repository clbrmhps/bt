# -*- coding: utf-8 -*-
"""
Created on Sat May 13 22:26:50 2017

@author: Hao Guo
"""
'''
risk budgeting for equal weights portfolio
'''

import numpy as np 
import pandas as pd
import plotly
from plotly.graph_objs import Surface

obs_window = 900
roll_window = 30
T = int((len(data) - obs_window)/30) # number of windows
N = int(ret.shape[1]) # number of stocks

p_pc = np.asmatrix(np.zeros((N,T)))
p_mt = np.asmatrix(np.zeros((N,T)))
enb_pc = np.zeros(T)
enb_mt = np.zeros(T)
margin_risk = np.asmatrix(np.zeros((N,T)))

for t in range(T):
    Sigma = np.asmatrix(np.cov(ret.iloc[t*roll_window:t*roll_window+obs_window].T))
    w = weights_equal(Sigma)
    t_pc = torsion(Sigma, 'pca')
    t_mt = torsion(Sigma, 'minimum-torsion',  method='exact')
    p_pc[:,t], enb_pc[t] = EffectiveBets(w, Sigma, t_pc)
    p_mt[:,t], enb_mt[t] = EffectiveBets(w, Sigma, t_mt)
    margin_risk[:,t] = w.reshape((N,1)) * np.asarray(np.asmatrix(Sigma)*np.asmatrix(w).T) \
                       / (np.asmatrix(w) * np.asmatrix(Sigma) * np.asmatrix(w).T)
    

import plotly
from plotly.graph_objs import Surface

plotly.offline.plot([
    dict(z=np.sort(p_mt,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])
'''

plotly.offline.plot([
    dict(z=np.sort(p_pc,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])

plotly.offline.plot([
    dict(z=np.sort(margin_risk,axis=0)[::-1], type='surface'),
    dict(z=np.tile(np.asmatrix(w),(T,1)).T, showscale=False, opacity=0.9, type='surface')])
'''
