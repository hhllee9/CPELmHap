from typing import List
import numpy as np
from scipy.optimize import approx_fprime
from typing import List, Callable
from scipy.optimize import minimize
from Inference import *
import pandas as pd
from statsmodels.stats.multitest import multipletests
import os
from itertools import combinations
from math import comb  


def get_all_grad_logZs(n: np.ndarray, θs:np.ndarray) -> np.ndarray:
    if θs.ndim == 1:
        θs = θs.reshape(1, -1)
    return np.array([get_grad_logZ(n, θ) for θ in θs], dtype=float)


# # check
# n=np.array([5])
# θs=np.array([[1.0,2.0],[3.0,4.0],[-1.0,-2.0]])
# print(get_all_grad_logZs(n, θs))

#Compute the difference in mean test statistics between two unpaired samples
def comp_unmat_stat_mml(n: np.ndarray, grad_logZ1s: np.ndarray, grad_logZ2s:np.ndarray) ->float:
    mml1 = np.mean([comp_mml(n, g) for g in grad_logZ1s])
    mml2 = np.mean([comp_mml(n, g) for g in grad_logZ2s])
    return float(mml1 - mml2)

# # check
# n=np.array([5])
# θ1s=np.array([[1.0,1.0],[1.0,1.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s =get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s))


def comp_unmat_perm_stat_mml(n:np.ndarray, grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray, perm_ids: List[int]) -> float:
    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s),axis=0)
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis = 0)

    return comp_unmat_stat_mml(n, grad_logZ1sp, grad_logZ2sp)



# # check
# n=np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])

# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)

# print(comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids))


def comp_unmat_stat_nme(n,  θ1s, θ2s, grad_logZ1s, grad_logZ2s) -> float:
    
    if θ1s.ndim == 1: θ1s = θ1s.reshape(1, -1)
    if θ2s.ndim == 1: θ2s = θ2s.reshape(1, -1)
    if grad_logZ1s.ndim == 1: grad_logZ1s = grad_logZ1s.reshape(1, -1)
    if grad_logZ2s.ndim == 1: grad_logZ2s = grad_logZ2s.reshape(1, -1)

    nme1 = np.mean([comp_nme(n, θ, g) for θ, g in zip(θ1s, grad_logZ1s)])
    nme2 = np.mean([comp_nme(n, θ, g) for θ, g in zip(θ2s, grad_logZ2s)])
    return float(nme1 - nme2)

# # check
# n=np.array([5])
# θ1s = np.array([[0.0,0.0],[0.0,0.0]])
# θ2s = np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_nme(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))


def comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s,θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_nme(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)
# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))


def comp_unmat_stat_pdm(n:np.ndarray,  θ1s:np.ndarray, θ2s:np.ndarray, grad_logZ1s:np.ndarray, grad_logZ2s:np.ndarray) -> float:
    cmds = []
    
    for s1 in range(len(θ1s)):
        for s2 in range(len(θ2s)):
            cmds.append(comp_cmd(n, θ1s[s1], θ2s[s2], grad_logZ1s[s1], grad_logZ2s[s2]))

    return float(np.mean(cmds))


# # check
# n=np.array([5])
# θ1s = np.array([[-1.0,1.0],[-1.0,1.0]])
# θ2s = np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_stat_pdm(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))

def comp_unmat_perm_stat_pdm(n:np.ndarray,  θ1s:np.ndarray, θ2s:np.ndarray, grad_logZ1s:np.ndarray, grad_logZ2s:np.ndarray, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s,θ2s))
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)

    grad_logZ2sp = np.concatenate((grad_logZ1s, grad_logZ2s))
    grad_logZ1sp = grad_logZ2sp[perm_ids]
    grad_logZ2sp = np.delete(grad_logZ2sp, perm_ids, axis=0)

    return comp_unmat_stat_pdm(n, θ1sp, θ2sp, grad_logZ1sp, grad_logZ2sp)

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# perm_ids = [0,2,4,6,8]
# grad_logZ1s = get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print(comp_unmat_perm_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))


#T_PDR
def comp_unmat_stat_pdr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray) -> float:
    pdr1 =  np.mean([comp_pdr(n, θ) for θ in θ1s]) 
    pdr2 =  np.mean([comp_pdr(n, θ) for θ in θ2s]) 
    return float(pdr1 - pdr2)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[1.0,1.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# r1=comp_unmat_stat_pdr(n,θ1s,θ2s)
# r2=0.5*(comp_pdr(n,[0.0,0.0])+comp_pdr(n,[1.0,1.0]))-comp_pdr(n,[1.0,1.0])
# print(r1,r2)

#T_PDR
def comp_unmat_perm_stat_pdr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray, perm_ids: List[int]) -> float:
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_pdr(n, θ1sp, θ2sp)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# perm_ids = [0,2]
# print(comp_unmat_perm_stat_pdr(n, θ1s,θ2s, perm_ids))

#T_CHALM
def comp_unmat_stat_chalm(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray) -> float:
    chalm1 =  np.mean([comp_chalm(n, θ) for θ in θ1s])
    chalm2 =  np.mean([comp_chalm(n, θ) for θ in θ2s]) 
    return float(chalm1 - chalm2)

# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# s1=comp_unmat_stat_chalm(n,θ1s,θ2s)
# s2=comp_chalm(n,[0.0,0.0])-comp_chalm(n,[1.0,1.0])
# print(s1,s2)


#T_CHALM
def comp_unmat_perm_stat_chalm(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray, perm_ids:List[int]) -> float:
    θ2sp = np.concatenate((θ1s, θ2s), axis=0)
    θ1sp = θ2sp[perm_ids]
    θ2sp = np.delete(θ2sp, perm_ids, axis=0)
    return comp_unmat_stat_chalm(n, θ1sp, θ2sp)

# #check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# perm_ids = [0,2]
# print(comp_unmat_perm_stat_chalm(n, θ1s,θ2s, perm_ids))

#T_MCR
def comp_unmat_stat_mcr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray,
                        grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray) -> float:
    tchalm = comp_unmat_stat_chalm(n, θ1s, θ2s)
    tmml = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    return tchalm - tmml
    
# # check
# n=np.array([5])
# θ1s =np.array( [[0.0,0.0],[0.0,0.0]])
# θ2s =np.array( [[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = get_all_grad_logZs(n,θ1s)
# grad_logZ2s= get_all_grad_logZs(n,θ2s)
# m1 = comp_unmat_stat_mcr(n,θ1s,θ2s,grad_logZ1s, grad_logZ2s)
# m2=comp_mcr(n,[0.0,0.0],get_grad_logZ(n, [0.0,0.0]))-comp_mcr(n,[1.0,1.0],get_grad_logZ(n, [1.0,1.0]))
# print(m1,m2)

#T_MCR
def comp_unmat_perm_stat_mcr(n: np.ndarray, θ1s: np.ndarray, θ2s: np.ndarray,
                             grad_logZ1s: np.ndarray, grad_logZ2s: np.ndarray,
                             perm_ids: List[int]) -> float:
    ptchalm = comp_unmat_perm_stat_chalm(n, θ1s, θ2s, perm_ids)
    ptmml = comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids)
    return ptchalm - ptmml
    
# # check
# n=np.array([5,5])
# θ1s =np.array( [[0.0,1.0,0.0],[0.0,1.0,0.0]])
# θ2s =np.array( [[1.0,1.0,1.0],[1.0,1.0,1.0]])
# perm_ids = [0,2]
# grad_logZ1s = get_all_grad_logZs(n,θ1s)
# grad_logZ2s= get_all_grad_logZs(n,θ2s)
# print(comp_unmat_perm_stat_mcr(n,θ1s,θ2s,grad_logZ1s, grad_logZ2s, perm_ids))



##Unpaired group comparison (M1 != M2).
def unmat_tests(n, θ1s, θ2s, Lmax=1000,test_types=None):
    """
    Unpaired group comparison (M1 != M2).
    test_types: list of strings, default is None (computes all).
                Options: ['mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr']
    """
    valid_types = {'mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr'}
    if test_types is None:
        test_types = list(valid_types)
    else:
        test_types = [t for t in test_types if t in valid_types]   

    need_grads = any(t in test_types for t in ['mml', 'nme', 'pdm', 'mcr'])
    
    if need_grads:
        grad_logZ1s = get_all_grad_logZs(n, θ1s)
        grad_logZ2s = get_all_grad_logZs(n, θ2s)
    else:
        grad_logZ1s = grad_logZ2s = None

    diff_stats = {}
    if 'mml' in test_types:
        diff_stats['mml'] = comp_unmat_stat_mml(n, grad_logZ1s, grad_logZ2s)
    if 'nme' in test_types:
        diff_stats['nme'] = comp_unmat_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdm' in test_types:
        diff_stats['pdm'] = comp_unmat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdr' in test_types:
        diff_stats['pdr'] = comp_unmat_stat_pdr(n, θ1s, θ2s)
    if 'chalm' in test_types:
        diff_stats['chalm'] = comp_unmat_stat_chalm(n, θ1s, θ2s)
    if 'mcr' in test_types:
        diff_stats['mcr'] = comp_unmat_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)

    
    # Compute number of possible randomizations
    L = comb(len(θ1s) + len(θ2s), len(θ1s))
    exact = L < Lmax

    # Create iterable object with all combinations
    comb_iter = combinations(range(len(θ1s) + len(θ2s)), len(θ1s))

    # Get group label combinations to use
    if exact:
        comb_iter_used = list(comb_iter)
    else:
        ind_subset = np.random.choice(L, Lmax, replace=False)
        comb_iter_used = [comb for idx, comb in enumerate(comb_iter) if idx in ind_subset]

    perm_results = {t: [] for t in test_types}

    for x in comb_iter_used:
        perm_ids = list(x)
        if 'mml' in test_types:
            perm_results['mml'].append(comp_unmat_perm_stat_mml(n, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'nme' in test_types:
            perm_results['nme'].append(comp_unmat_perm_stat_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'pdm' in test_types:
            perm_results['pdm'].append(comp_unmat_perm_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
        if 'pdr' in test_types:
            perm_results['pdr'].append(comp_unmat_perm_stat_pdr(n, θ1s, θ2s, perm_ids))
        if 'chalm' in test_types:
            perm_results['chalm'].append(comp_unmat_perm_stat_chalm(n, θ1s, θ2s, perm_ids))
        if 'mcr' in test_types:
            perm_results['mcr'].append(comp_unmat_perm_stat_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s, perm_ids))
    
    results = {}
    
    for t in test_types:
        stat_val = diff_stats[t]
        perms = np.array(perm_results[t])
        n_perms = len(perms)
        
        if t == 'pdm':
            if exact:
                pval = np.sum(perms >= stat_val) / n_perms
            else:
                pval = (1.0 + np.sum(perms >= stat_val)) / (1.0 + n_perms)
        else:
            if exact:
                pval = np.sum(np.abs(perms) >= abs(stat_val)) / n_perms
            else:
                pval = (1.0 + np.sum(np.abs(perms) >= abs(stat_val))) / (1.0 + n_perms)
        
        results[t] = (stat_val, float(pval))

    return results
    
   
# #check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# print(unmat_tests(n, θ1s, θ2s,test_types=['pdr', 'mcr']))



##Difference statistics for matched pair i
def comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)->np.ndarray:

    diffs = []
    
    # Iterate over the pairs of gradients and compute the MML difference
    for grad1, grad2 in zip(grad_logZ1s, grad_logZ2s):
        mml_diff = comp_mml(n, grad1) - comp_mml(n, grad2)
        diffs.append(mml_diff)
    return diffs

  
def comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        nme_diff = comp_nme(n, θ1, grad1) - comp_nme(n, θ2, grad2)
        diffs.append(nme_diff)
    return diffs



def comp_mat_diff_pdr(n, θ1s, θ2s):
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        pdr_diff = comp_pdr(n, θ1) - comp_pdr(n, θ2)
        diffs.append(pdr_diff)
    return diffs

def comp_mat_diff_chalm(n, θ1s, θ2s):
    diffs = []
    for θ1, θ2 in zip(θ1s, θ2s):
        chalm_diff = comp_chalm(n, θ1) - comp_chalm(n, θ2)
        diffs.append(chalm_diff)
    return diffs

    
def comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    diffs = []
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        mcr_diff = comp_mcr(n, θ1, grad1) - comp_mcr(n, θ2, grad2)
        diffs.append(mcr_diff)
    return diffs

#or
# def comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):   
#     dchalm = comp_mat_diff_chalm(n, θ1s, θ2s)
#     dmml = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
#     return dchalm-dmml

# # check
# n=np.array([5])
# θ1s=np.array([[0.0,0.0],[0.0,0.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s =get_all_grad_logZs(n, θ1s)
# grad_logZ2s = get_all_grad_logZs(n, θ2s)
# print("mml:",comp_mat_diff_mml(n,grad_logZ1s,grad_logZ2s),"nme:",comp_mat_diff_nme(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s),"pdr:",comp_mat_diff_pdr(n,θ1s,θ2s),"chalm:",comp_mat_diff_chalm(n,θ1s,θ2s),"mcr:",comp_mat_diff_mcr(n,θ1s,θ2s,grad_logZ1s,grad_logZ2s))


def comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s):
    cmds = 0.0
    for θ1, θ2, grad1, grad2 in zip(θ1s, θ2s, grad_logZ1s, grad_logZ2s):
        cmds += comp_cmd(n, θ1, θ2, grad1, grad2)
    return cmds / len(θ1s)

# # check
# n=np.array([5])
# θ1s=np.array([[-1.0,1.0],[-1.0,1.0]])
# θ2s=np.array([[1.0,1.0],[1.0,1.0]])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# print(comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s))



def comp_mat_j_stat(diffs, j):
    diffs_perm = 0.0
    for i in range(len(diffs)):
        diffs_perm += diffs[i] if (j & 1) else -diffs[i]
        j >>= 1
    return diffs_perm/len(diffs)

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# mml_diffs = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
# print(comp_mat_j_stat(mml_diffs, 0))





def comp_mat_perm_stats(diffs, js):
    return [comp_mat_j_stat(diffs, j) for j in js]

# # check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(2)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(2)])
# grad_logZ1s = [get_grad_logZ(n,θ1) for θ1 in θ1s]
# grad_logZ2s = [get_grad_logZ(n,θ2) for θ2 in θ2s]
# mml_diffs = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
# js = list(range(2**len(θ1s)))
# print(comp_mat_perm_stats(mml_diffs,js))






def mat_tests(n, θ1s, θ2s, Lmax=1000, test_types=None):
    """
    配对样本比较。
    test_types: list of strings, default is None (computes all).
                Options: ['mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr']
    """
    valid_types = {'mml', 'nme', 'pdm', 'pdr', 'chalm', 'mcr'}
    if test_types is None:
        test_types = list(valid_types)
    else:
        test_types = [t for t in test_types if t in valid_types]

    # 1. 优化：仅在需要时计算梯度
    need_grads = any(t in test_types for t in ['mml', 'nme', 'pdm', 'mcr'])

    if need_grads:
        grad_logZ1s = get_all_grad_logZs(n, θ1s)
        grad_logZ2s = get_all_grad_logZs(n, θ2s)
    else:
        grad_logZ1s = grad_logZ2s = None

    # 2. 计算差异统计量 (Difference Statistics)
    diff_map = {} 
    results = {}

    if 'mml' in test_types:
        diff_map['mml'] = comp_mat_diff_mml(n, grad_logZ1s, grad_logZ2s)
    if 'nme' in test_types:
        diff_map['nme'] = comp_mat_diff_nme(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    if 'pdr' in test_types:
        diff_map['pdr'] = comp_mat_diff_pdr(n, θ1s, θ2s)
    if 'chalm' in test_types:
        diff_map['chalm'] = comp_mat_diff_chalm(n, θ1s, θ2s)
    if 'mcr' in test_types:
        diff_map['mcr'] = comp_mat_diff_mcr(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
    
    if 'pdm' in test_types:
        # PDM 直接计算统计量，不经过差分数组
        tpdm_stat = comp_mat_stat_pdm(n, θ1s, θ2s, grad_logZ1s, grad_logZ2s)
        results['pdm'] = (tpdm_stat, np.nan)

    # 3. 置换检验设置
    exact = 2**len(θ1s) < Lmax
    if exact:
        js = list(range(2**len(θ1s)))
    else:
        js = [random.randint(0, 2**len(θ1s) - 1) for _ in range(Lmax)]

    # 4. 计算 P 值
    for t in test_types:
        if t == 'pdm': continue

        diffs = diff_map[t]
        perms = comp_mat_perm_stats(diffs, js)
        
        # 计算差异统计量（均值）
        stat_val = sum(diffs) / len(diffs)

        if exact:
            pval = np.sum(np.abs(perms) >= np.abs(stat_val)) / len(perms)
        else:
            pval = (1.0 + np.sum(np.abs(perms) >= np.abs(stat_val))) / (1.0 + len(perms))
        
        results[t] = (stat_val, float(pval))

    return results

# #check
# n = np.array([10])
# θ1s = np.array([[0.5, 0.5] for _ in range(5)])
# θ2s = np.array([[-0.5, 0.5] for _ in range(5)])
# print(mat_tests(n, θ1s, θ2s,test_types=['pdr', 'mcr']))


