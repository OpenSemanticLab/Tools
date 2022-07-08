# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:39:29 2021

@author: jo28dohe
"""
import numpy as np
import scipy.constants as const
try:
    from LAP_eval import refractive_index as ri
except: 
    import refractive_index as ri



def ref_E(n_1,n_2):
    return (n_1-n_2)/(n_1+n_2)
    
def trans_E(n_1,n_2):
    if np.all(n_1<n_2):
        return 2*n_1/(n_1+n_2)
    else: 
        return 2*n_2/(n_1+n_2)

def Planck_lamb(lamb,A,T):
    return 2*const.pi*const.h*const.c**2*A/lamb**5/(np.exp(const.h*const.c/lamb/const.k/T)-1)




def coupling(lamb_list,d,n_Au_list,n_SiC_list,E_0=1,
             extra_imag_Au=0,extra_imag_SiC=0.14,scatter_param=0.83):
    
    r_Au=ref_E(1,n_Au_list)
    r_Au = scatter_param*r_Au+extra_imag_Au*1j ## with correction for imperfect scattering and refelction phase
    k=2*const.pi/lamb_list
    r_SiC = ref_E(1,n_SiC_list)+extra_imag_SiC*1j
    t_SiC = trans_E(n_SiC_list,1)+extra_imag_SiC*1j
    return (n_SiC_list**2*np.abs(E_0*t_SiC*(1+((1+r_SiC)*r_Au*np.exp(2j*k*d)/(1-r_Au*r_SiC*np.exp(2j*k*d)))))**2)



def thermal_radiation_mirror(lamb,d,A,T,n_Au,n_SiC,extra_imag_Au=0,extra_imag_SiC=0.14,scatter_param=0.83):
    return(Planck_lamb(lamb,A,T)*coupling(lamb,d,n_Au,n_SiC,E_0=1,
                                          extra_imag_Au=extra_imag_Au,extra_imag_SiC=extra_imag_SiC,scatter_param=scatter_param))