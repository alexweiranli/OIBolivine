## author: Wei-Ran Li (HKU, weiranli@hku.hk)

from sympy.solvers import solve
from sympy import Symbol
import pandas as pd
import numpy as np

P00 = 1;            # in GPa
minFo = 80
maxFo = 91
nn = 1000

c_alkali_ol = 1E-4
c_sio2_ol = 41.0497  #at Fo91

def cal(fo0,Tcrys0,kd=0.3,P00=1,c_sio2 = 48,c_alkali  = 2,c_h2o=0):

    mg_ol0 = 0.667/(100/fo0)
    fe_ol0 = 0.667 - mg_ol0
    lnDMg0 = -2.158 + 55.09*(P00/Tcrys0) - 6.213 * 0.01*c_h2o + 4430/Tcrys0 + 5.115*0.01*c_alkali
    c_sio20 = c_sio2
    c_alkali0 = c_alkali

    lnDFe0 = np.log(kd*np.exp(lnDMg0))
    mg_liq = mg_ol0/np.exp(lnDMg0)
    fe_liq = fe_ol0/np.exp(lnDFe0)
    Tcrys_fo = pd.DataFrame({'Fo': fo0, 'Tcrys': Tcrys0, 'MgO_liq': mg_liq, 'FeO_liq': fe_liq, 'DMgO': np.exp(lnDMg0),'C_SiO2':c_sio20,'C_alka':c_alkali0},index=[0])
    T_prim = Tcrys0
    fe_mg_liq0 = fe_liq/mg_liq

    new_mgfe_ol = (1/fe_mg_liq0)/kd
    new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
    new_fe_ol = new_mg_ol/new_mgfe_ol
    
    ## add olivine towards higher Fo
    vol = 1/nn

    for kk in range(nn):
        ## add a small volume of olivine to the liquid
        mg_liq_new = (vol*new_mg_ol + mg_liq)/(1+vol)
        fe_liq_new = (vol*new_fe_ol + fe_liq)/(1+vol)
        fe_mg_liq_new = fe_liq_new/mg_liq_new

        c_sio2_new = (vol*c_sio2_ol + c_sio2)/(1+vol)
        c_alkali_new = (vol*c_alkali_ol + c_alkali)/(1+vol)
        
        for tt in np.arange(T_prim-10,T_prim+10,0.1):

            lnDMg = -2.158 + 55.09*(P00/tt) - 6.213 * 0.01*c_h2o + 4430/tt + 5.115*0.01*c_alkali_new
            lnDFe = np.log(np.exp(lnDMg) * kd)
            if abs(fe_liq_new*np.exp(lnDFe) + mg_liq_new*np.exp(lnDMg) - 0.667) <= 1e-4:
                T_prim = tt 
                break

        lnDMg = -2.158 + 55.09*(P00/T_prim) - 6.213 * 0.01*c_h2o + 4430/T_prim + 5.115*0.01*c_alkali_new
        lnDFe = np.log(kd*np.exp(lnDMg))

        # ## calculate the olivine mg-fe content in equilibrium with the new liquid
        new_mgfe_ol = (1/fe_mg_liq_new)/kd
        new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
        new_fe_ol = new_mg_ol/new_mgfe_ol
        equilfo = 100/(1+1/new_mgfe_ol)

        ## store the results in a dataframe
        df__ = pd.DataFrame({'Fo': equilfo, 'Tcrys': T_prim, 'MgO_liq': mg_liq_new, 'FeO_liq': fe_liq_new, 'DMgO': np.exp(lnDMg),'DFeO': np.exp(lnDFe),'C_SiO2':c_sio2_new,'C_alka':c_alkali_new},index=[0])
        Tcrys_fo = pd.concat([Tcrys_fo, df__], ignore_index=True)

        mg_liq = mg_liq_new
        fe_liq = fe_liq_new

        c_sio2 = c_sio2_new 
        c_alkali = c_alkali_new 

        if equilfo >= maxFo:
            break


    # # ## remove olivine
    mg_ol0 = 0.667/(100/fo0)
    fe_ol0 = 0.667 - mg_ol0
    lnDMg0 = -2.158 + 55.09*(P00/Tcrys0) - 6.213 * 0.01*c_h2o + 4430/Tcrys0 + 5.115*0.01*c_alkali0
    lnDFe0 = np.log(kd*np.exp(lnDMg0))

    mg_liq = mg_ol0/np.exp(lnDMg0)
    fe_liq = fe_ol0/np.exp(lnDFe0)    
    T_prim = Tcrys0
    
    fe_mg_liq0 = fe_liq/mg_liq
    new_mgfe_ol = (1/fe_mg_liq0)/kd
    new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
    new_fe_ol = new_mg_ol/new_mgfe_ol
    
    ## remove olivine
    vol =  - 1/nn
    c_sio2 = c_sio20
    c_alkali = c_alkali0
    
    for kk in range(nn):
        ## remove a small volume of olivine to the liquid
        mg_liq_new = (vol*new_mg_ol + mg_liq)/(1+vol)
        fe_liq_new = (vol*new_fe_ol + fe_liq)/(1+vol)
        fe_mg_liq_new = fe_liq_new/mg_liq_new

        c_sio2_new = (vol*c_sio2_ol + c_sio2)/(1+vol)
        c_alkali_new = (vol*c_alkali_ol + c_alkali)/(1+vol)
        
        for tt in np.arange(T_prim-10,T_prim+10,0.1):
            lnDMg = -2.158 + 55.09*(P00/tt) - 6.213 * 0.01*c_h2o + 4430/tt + 5.115*0.01*c_alkali_new 
            lnDFe = np.log(np.exp(lnDMg) * kd)

            if abs(fe_liq_new*np.exp(lnDFe) + mg_liq_new*np.exp(lnDMg) - 0.667) <= 1e-4:
                T_prim = tt 
                break

        lnDMg = -2.158 + 55.09*(P00/T_prim) - 6.213 * 0.01*c_h2o + 4430/T_prim + 5.115*0.01*c_alkali_new
        lnDFe = np.log(kd*np.exp(lnDMg))

        # ## calculate the olivine mg-fe content in equilibrium with the new liquid
        new_mgfe_ol = (1/fe_mg_liq_new)/kd
        new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
        new_fe_ol = new_mg_ol/new_mgfe_ol
        equilfo = 100/(1+1/new_mgfe_ol)

        ## store the results in a dataframe
        df__ = pd.DataFrame({'Fo': equilfo, 'Tcrys': T_prim, 'MgO_liq': mg_liq_new, 'FeO_liq': fe_liq_new, 'DMgO': np.exp(lnDMg),'DFeO': np.exp(lnDFe),'C_SiO2':c_sio2_new,'C_alka':c_alkali_new},index=[0])
        Tcrys_fo = pd.concat([Tcrys_fo, df__], ignore_index=True)

        mg_liq = mg_liq_new
        fe_liq = fe_liq_new

        c_sio2 = c_sio2_new 
        c_alkali = c_alkali_new 

        if equilfo <= minFo:
            break

    Tcrys_fo = Tcrys_fo.sort_values(by="Fo")
    Tcrys_fo = Tcrys_fo.reset_index(drop=True)

    return Tcrys_fo
   