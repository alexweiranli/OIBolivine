## author: Wei-Ran Li (HKU, weiranli@hku.hk)

from sympy.solvers import solve
from sympy import Symbol
import pandas as pd
import numpy as np

P00 = 1;            # in GPa
minFo = 80          # olivine Fo content, min. on the LLD
maxFo = 93          # olivine Fo content, max. on the LLD
nn = 1000

def cal(fo0,Tcrys0,kd=0.3,P00=1,c_sio2=48,c_alkali=5,c_h2o=0):

    mg_ol = 0.667/(100/fo0)
    fe_ol = 0.667 - mg_ol
    lnDMg0 = -2.158 + 55.09*(P00/Tcrys0) - 6.213 * 0.01*c_h2o + 4430/Tcrys0 + 5.115*0.01*c_alkali

    mg_liq = mg_ol/np.exp(lnDMg0)
    fe_mg_liq = (fe_ol/mg_ol)/kd
    fe_liq = fe_mg_liq*mg_liq

    new_mgfe_ol = (1/fe_mg_liq)/kd
    new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
    new_fe_ol = new_mg_ol/new_mgfe_ol
    
    Tcrys_fo = pd.DataFrame({'Fo': fo0, 'Tcrys': Tcrys0,  'MgO_liq': mg_liq, 'FeO_liq': fe_liq},index=[0])

    ## add olivine towards higher Fo
    vol = 1/nn

    for kk in range(nn):

        ## add a small volume of olivine to the liquid
        mg_liq = (vol*new_mg_ol + mg_liq)/(1+vol)
        fe_liq = (vol*new_fe_ol + fe_liq)/(1+vol)
        fe_mg_liq = fe_liq/mg_liq
        DMg = new_mg_ol/mg_liq
        T = Symbol('T')
        T_new = solve(np.log(DMg) - (-2.158 + 55.09*(P00/T) - 6.213*0.01*c_h2o + 4430/T + 5.115*0.01*c_alkali),T)

        ## calculate the olivine mg-fe content in equilibrium with the new liquid
        new_mgfe_ol = (1/fe_mg_liq)/kd
        new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
        new_fe_ol = new_mg_ol/new_mgfe_ol
        equilfo = 100/(1+1/new_mgfe_ol)

        ## save results in a dataframe
        T_prim = T_new[0]
        df__ = pd.DataFrame({'Fo': equilfo, 'Tcrys': T_prim, 'MgO_liq': mg_liq, 'FeO_liq': fe_liq},index=[0])
        Tcrys_fo = pd.concat([Tcrys_fo, df__], ignore_index=True)

        if equilfo >= maxFo:
            break

    ## remove olivine towards lower Fo
    mg_ol = 0.667/(100/fo0)
    fe_ol = 0.667 - mg_ol
    lnDMg0 = -2.158 + 55.09*(P00/Tcrys0) - 6.213 * 0.01*c_h2o + 4430/Tcrys0 + 5.115*0.01*c_alkali
    mg_liq = mg_ol/np.exp(lnDMg0)
    fe_mg_liq = (fe_ol/mg_ol)/kd
    fe_liq = fe_mg_liq*mg_liq

    new_mgfe_ol = (1/fe_mg_liq)/kd
    new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
    new_fe_ol = new_mg_ol/new_mgfe_ol

    vol = -1/nn

    for kk in range(nn):

        mg_liq = (vol*new_mg_ol + mg_liq)/(1+vol)
        fe_liq = (vol*new_fe_ol + fe_liq)/(1+vol)
        DMg = new_mg_ol/mg_liq

        T = Symbol('T')
        T_new = solve(np.log(DMg) - (-2.158 + 55.09*(P00/T) - 6.213*0.01*c_h2o + 4430/T + 5.115*0.01*c_alkali),T)

        fe_mg_liq = fe_liq/mg_liq
        new_mgfe_ol = (1/fe_mg_liq)/kd
        new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
        new_fe_ol = new_mg_ol/new_mgfe_ol

        equilfo = 100/(1+1/new_mgfe_ol)

        ## save results in a dataframe
        T_prim = T_new[0]
        df__ = pd.DataFrame({'Fo': equilfo, 'Tcrys': T_prim, 'MgO_liq': mg_liq, 'FeO_liq': fe_liq},index=[0])
        Tcrys_fo = pd.concat([Tcrys_fo, df__], ignore_index=True)

        if equilfo <= minFo:
            break

    return Tcrys_fo     # a dataframe
   