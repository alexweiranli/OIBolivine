from sympy.solvers import solve
from sympy import Symbol
import math
import pandas as pd
import numpy as np

c_sio2 = 48;        # in wt.%
c_alkali  = 5;      # in wt.%
c_h2o = 0;          # in wt.%
P00 = 1;            # in GPa
nn = 1000
minFo = 85
maxFo = 95

def cal(fo0,Tcrys0,kd=0.3):

    Tcrys_fo = pd.DataFrame({'Fo': fo0, 'Tcrys': Tcrys0},index=[0])

    mg_ol = 0.667/(100/fo0)
    fe_ol = 0.667 - mg_ol
    lnDMg0 = -2.158 + 55.09*(P00/Tcrys0) - 6.213 * 0.01*c_h2o + 4430/Tcrys0 + 5.115*0.01*c_alkali
    mg_liq = mg_ol/np.exp(lnDMg0)
    fe_mg_liq = (fe_ol/mg_ol)/kd
    fe_liq = fe_mg_liq*mg_liq

    new_mgfe_ol = (1/fe_mg_liq)/kd
    new_mg_ol = 0.667*(new_mgfe_ol/(1+new_mgfe_ol))
    new_fe_ol = new_mg_ol/new_mgfe_ol
    
    ## add olivine
    vol = 1/nn

    for kk in range(nn):

        ## add a small volume of olivine back to the liquid
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

        ## save the results in a dataframe
        T_prim = T_new[0]
        df__ = pd.DataFrame({'Fo':equilfo, 'Tcrys':T_prim},index=[0])
        Tcrys_fo = pd.concat([Tcrys_fo, df__], ignore_index=True)

        if equilfo >= maxFo:
            break

    return Tcrys_fo