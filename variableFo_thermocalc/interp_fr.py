from scipy.interpolate import LinearNDInterpolator

def interp(df):
    
    df.fillna(0)
    x = df['liq']
    y = df['pressure'] # in kbar
    z = df['liq_MgO']
    z2 = df['liq_FeO'] - 2*df['liq_O']  # calculate Fe2+ by subtracting 2oxygen from total FeO
    points = list(zip(x,y))
    values  = list(z)
    values2 = list(z2)
    f_mg = LinearNDInterpolator(points,values)
    f_fe = LinearNDInterpolator(points,values2)

    return f_mg, f_fe
