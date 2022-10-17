import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import elementary_charge
from preprocssing import interpolating, decomposing, to_rho

e = elementary_charge
df = pd.read_csv("30K 6221-Lockin-DAQ Multi-Parameter _RvB_8to-8T_1mA.txt", sep="\t", float_precision='high')
df = df.loc[425:1268, ["Vxx::X", "multi[1]:Control:Magnet Output", "Vyx::X"]]
conv_arr = df.values

'''split matrix into 3 columns each into 1d array'''
arr1 = np.delete(conv_arr, [1, 2], axis=1)
arr2 = np.delete(conv_arr, [0, 2], axis=1)
arr3 = np.delete(conv_arr, [0, 1], axis=1)

'''converting into 1D array'''
V_xx_x = arr1.ravel()
B_field = arr2.ravel()
V_yx_x = arr3.ravel()

V_yx_x_new, B_field_new = interpolating(B_field, V_yx_x)
V_xx_x_new, B_field_new = interpolating(B_field, V_xx_x)

rho_yx, rho_xx = to_rho(V_yx_x_new, V_xx_x_new)

rho_xx_new, rho_yx_new = decomposing(rho_xx, rho_yx)

'''Reduced one band first attempt'''
def reduced_model1(B,n,  miu_e):
    miu_h = 100
    return (B/e)*(n*miu_h**2 - n*miu_e**2)/(n*miu_h + n*miu_e)**2

popt, pcov = curve_fit(reduced_model1, B_field_new, rho_yx_new/100,
                       p0=[1e27,  1000],
                       bounds=((1e26, 0),
                               (1e28,  2000)))
print(popt)
print(np.sqrt(np.diag(pcov)))

output = pd.DataFrame({"B_field": B_field_new, "rho_yx": rho_yx_new, "rho_xx": rho_xx_new})

plt.plot(B_field_new, reduced_model1(B_field_new,*popt), 'b', label="Two band model fit")
plt.plot(B_field_new,rho_yx_new, 'r', label="Resistivity Data")
plt.grid()
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Resistivity in yx Direction (ohm/cm)')
plt.title("Two Band Model Fit of Resistivity in yx Direction")
plt.legend()
plt.show()

