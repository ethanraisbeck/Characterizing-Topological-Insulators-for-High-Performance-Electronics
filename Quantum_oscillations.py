import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi, hbar, elementary_charge, c
from scipy import interpolate
e = elementary_charge

def find_closest(B_field, value):
    array1 = np.abs(B_field)
    zero_index = array1.argmin()
    print(zero_index)
    low_lim_array = np.abs(B_field[0:zero_index] - value)
    high_lim_array = np.abs(np.abs(B_field[zero_index:5000]) - value)
    low_lim = low_lim_array.argmin()
    high_lim = high_lim_array.argmin()
    high_lim = high_lim + zero_index
    return low_lim, high_lim, zero_index


def interpolating(x, y, low_lim=425, high_lim=1268, no_of_pt=0.01):
    x = x[low_lim:high_lim]
    y = y[low_lim:high_lim]
    f = interpolate.interp1d(x, y)
    x_new = np.arange(-7.9, 7.9, no_of_pt)
    return f(x_new), x_new


def decomposing_xx(rho_xx):
    rho_o_xx = (rho_xx[790:1580] + np.flip(rho_xx[0:790])) / 2
    rho_o_xx = np.append(np.flip(rho_o_xx), rho_o_xx)
    return rho_o_xx


def decomposing_yx(rho_yx):
    rho_o_yx = (np.flip(rho_yx[0:790]) - rho_yx[790:1580])
    rho_o_yx = rho_o_yx / 2
    rho_o_yx = np.append(-np.flip(rho_o_yx), rho_o_yx)
    return rho_o_yx


def to_rho(V_yx_x_new, V_xx_x_new, I=1e-3, width_yx=0.0136e-2, width_xx=0.0556e-2, length=0.1584e-2):
    Ryx_x = V_yx_x_new / I
    Rxx_x = V_xx_x_new / I
    rho_yx = Ryx_x * width_yx * length / width_xx
    rho_xx = Rxx_x * width_xx * width_yx / length
    return rho_yx, rho_xx


def to_rho2(V, t, I=100e-6):
    return 2*t*V/I

#%%
file_name = "6221-Lockin-DAQ Multi-Parameter _RvB_9.9K_8to-8T_1mA_ (1).txt"

df = pd.read_csv(file_name, sep="\t", float_precision='high')
df = df.iloc[1500:6000, [2, 30, 6]]
conv_arr = df.values

# split matrix into 3 columns each into 1d array
arr1 = np.delete(conv_arr, [1, 2], axis=1)
arr2 = np.delete(conv_arr, [0, 2], axis=1)
arr3 = np.delete(conv_arr, [0, 1], axis=1)

# converting into 1D array
V_xx_x = arr1.ravel()
B_field = arr2.ravel()
V_yx_x = arr3.ravel()

print(df)


low_lim, high_lim, zero_index = find_closest(B_field, 8)
V_yx_x_new, B_field_new = interpolating(B_field, V_yx_x,
                                        low_lim, high_lim)
V_xx_x_new, _ = interpolating(B_field, V_xx_x, low_lim, high_lim)

rho_yx, rho_xx = to_rho(V_yx_x_new, V_xx_x_new)
rho_xx_new = decomposing_xx(rho_xx) 
rho_yx_new = decomposing_yx(rho_yx) 

plt.plot(B_field_new, rho_xx_new, 'r')

#%%
del V_xx_x_new, V_yx_x_new, B_field, V_xx_x, V_yx_x

def quadratic(B, a, b):
  return a*B**2 + b

popt_qd, pcov_qd = curve_fit(quadratic, B_field_new[700:880], 
                         rho_xx_new[700:880])

plt.plot(B_field_new[700:880], rho_xx_new[700:880], 'r')
plt.plot(B_field_new[700:880], quadratic(B_field_new[700:880], *popt_qd))

print(popt_qd)

def mr_bg(B, c):
  return (1/(popt_qd[0]*B**2 ) + (1/c))**-1

rho_xx_alligned = rho_xx_new - popt_qd[1]
popt_bg, pcov_bg = curve_fit(mr_bg, B_field_new, rho_xx_alligned)


plt.plot(B_field_new, rho_xx_alligned, 'r', label="Resistivity Data")
plt.plot(B_field_new, mr_bg(B_field_new, *popt_bg), label="Saturation of MR")
plt.title("Resisitvity of Bi2Se3 in xx direction under an applied B field")
plt.ylabel("Resistivity (ohm m)")
plt.xlabel("Magnetic Flux Density (B)")
plt.legend()
print(popt_bg)

#%%

rho_xx_osc_bg = -rho_xx_alligned +mr_bg(B_field_new, *popt_bg)
plt.plot(1/B_field_new[990:1580], rho_xx_osc_bg[990:1580], 'r')

f = interpolate.interp1d(1/B_field_new[790:1580], rho_xx_osc_bg[790:1580])
B_field_new3 = np.arange(7.9**-1, 6**-1, 1e-6) 
rho_xx_osc_bg2 = f(B_field_new3) 


plt.plot(B_field_new3, rho_xx_osc_bg2, 'b')
plt.show()

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, argrelmin

sample_rate=1e-6
N=B_field_new3.size 

yf = np.abs(rfft(rho_xx_osc_bg2))
xf = rfftfreq(N, sample_rate)
peaks, _ = find_peaks(yf, height=0.0001)


plt.plot(xf[peaks], yf[peaks], "rx")
plt.xlim(0, 500)
plt.plot(xf, yf, 'b')
plt.annotate("F=153.9T", (xf[peaks+5], yf[peaks]))
plt.title("Fourier Transform of the SdH oscillation")
plt.xlabel("Frequency Domain (T)")
plt.ylabel("Amplitude (A.U.)")
plt.show()

print(xf[peaks])

F=154
k = np.sqrt(F*2*pi*e/(hbar*c*pi))
pi*k**2/(2*pi)**2

index = int(5e4)
B_lf = B_field_new3[0:index]
rho_xx_lf = rho_xx_osc_bg2[0:index]
F=154 

def cosine_function(B, a, b):
  return a*np.cos(2*pi*F*B_lf  + b)

popt_ll, pcov_ll = curve_fit(cosine_function, B_lf, rho_xx_lf)
cos_fit = cosine_function(B_lf, *popt_ll)
plt.plot(B_lf , rho_xx_lf)
plt.plot(B_lf, cos_fit)


min_ll = argrelmin(cos_fit)
min_ll = np.array(B_lf[min_ll])
print(min_ll)
print(min_ll[0]/(min_ll[1]-min_ll[0]))

#%%


"""Landau Fan Diagram"""

def linear_ll_diagram(B, b):
  F_1 =1/F
  return F_1*B+b

min_ll = min_ll[0:4]
start = 20
ll_domain = np.arange(start, int(start+min_ll.size), 1)

popt_ll_diagram, pcov_ll_diagram = curve_fit(linear_ll_diagram, ll_domain, min_ll)
print(popt_ll_diagram*F)

plt.plot(ll_domain, min_ll, 'bo')
plt.plot(np.arange(0, 30), linear_ll_diagram(np.arange(0, 30), *popt_ll_diagram), "r")
plt.grid()

def poly(B_field, a, b):
  return a*B_field + b

popt_poly, pcov_poly = curve_fit(poly, B_field_new[1400:1580], rho_xx_alligned[1400:1580])

plt.plot(B_field_new[1400:1580], rho_xx_alligned[1400:1580])
plt.plot(B_field_new[1400:1580], poly(B_field_new[1400:1580], *popt_poly))

rho_xx_osc_hf = -rho_xx_alligned[1400:1580] + poly(B_field_new[1400:1580], *popt_poly)
plt.plot(1/B_field_new[1400:1580], rho_xx_osc_hf)
plt.show()

f = interpolate.interp1d(1/B_field_new[1400:1580], rho_xx_osc_hf)
B_field_new3 = np.arange(7.8**-1, 6.1**-1, 1e-6)
rho_xx_osc_hf2 = f(B_field_new3) 


plt.plot(B_field_new3, rho_xx_osc_hf2)
plt.show()

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, argrelmin


sample_rate=1e-6
N=B_field_new3.size

yf = np.abs(rfft(rho_xx_osc_hf2))
xf = rfftfreq(N, sample_rate)

peaks, _ = find_peaks(yf, height=1e-6)
plt.plot(xf[peaks], yf[peaks], "x")
plt.xlim(0, 1000)
plt.plot(xf, yf)
plt.show()

print(xf[peaks])