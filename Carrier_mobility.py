import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate 
from scipy.constants import e, h, m_e, pi
#%%

data = np.genfromtxt('30K2.csv',delimiter=',',skip_header=1, dtype=float)

'''Separating useful Magnetic field into x and y directions'''

B = data[:,29]

Ba = []
for i in B:
    Ba.append(i)
    if i == max(B):
        break

Bb = []
for i in B:
    Bb.append(i)
    if i == min(B):
        break

denoiseB = Bb[len(Ba):len(Bb)]

#Transverse

Vxxx = data[:,5]
Vxxx = Vxxx[len(Ba):len(Bb)]

#Longitudional

Vyxx = data[:,1]
Vyxx = Vyxx[len(Ba):len(Bb)]

#%%
'''Scipy interpolation Vxxx'''

interp_amount = 1001
InterpB = np.linspace(denoiseB[0], denoiseB[-1], interp_amount)

def interp_Vxy(denoiseB,V):
    f = interpolate.interp1d(denoiseB, V,'cubic')
    return f(InterpB)


plt.plot(denoiseB, interp_Vxy(denoiseB,Vxxx))
plt.plot(InterpB,interp_Vxy(denoiseB,Vxxx))
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Vxx (V)')
plt.grid()
plt.show()

'''Scipy interpolation Vyxx'''

plt.plot(denoiseB,interp_Vxy(denoiseB,Vyxx))
plt.plot(InterpB,interp_Vxy(denoiseB,Vyxx))
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('pxx (Ω cm)')
plt.grid()
plt.show()


#%%
#Resistances
#Current = 1mA

Rxx = interp_Vxy(denoiseB,Vxxx)/0.001             
Ryx = interp_Vxy(denoiseB,Vyxx)/0.001                        
    
'''Resistivities'''

#Units in m
L = 0.001111
A = 0.000546*0.000166 

pxx = (Rxx*A)/L
pyx = (Ryx*A)/L

'''separating for pxx'''

pxxS = -np.flip((pxx + pxx[::-1] )/2)
pxxA = (pxx - pxx[::-1] )/2

pxxS_format = pxxS*10000
pxxA_format = pxxA*10000

plt.plot(InterpB, pxxS_format, label='')
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Magnetoresistivity (x10^-4 Ω cm)')
plt.title('Magnetoresistivity data for Bi2Se3 sample')
plt.grid()
plt.show()

'''separating for pyx'''
pyxS = (pyx + pyx[::-1] )/2
pyx_Asymetrical = (pyx - pyx[::-1] )/2
pyxA_transpose = np.flip(pyx_Asymetrical)

pyxS_format = pyxS*10000
pyxA_format = pyx_Asymetrical*10000

plt.plot(InterpB, pyxA_format, label='')
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('ρyx0 (x10^-4 Ω cm)')
plt.grid()
plt.show()

#%%

'''Simple one band approximation curve fit'''
def one_band(InterpB, m):
    fit = m*InterpB
    return fit

popt, pcov = curve_fit(one_band, InterpB, pyxA_transpose)
perr = np.sqrt(np.diag(pcov))
for param,val,err in zip('a',popt,perr):
    print('{}={}+/-{}'.format(param,val,err))

plt.plot(InterpB, one_band(InterpB, *popt))
plt.plot(InterpB, pyxA_transpose)
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Hall Resistivity (Ω cm)')
plt.title('Hall Resistivity of Bi2Se3 at 30K')
plt.grid()
plt.show()

#%%
'''Calculating Carrier Density and Mobility from Hall'''

R1_m = popt[0]

Carr_Density1m = -1/(R1_m*e)
Carr_Density1cm = Carr_Density1m/1000000
print(str('Carrier Density 1 is ' + str(Carr_Density1cm)))

pxx0 = 3.7095e-06

def parabola(InterpB, mu, pxx0):
    return pxx0 * (1 + (mu*InterpB)**2)

popt1, pcov1 = curve_fit(parabola, InterpB, pxxS)
perr1 = np.sqrt(np.diag(pcov1))
for param,val,err in zip('ab',popt1,perr1):
    print('{}={}+/-{}'.format(param,val,err))

plt.plot(InterpB, parabola(InterpB, *popt1))
plt.plot(InterpB, pxxS)
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Magnetoresistance (Ω cm)')
plt.title('Magnetoresistance of Bi2Se3')
plt.grid()
plt.show()


#%%

'''Low field one band curve fit'''
newB = np.linspace(-8,8,1001)

#Fitting at low MF
left = 420
right = 581

pxxLow = pxxS[left:right]
BLow = newB[left:right]

def parabola1(BLow, mu, pxx0):
    return pxx0 * (1 + np.square(mu*BLow))

popt2, pcov2 = curve_fit(parabola1, BLow, pxxLow)
perr2 = np.sqrt(np.diag(pcov2))
for param,val,err in zip('ab',popt2,perr2):
    print('{}={}+/-{}'.format(param,val,err))

plt.plot(BLow, parabola1(BLow, *popt2))
plt.plot(BLow, pxxLow)
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Hall Resistivity (Ω cm)')
plt.title('Low Field 1-band fit of $Sb_2Te_3$ at 30K')
plt.grid()
plt.show()

#%%
  
#Calculating Mobility
Mobility = popt2[0]*10000
print(str('Mobility is ' + str(Mobility)))


#Calculating Fermi Energy
EF = ((h**2/(2*m_e))*((3*Carr_Density1m)/(8*pi))**(2/3))/e
print('Fermi Energy is %.1g' % EF + ' eV')


#Calculating Resistivity
Res = 1/(e*Carr_Density1cm*Mobility)
print('Resistivity is %.1g' % Res + ' Ω cm')

#Calculating Conductivity
Conductivity = 1/Res
print('Conductivity is %.1g' % (Conductivity*100) + ' S/m')



#https://www.physics.purdue.edu/quantum/files/chen_tispie_2012.pdf
#http://homepages.wmich.edu/~leehs/ME695/Chapter%2011.pdf
#https://arxiv.org/ftp/arxiv/papers/1408/1408.1614.pdf
#https://drum.lib.umd.edu/bitstream/handle/1903/13999/Kim_umd_0117E_14035.pdf?sequence=1&isAllowed=y
#https://iopscience.iop.org/article/10.1088/0022-3719/19/6/010/pdf
#https://kopernio.com/viewer?doi=10.1088%2F1361-6528%2Faa5&token=WzIyMjc2MjksIjEwLjEwODgvMTM2MS02NTI4L2FhNSJd.E0hvlww_fRH2ePp0X9FN7_UkYds
#https://kopernio.com/viewer?doi=10.1103%2Frevmodphys.82.1539&token=WzIyMjc2MjksIjEwLjExMDMvcmV2bW9kcGh5cy44Mi4xNTM5Il0.i7RX2kpaSEPNnOrj1uRSkB9zDTg