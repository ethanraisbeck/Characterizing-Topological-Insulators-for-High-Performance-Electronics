import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate 
from scipy.constants import e

#Importing all temperature data

K2 = np.genfromtxt('2K.csv' ,delimiter=',',skip_header=1, dtype=float)
K3 = np.genfromtxt('3K.csv',delimiter=',',skip_header=1, dtype=float)
K5 = np.genfromtxt('5K.csv',delimiter=',',skip_header=1, dtype=float)
K10 = np.genfromtxt('10K.csv',delimiter=',',skip_header=1, dtype=float)
K15 = np.genfromtxt('15K.csv',delimiter=',',skip_header=1, dtype=float)
K20 = np.genfromtxt('20K.csv',delimiter=',',skip_header=1, dtype=float)
K30 = np.genfromtxt('30K2.csv',delimiter=',',skip_header=1, dtype=float)

lst = [K2,K3,K5,K10,K15,K20,K30]
names = ['2K', '3K', '5K', '10K', '20K', '30K']

# Isolating L/T Voltages

TransV = []
LongV = []

for i in lst:
    TransV.append(i[:,5])
    LongV.append(i[:,1])

#Creating lists for indexed data

usefulB = []
usefulTransV = []
usefulLongV = []

# Indexing all data

for num, i in enumerate(lst):
    B = i[:,29]   
    Ba = []
    for nuum, i in enumerate(B):
        Ba.append(i)
        if i == max(B):
            nuuum = nuum 
            break
    Ba = B[0:nuuum]
    
    Bb = []
    for nuum1, i in enumerate(B):
        Bb.append(i)
        if i == min(B):
            break
    
    denoiseB = Bb[len(Ba):len(Bb)]
    usefulB.append(denoiseB)
  
    for num1, j in enumerate(TransV):
        if num == num1:
            TransVu = j[len(Ba):len(Bb)]
            usefulTransV.append(TransVu)
     
    for num1, k in enumerate(LongV):
        if num == num1:
            LongVu = k[len(Ba):len(Bb)]
            usefulLongV.append(LongVu)

#Scipy interpolation

interp_amount = 1001

InterpBs = []
InterpTrans = []
InterpLong = []

for num, i in enumerate(usefulB):
    for num1, j in enumerate(usefulTransV):
        if num == num1:
            InterpB = np.linspace(i[0], i[-1], interp_amount)
            f = interpolate.interp1d(i, j,'cubic')
            InterpVxxx = f(InterpB)  
            
            InterpBs.append(InterpB)
            InterpTrans.append(InterpVxxx)
        
    for num2, k in enumerate(usefulLongV):
         if num == num2:
            f = interpolate.interp1d(i, k,'cubic')
            InterpVyxx = f(InterpB)  
            InterpLong.append(InterpVyxx)

# Calculate L/T Resistance

I = 0.001
TransR = []
LongR = []

for i in InterpTrans:
    x = i / 0.001
    TransR.append(x)

for j in InterpLong:
    x = j / 0.001
    LongR.append(x)


# Calculate L/T Resistivites in SI units

L = 0.001111
W = 0.000546
H = 0.000116
A = H*W

Transp = []
Longp= []

for i in TransR:
    x = (i*A)/L
    Transp.append(x)
    
for j in LongR:
    x = (j*A)/L
    Longp.append(x)

# Symetrising / Asymetrising 

pyxA = []
pxxS = []

for i in Transp:
    x = np.flip((i - i[::-1] )/2)
    pyxA.append(x)

for j in Longp:
    y = np.flip((j + j[::-1] )/2)
    pxxS.append(y)



#%%
Bnew = InterpBs[0]

'''Full two-band model'''

def hallc(Bnew, R1,R2,p1,p2):
    return ( (( (R1*(p2*p2))  + (R2*(p1*p1)) )*InterpB) + (R1*R2*(R1+R2)*(Bnew**3) ))  / (((p1+p2)**2)  +  (((R1+R2)**2)*(Bnew**2)))

popt, pcov = curve_fit(hallc, Bnew, pyxA[0],
                       p0 = [-5e-7, 5e-6, 1e-6, 5e-6])
                      
perr = np.sqrt(np.diag(pcov))
for param,val,err in zip('abcd',popt,perr):
    print('{}={}+/-{}'.format(param,val,err))
    
plt.plot(Bnew, pyxA[0])
plt.plot(Bnew, hallc(Bnew, *popt), color = 'orange')
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Hall Resistivity (Ω m)')
plt.title('Transverse Resistivity vs Magnetic Flux Density of Sb2Te3')
plt.legend()
plt.grid()
plt.show()


R1_m , R2_m = (popt[0]), (popt[1])

Carr_Density1m = 1/(R1_m*e)
Carr_Density1cm = Carr_Density1m/1000000
nn = Carr_Density1m
print(str('Carrier Density 1 is ' + str(Carr_Density1cm)))

Carr_Density2m = 1/(R2_m*e)
Carr_Density2cm = Carr_Density2m/1000000
pn = Carr_Density2m
print(str('Carrier Density 2 is ' + str(Carr_Density2cm)))


Mobility1 = (1/(Carr_Density1m*e*popt[2]))*10000
un = Mobility1/10000
print(str('Mobility 1 is ' + str(Mobility1)))


Mobility2 = (1/(Carr_Density2m*e*popt[3]))*10000
up = Mobility2/10000
print(str('Mobility 2 is ' + str(Mobility2)))


#%%

'''2 band fit with xx data'''

#Making everything low field
Bnew = Bnew[int(interp_amount*(15/32)):int(interp_amount*(17/32))]
pxxS[1] = pxxS[1][int(interp_amount*(15/32)):int(interp_amount*(17/32))]

def magnetor(R1,R2,p1,p2,Bnew):
    return ((p1*p2)*(p1+p2) + ((p1*R2*R2)+(p2*R1*R1))*(Bnew*Bnew)) / (((p1+p2)*(p1+p2)) + (np.square(R1+R2)*(Bnew*Bnew)))

popt, pcov = curve_fit(magnetor, Bnew, pxxS[1],
                       p0 = [-5e-7, 5e-6, 5e-6, 5e-6],
                       maxfev = 50000)

perr = np.sqrt(np.diag(pcov))
for param,val,err in zip('abcd',popt,perr):#
    print('{}={}+/-{}'.format(param,val,err))
plt.plot(Bnew, magnetor(Bnew, *popt))
plt.plot(Bnew, pxxS[1])
plt.xlabel('Magnetic Flux Density (T)')
plt.ylabel('Magnetoresistance (Ω m)')
plt.grid()
plt.show()

#%%

def quadratic(B, a, b):
  return a*B**2 + b

#Finding pxx0
pxx0 = []
for i in range(0,7,1):
    popt, pcov = curve_fit(quadratic, Bnew, pxxS[i])
    pxx0.append(popt[1])

#Calculating % change

change = []
for i in range(0,7,1):
    change.append(((pxx0[i]-pxx0[0])/pxx0[0])*100)
   
temperatures = np.array([0,2,3,5,10,20,30])
plt.plot(temperatures, change, 'x')
plt.title('% Change in MR at low temperatures in Sb2Te3')
plt.xlabel('Temperature (K)')
plt.ylabel('% Change in Magnetoresistance')
plt.show()


pxx0c = ( (1/(pn*e*up) ) + (1/(nn*e*un)))**-1
print(pxx0c)

'''Mobility graph'''

change = np.array(change)
mobility_change = -change

curve = np.polyfit(temperatures,mobility_change,7)
pl = np.poly1d(curve)

def polyfit(temperatures,mobility_change):
    coefs = np.polyfit(temperatures,mobility_change,deg=8)
    p_obj = np.poly1d(coefs) 
    return p_obj
    
p_obj = polyfit(temperatures,mobility_change)
x_line = np.linspace(min(temperatures), max(temperatures), 30) 
y_line = p_obj(x_line)


plt.plot(temperatures, mobility_change, 'x')
plt.plot(x_line,y_line, 'r--')
plt.title('Percentage Change in Mobility at low temperatures in $Sb_2Te_3$',fontsize=(11))
plt.xlabel('Temperature (K)')
plt.ylabel('Percentage Change in Mobility (%)')
plt.show()



