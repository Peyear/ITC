import os
import numpy as np
import pandas as pd
import xlrd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.table import Table

os.chdir(r'H:\Articles\SoftwareX ITC\ITC-master') # Files Location
dlsFile='Data7.xls'  # Excel File Name For Drug
dlsFile2='Data1.xls'  # Excel File Name For Control

Delete=1;    # Number of initial points to delete
Clipid=15e-3 # Concentration of the Lipid in the syringe [M]
Cdrug=100e-6 # Concentration of the Drug in the cell [M]
Vw=200e-6   # Volume of the Cell [L]

vL=(1398e-27)*(6.0221409e23)    #Vesicles Volume
aL=(7e-19)*(6.0221409e23)       #Vesicles Area
aD=(3e-19)*(6.0221409e23)       #Drug Area

R=8.314459848; #J/(mol*Kelvin)
T=298.15; #Kelvin
F=96485.336521; #C/mol
eps0=8.85*1e-12; #Farads/meter
eps=78.4; #Farads/meter
SaltC=0.15; #moles/L Salt concentration
zdrug= 1; #Charge of drug

def func1(x, dH, Kp):
    return (Vw*Cdrug)*Vw*Kp*Vinj_add*Clipid*aL*dH*(1e6)/np.power((Vw+(x-0.5)*Vinj_add*(1+Clipid*aL*Kp)),2)
def func2(X, dH, Kp):
    x,Phi = X
    return 1e6*((dH*(Vw*Cdrug)*x*const1*Kp*np.exp(-beta*Phi)/(Vw+x*Vinj+x*const1*Kp*np.exp(-beta*Phi)))-(dH*(Vw*Cdrug)*(x-1)*const1*Kp*np.exp(-beta*Phi)/(Vw+(x-1)*Vinj+(x-1)*const1*Kp*np.exp(-beta*Phi))))

df = pd.read_excel(dlsFile,)
a=df.shape
wb = xlrd.open_workbook(dlsFile)
sheet = wb.sheet_by_index(0)

df2 = pd.read_excel(dlsFile2,)
a2=df2.shape
wb2 = xlrd.open_workbook(dlsFile2)
sheet2 = wb2.sheet_by_index(0)

DH=[]
Vinj=[]
Inj=[]
for i in range(a[0]-1-Delete):
    DH_add=sheet.cell_value(i+1+Delete, 0)
    DH.append(DH_add)
    Vinj_add=sheet.cell_value(i+1+Delete, 1)
    Vinj_add=Vinj_add/1e6
    Vinj.append(Vinj_add)
    Inj.append(i+1)
DH=np.array(DH)
Vinj=np.array(Vinj)
Inj=np.array(Inj)

DH2=[]
for i in range(a2[0]-1-Delete):
    DH_add2=sheet2.cell_value(i+1+Delete, 0)
    DH2.append(DH_add2)

DH3=np.array(DH2)
if DH.shape[0]>DH3.shape[0]:
    for i in range(DH.shape[0]-DH3.shape[0]):
        DH2.append(0)
    DH2=np.array(DH2)
if DH.shape[0]<DH3.shape[0]:
    DH2=np.array(DH2)
    DH2=DH2[range(DH.shape[0])]
DH=DH-DH2

xdata = Inj
ydata = DH

plt.plot(xdata,ydata,'.')
plt.xticks(np.arange(0, np.max(xdata)+1, step=2))
plt.xlabel('Injection Number')
plt.ylabel('\u03BC cal')

Kp=1
dH=ydata[0]*T
dHeat=(0,ydata[0]*np.inf)
for i in range(1000):
    popt, pcov = curve_fit(func1, xdata, ydata, p0=[dH,Kp], bounds=([np.min(dHeat), 1e-10], [np.max(dHeat), 10e10]))
    residuals2 = ydata- func1(xdata, *popt)
    ss_res2 = np.sum(residuals2**2)
    ss_tot2 = np.sum((ydata-np.mean(ydata))**2)
    r_squared2 = 1 - (ss_res2 / ss_tot2)
    dH2=popt[0]
    Kp2=popt[1]
    dG2=-1.9858775*298.15*np.log(Kp2*(aL/vL))
    TdS2=dH2-dG2
    if np.abs(Kp-Kp2)+np.abs(dH-dH2)<=0:
        break
    else:
        Kp=Kp2
        dH=dH2
Sample1=[]
Sample1.append([Kp2*(aL/vL),dH2,dG2,TdS2,r_squared2,ss_res2])
plt.plot(xdata, func1(xdata, *popt))
plt.show()

if zdrug != 0:
    k=np.power(8000*R*T*eps*eps0*SaltC,.5)
    beta=zdrug*F/(R*T)
    const1=Clipid*aL*Vinj_add
    const2=const1*k/(zdrug*F)
    for i in range(1000):
        Phi=[]
        DH=DH/1e6
        for j in range(xdata.shape[0]):
            A=np.sum(DH[range(j+1)])/(xdata[j]*const2*dH)
            B=dH*xdata[j]*const1/(np.sum(DH[range(j+1)])*aD+dH*xdata[j]*const1)
            Phi_=(2/beta)*np.arcsinh(A*B)
            Phi.append(Phi_)
        Phi=np.array(Phi)
        DH=DH*1e6
        popt, pcov = curve_fit(func2, (xdata,Phi), ydata, p0=[dH, Kp], bounds=([-10e10, 1e-10], [10e10, 10e10]))
        residuals2 = ydata- func2((xdata,Phi), *popt)
        ss_res2 = np.sum(residuals2**2)
        ss_tot2 = np.sum((ydata-np.mean(ydata))**2)
        r_squared2 = 1 - (ss_res2 / ss_tot2)
        dH2=popt[0]
        Kp2=popt[1]
        dG2=-1.9858775*298.15*np.log(Kp2*(aL/vL))
        TdS2=dH2-dG2
        if np.abs(Kp-Kp2)+np.abs(dH-dH2)<=0:
            break
        else:
            Kp=Kp2
            dH=dH2
    plt.plot(xdata, Phi*1000,'+')
    plt.xticks(np.arange(0, np.max(xdata)+1, step=2))
    plt.xlabel('Injection Number')
    plt.ylabel('\u03A6 mV')
    plt.show()

Sample1.append([Kp2*(aL/vL),dH2,dG2,TdS2,r_squared2,ss_res2])
t = Table(np.round(np.array(Sample1),decimals=4), names=('Kp','\u0394H','\u0394G','T\u0394S','r\u00b2','SSE'))
print(t)

if zdrug != 0:
    df = pd.DataFrame({'Inj #':xdata,'Phi mV':Phi*1000})
    writer = pd.ExcelWriter(dlsFile[:-4]+'Phi.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Phi',index=False)
    writer.save()

df = pd.DataFrame({'\u0394H':(Sample1[0][0],Sample1[1][0]),'Kp':(Sample1[0][1],Sample1[1][1]),'\u0394G':(Sample1[0][2],Sample1[1][2]),'T\u0394S':(Sample1[0][3],Sample1[1][3]),'r\u00b2':(Sample1[0][4],Sample1[1][4]),'SSE':(Sample1[0][5],Sample1[1][5])})
writer = pd.ExcelWriter((dlsFile[:-4]+'Kp.xlsx'), engine='xlsxwriter')
df.to_excel(writer, sheet_name='Kp',index=False)
writer.save()
