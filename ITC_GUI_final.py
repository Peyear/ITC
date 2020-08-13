import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import sys
import tkinter as Tk
import tkinter.filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy.signal import butter, lfilter
import pandas as pd
import xlrd
from astropy.table import Table
import xlsxwriter
from tkinter import ttk 
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)

root = Tk.Tk()

#Initialize global variables for analysis
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

#Initializing GUI global variables
sheet = None
a = None
dh_drug = None
injv_drug = None
sheet2 = None
a2 = None
dh_control = None
injv_control = None
savename=''
CD = None
label5 = None
drug_charge_prev = None
canvas=None
canvas2=None
lipidselect=None
lipid_col=3
label4 = None
label3=None

#Reset GUI
def Reset1():
    python = sys.executable
    os.execl(python, python, * sys.argv)

#Initial GUI screen, select data directory/files
def Command1():
    global CD
    global label3
    global label4
    global label5
    if label3 != None:
        label3.destroy()
    if savename == '': #Choosing directory for the first time
        root.directory = Tk.filedialog.askdirectory()
        CD=root.directory
        label3=Tk.Label(root,text=CD)
        label3.grid(row=0,column=1,sticky=Tk.W,columnspan=6)
        label4=Tk.Label(root,text='Select ITC Experiment File - Drug')
        label4.grid(row=1,column=1,sticky=Tk.W)    
        label5=Tk.Label(root,text='Select ITC Background File - Control (Optional)')
        label5.grid(row=2,column=1,sticky=Tk.W)
    else: #Reselecting directory
        label4.destroy()
        label5.destroy()
        root.directory = Tk.filedialog.askdirectory()
        CD=root.directory
        label3=Tk.Label(root,text=CD)
        label3.grid(row=0,column=1,sticky=Tk.W,columnspan=6)
        label4=Tk.Label(root,text='Select ITC Experiment File - Drug')
        label4.grid(row=1,column=1,sticky=Tk.W)  
        label5=Tk.Label(root,text='Select ITC Background File - Control (Optional)')
        label5.grid(row=3,column=1,sticky=Tk.W)

#Choose drug file
def Drug1():
    global a
    global sheet
    global savename
    global dh_drug
    global injv_drug
    global label1001
    global entry1001
    global label1002
    global entry1002
    global label1003
    global entry1003
    global label1004
    global entry1004
    global label1005
    global entry1005
    global label1006
    global entry1006
    global label1007
    global entry1007
    global label1008
    global entry1008
    global label1009
    global entry1009
    global button3
    global label5
    global label4
    global button99
    global lipidselect
    global label_lip_area
    global label_lip_area_e
    global label_lip_thickness
    global label_lip_thickness_e
    global button10
    global lipid_col
    
    #User can choose experiment drug excel file, which is then read
    root.filename =  Tk.filedialog.askopenfilename(initialdir = root.directory,title = "Select file",filetypes = (("XLS","*.xls"),("XLSX","*.xlsx"),("all files","*.*")))
    df = pd.read_excel(root.filename,)
    a=df.shape
    wb = xlrd.open_workbook(root.filename)
    sheet = wb.sheet_by_index(0)
    label4.destroy()    
    label4=Tk.Label(root,text=root.filename)
    label4.grid(row=1,column=1,sticky=Tk.W)
    savename=root.filename
    
    #User can select columns for the heat and injection volume from excel file
    button3.destroy()
    label5.destroy()
    button99.destroy()
    labeldh1=Tk.Label(root,text='Column for Heat (DH):')
    labelinjv1=Tk.Label(root,text='Column for Injection Volume (INJV):')
    entrydh1=Tk.Entry(root,width=5)
    entryinjv1=Tk.Entry(root,width=5)
    labeldh1.grid(row=1,column=3,sticky=Tk.W,pady=(2,2), padx=(5,0))
    entrydh1.grid(row=1,column=4,sticky=Tk.W)
    entrydh1.insert(Tk.END, '0')
    dh_drug = int(entrydh1.get())
    labelinjv1.grid(row=2,column=3,sticky=Tk.W,pady=(2,2), padx=(5,0))
    entryinjv1.grid(row=2,column=4,sticky=Tk.W)
    entryinjv1.insert(Tk.END, '1')
    injv_drug = int(entryinjv1.get())

    #Moving buttons and labels in GUI to make it look nice
    button3=Tk.Button(text='Select Control Background File',fg='blue',command=Background1,width=25)
    button3.grid(row=3,column=0,sticky=Tk.W)
    label5=Tk.Label(root,text='Select ITC Background File - Control (Optional)')
    label5.grid(row=3,column=1,sticky=Tk.W)
    button99=Tk.Button(text='Run ITC Analysis',fg='black',command=testing,height=5,width=25)
    button99.grid(row=4,column=0,sticky=Tk.W,columnspan=2,rowspan=5)

    lipid_col = 5
    label1001.grid(row=1,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1001.grid(row=1,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label1002.grid(row=2,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1002.grid(row=2,column=6,sticky=Tk.W,pady=(2,2), padx=(0,))
    label1003.grid(row=3,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1003.grid(row=3,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label1004.grid(row=4,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1004.grid(row=4,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label1007.grid(row=8,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1007.grid(row=8,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label1008.grid(row=9,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1008.grid(row=9,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label1009.grid(row=10,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))
    entry1009.grid(row=10,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))

    button8.grid(row=0,column=8,sticky=Tk.E)

    label10100.grid(row=5,column=5,sticky=Tk.W,pady=(2,2), padx=(15,0))

    labelarrow.grid(row=6,column=6,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
    label1005.grid(row=6,column=6,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
    entry1005.grid(row=6,column=8,sticky=Tk.W,pady=(2,2), padx=(0,5))
    label1006.grid(row=7,column=6,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
    entry1006.grid(row=7,column=8,sticky=Tk.W,pady=(2,2), padx=(0,5))
    label_lip_area.grid(row=6,column=5,sticky=Tk.W,pady=(2,2), padx=(90,0))
    label_lip_area_e.grid(row=6,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    label_lip_thickness.grid(row=7,column=5,sticky=Tk.W,pady=(2,2), padx=(90,0))
    label_lip_thickness_e.grid(row=7,column=6,sticky=Tk.W,pady=(2,2), padx=(0,0))
    lipidselect.grid(column = 6, row = 5, sticky= Tk.W) 
    button10.grid(row=5,column=7,sticky=Tk.W,pady=(2,2), padx=(5,0)) #15

#Choose background file
def Background1():
    global sheet2
    global a2
    global dh_control
    global injv_control
    global button99
    global label5
    label5.destroy()

    #User can choose experiment drug excel file, which is then read
    root.filename =  Tk.filedialog.askopenfilename(initialdir = root.directory,title = "Select file",filetypes = (("XLS","*.xls"),("XLSX","*.xlsx"),("all files","*.*")))
    df2 = pd.read_excel(root.filename,)
    a2=df2.shape
    wb2 = xlrd.open_workbook(root.filename)
    sheet2 = wb2.sheet_by_index(0)
    label5=Tk.Label(root,text=root.filename)
    label5.grid(row=3,column=1,sticky=Tk.W)

    #User can select columns for the heat and injection volume from excel file
    labeldh2=Tk.Label(root,text='Column for Heat (DH):')
    labelinjv2=Tk.Label(root,text='Column for Injection Volume (INJV):')
    entrydh2=Tk.Entry(root,width=5)
    entryinjv2=Tk.Entry(root,width=5)
    labeldh2.grid(row=3,column=3,sticky=Tk.W,pady=(2,2), padx=(5,0))
    entrydh2.grid(row=3,column=4,sticky=Tk.W)
    entrydh2.insert(Tk.END, '0')
    dh_control = int(entrydh2.get())
    labelinjv2.grid(row=4,column=3,sticky=Tk.W,pady=(2,2), padx=(5,0))
    entryinjv2.grid(row=4,column=4,sticky=Tk.W)
    entryinjv2.insert(Tk.END, '1')
    injv_control = int(entryinjv2.get())
    button99.destroy()
    button99=Tk.Button(text='Run ITC Analysis',fg='black',command=testing,height=5,width=25)
    button99.grid(row=5,column=0,sticky=Tk.W,columnspan=2,rowspan=5)

#Run ITC analysis
def testing():
    global sheet
    global a
    global sheet2
    global a2
    global savename
    global CD
    global injv_drug
    global injv_control
    global dh_drug
    global dh_control
    global drug_charge_prev
    global canvas
    global canvas2
    savename = savename.split('.')[0]

    #Get parameters from GUI
    Delete=int(entry1001.get())
    Clipid=float(entry1002.get())
    Cdrug=float(entry1003.get())
    Vw=float(entry1004.get())

    vL_str = entry1005.get()
    if '*' in vL_str:
        vL_arr = vL_str.split('*')
        if '(' in vL_str:
            vL =float(vL_arr[0].strip("()"))*float(vL_arr[1].strip("()"))
        else:
            vL =float(vL_arr[0])*float(vL_arr[1])
    else:
        if '(' in vL_str:
            vL = float(vL_str.strip("()"))
        else:
            vL = float(vL_str)

    aL_str = entry1006.get()
    if '*' in aL_str:
        aL_arr = aL_str.split('*')
        if '(' in aL_str:
            aL =float(aL_arr[0].strip("()"))*float(aL_arr[1].strip("()"))
        else:
            aL =float(aL_arr[0])*float(aL_arr[1])
    else:
        if '(' in aL_str:
            aL = float(aL_str.strip("()"))
        else:
            aL = float(aL_str)
    
    aD_str = entry1007.get()
    if '*' in aD_str:
        aD_arr = aD_str.split('*')
        if '(' in aD_str:
            aD =float(aD_arr[0].strip("()"))*float(aD_arr[1].strip("()"))
        else:
            aD =float(aD_arr[0])*float(aD_arr[1])
    else:
        if '(' in aD_str:
            aD = float(aD_str.strip("()"))
        else:
            aD = float(aD_str)

    R=8.314459848; #J/(mol*Kelvin)
    T=298.15; #Kelvin
    F=96485.336521; #C/mol
    eps0=8.85*1e-12; #Farads/meter
    eps=78.4; #Farads/meter
    SaltC=float(entry1008.get())
    zdrug=int(entry1009.get())

    #Define fit functions used for Kp and Phi
    def func1(x, dH, Kp):
        return (Vw*Cdrug)*Vw*Kp*Vinj_add*Clipid*aL*dH*(1e6)/np.power((Vw+(x-0.5)*Vinj_add*(1+Clipid*aL*Kp)),2)
    def func2(X, dH, Kp):
        x,Phi = X
        return 1e6*((dH*(Vw*Cdrug)*x*const1*Kp*np.exp(-beta*Phi)/(Vw+x*Vinj+x*const1*Kp*np.exp(-beta*Phi)))-(dH*(Vw*Cdrug)*(x-1)*const1*Kp*np.exp(-beta*Phi)/(Vw+(x-1)*Vinj+(x-1)*const1*Kp*np.exp(-beta*Phi))))

    #Getting values for heats and injection volumes from excel file
    DH=[]
    Vinj=[]
    Inj=[]
    for i in range(a[0]-1-Delete):
        DH_add=sheet.cell_value(i+1+Delete, dh_drug)
        DH.append(DH_add)
        Vinj_add=sheet.cell_value(i+1+Delete, injv_drug)
        Vinj_add=Vinj_add/1e6
        Vinj.append(Vinj_add)
        Inj.append(i+1)
    DH=np.array(DH)
    Vinj=np.array(Vinj)
    Inj=np.array(Inj)

    if sheet2 != None:
        DH2=[]
        for i in range(a2[0]-1-Delete):
            DH_add2=sheet2.cell_value(i+1+Delete, dh_drug)
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

    #Clears previous graphs and output text if present
    if drug_charge_prev != None:
        for ix in range(4):
            labempty = Tk.Label(root,text=' ' * 280)
            labempty.grid(row=14+ix,column=0,columnspan=4)
        canvas.get_tk_widget().destroy()
    if drug_charge_prev != 0 and canvas2 != None:
        canvas2.get_tk_widget().destroy()

    #Display Kp graph
    f = Figure(figsize=(5, 4), dpi=100)
    aa = f.add_subplot(111)
    aa.plot(xdata,ydata,'.')
    aa.set_xlabel('Injection Number')
    aa.set_xticks(np.arange(0, np.max(xdata)+1, step=2))
    aa.set_ylabel('\u03BC cal')
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    if zdrug == 0:
        canvas.get_tk_widget().grid(row=12,column=1, columnspan=3, pady=10, padx=10)
    else:
        canvas.get_tk_widget().grid(row=12,column=0, columnspan=2, pady=10, padx=10)

    #Fit for Kp
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
    aa.plot(xdata, func1(xdata, *popt))
    f.savefig(CD+'/figure1_Kp.png')
    xdata_np = np.array(xdata)
    fit_yvals = np.array(func1(xdata, *popt))
    fit_table2 = []
    for fiti in range(0,len(xdata_np)):
        xx = xdata[fiti]
        yy = fit_yvals[fiti]
        minilst = [xx,yy]
        fit_table2.append(minilst)
    fit_table2 = np.array(fit_table2)
    fit_df = pd.DataFrame(data=fit_table2, index=None, columns=["Injection Number", "Heat (ucal)"])
    writer = pd.ExcelWriter((savename+'fit.xlsx'), engine='xlsxwriter')
    fit_df.to_excel(writer, sheet_name='fit',index=False)
    writer.save()

    #Phi calculations and fit
    if zdrug != 0:
        k=np.power(8000*R*T*eps*eps0*SaltC,.5)
        beta=zdrug*F/(R*T)
        const1=Clipid*aL*Vinj_add
        const2=const1*k/(zdrug*F)
        for i in range(1000):
            Phi=[]
            DH=DH/1e6
            for j in range(xdata.shape[0]):
                A1=zdrug*F/(np.power(8000*R*T*eps*eps0,.5)*(np.power(SaltC,.5)))
                B1=np.sum(DH[range(j+1)])/((xdata[j]*Vinj_add*Clipid*aL*dH)+(np.sum(DH[range(j+1)])*aD))
                A=np.sum(DH[range(j+1)])/(xdata[j]*const2*dH)
                B=dH*xdata[j]*const1/(np.sum(DH[range(j+1)])*aD+dH*xdata[j]*const1)
                Phi_=(2/beta)*np.arcsinh(A1*B1)*(zdrug/abs(zdrug))
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
        #Display Phi graph
        f2 = Figure(figsize=(5, 4), dpi=100)
        aa2 = f2.add_subplot(111)
        aa2.plot(xdata, Phi*1000,'+')
        aa2.set_xlabel('Injection Number')
        aa2.set_xticks(np.arange(0, np.max(xdata)+1, step=2))
        aa2.set_ylabel('\u03A6 mV')
        f2.savefig(CD+'/figure2_Phi.png')
        canvas2 = FigureCanvasTkAgg(f2, master=root)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=12,column=3, columnspan=5, pady=10, padx=10)
    
    Sample1.append([Kp2*(aL/vL),dH2,dG2,TdS2,r_squared2,ss_res2])
    t = Table(np.round(np.array(Sample1),decimals=4), names=('Kp','\u0394H','\u0394G','T\u0394S','r\u00b2','SSE'))

    #Display results of analysis
    if zdrug == 0:
        charge_print = '0'
    elif zdrug > 0:
        charge_print = '+' + str(zdrug) + '  '
    elif zdrug < 0:
        charge_print = str(zdrug) + '  '

    if zdrug == 0:
        Sample1_copy = np.round(np.array(Sample1[0][:]),decimals=4)
        Sample1_copy = np.transpose(np.reshape(Sample1_copy, (-1,1)))
        all_charges=['0    ']
    else:
        Sample1_copy = np.round(np.array(Sample1),decimals=4)
        all_charges=['0    ',charge_print]
    outrow,outcol = np.shape(Sample1_copy)
    table_rows = ['Kp','\u0394H','\u0394G','T\u0394S','r\u00b2','SSE']
    for ix in range(outrow+2):
        row_txt = " \t"
        for iy in range(outcol):
            if ix == 0:
                val_t = str(Sample1_copy[0,iy])
                char = len(val_t)
                w_space = " " * 6
                name_t = table_rows[iy]
                extra1 = " " * (int(np.floor(char/2))-int(np.ceil(len(name_t)/2)))
                extra2 = " " * (int(np.ceil(char/2))-int(np.floor(len(name_t)/2)))
                row_txt = row_txt + w_space + extra1 + name_t + extra2 + w_space + "\t"
            if ix == 1:
                val_t = str(Sample1_copy[0,iy])
                char = len(val_t) + 1
                w_space = " " * 3
                barr = "-" * char
                extra = " " * 3
                row_txt = row_txt + w_space + barr + extra + "\t"
            if ix > 1:
                val_t = str(Sample1_copy[ix-2,iy])
                w_space = " " * 3
                row_txt = row_txt + w_space + val_t + w_space + "\t"
            if ix == 0 and iy == outcol-1:
                lab = Tk.Label(root,text=row_txt)
                lab.grid(row=14+ix,column=1,columnspan=3)
            elif ix == 1 and iy == outcol-1:
                lab = Tk.Label(root,text=row_txt)
                lab.grid(row=14+ix,column=1,columnspan=3)
            elif  ix > 1 and iy == outcol-1:
                lab = Tk.Label(root,text=row_txt)
                lab.grid(row=14+ix,column=1,columnspan=3)
                lab_charge = Tk.Label(root,text='\t Calculations for Z(Drug) = '+all_charges[ix-2]+' --->', justify=Tk.LEFT)
                lab_charge.grid(row=14+ix,column=0,columnspan=2,sticky=Tk.W)
                if zdrug == 0:
                    labempty = Tk.Label(root,text=' ' * 280)
                    labempty.grid(row=14+ix+1,column=0,columnspan=4)

    #Save data
    if zdrug != 0:
        df = pd.DataFrame({'Inj #':xdata,'Phi mV':Phi*1000})
        writer = pd.ExcelWriter(savename+'Phi.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Phi',index=False)
        writer.save()

    df = pd.DataFrame({'\u0394H':(Sample1[0][0],Sample1[1][0]),'Kp':(Sample1[0][1],Sample1[1][1]),'\u0394G':(Sample1[0][2],Sample1[1][2]),'T\u0394S':(Sample1[0][3],Sample1[1][3]),'r\u00b2':(Sample1[0][4],Sample1[1][4]),'SSE':(Sample1[0][5],Sample1[1][5])})
    writer = pd.ExcelWriter((savename+'Kp.xlsx'), engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Kp',index=False)
    writer.save()
    drug_charge_prev = zdrug

#Functionality for user to select the lipid in GUI
def lipidparams(lol):
    global lipidselect
    x = lipidselect.get()
    global label1005
    global label1006
    global entry1005
    global entry1006
    global label_lip_area
    global label_lip_area_e
    global label_lip_thickness
    global label_lip_thickness_e
    global labelarrow
    global lipid_col
    labelarrow.destroy()
    label1005.destroy()
    entry1005.destroy()
    label1006.destroy()
    entry1006.destroy()
    label_lip_area.destroy()
    label_lip_area_e.destroy()
    label_lip_thickness.destroy()
    label_lip_thickness_e.destroy()
    if x == ' Other (lipid area/thickness)':
        label_lip_area = Tk.Label(root,text='Area/Molecule of Lipid (\u212B\u00B2) =')
        label_lip_thickness = Tk.Label(root,text='Hydrocarbon Thickness (\u212B) =') 
        label_lip_area_e=Tk.Entry(root, width=10)
        label_lip_area_e.insert(Tk.END, '68.7')
        label_lip_thickness_e=Tk.Entry(root, width=10)
        label_lip_thickness_e.insert(Tk.END, '17')
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1006=Tk.Entry(root, width=25)
        label_lip_area.grid(row=6,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_area_e.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label_lip_thickness.grid(row=7,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_thickness_e.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' Other (vesicle volume/area)':
        label1005=Tk.Label(root,text='Vesicles Volume=')
        label1006=Tk.Label(root,text='Vesicles Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '(1398e-27)*(6.0221409e23)')
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '(7e-19)*(6.0221409e23)')
        label1005.grid(row=6,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        entry1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label1006.grid(row=7,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        entry1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
    elif x == ' DC18:1PC':
        label_lip_area = Tk.Label(root,text='Area/Molecule of Lipid (\u212B\u00B2) =')
        label_lip_thickness = Tk.Label(root,text='Hydrocarbon Thickness (\u212B) =') 
        label_lip_area_e=Tk.Entry(root, width=10)
        label_lip_area_e.insert(Tk.END, '70')
        label_lip_thickness_e=Tk.Entry(root, width=10)
        label_lip_thickness_e.insert(Tk.END, '13.5')
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:4] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label_lip_area.grid(row=6,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_area_e.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label_lip_thickness.grid(row=7,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_thickness_e.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' DC20:1PC':
        label_lip_area = Tk.Label(root,text='Area/Molecule of Lipid (\u212B\u00B2) =')
        label_lip_thickness = Tk.Label(root,text='Hydrocarbon Thickness (\u212B) =') 
        label_lip_area_e=Tk.Entry(root, width=10)
        label_lip_area_e.insert(Tk.END, '69')
        label_lip_thickness_e=Tk.Entry(root, width=10)
        label_lip_thickness_e.insert(Tk.END, '15.3')
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:6] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label_lip_area.grid(row=6,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_area_e.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label_lip_thickness.grid(row=7,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_thickness_e.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' DC22:1PC':
        label_lip_area = Tk.Label(root,text='Area/Molecule of Lipid (\u212B\u00B2) =')
        label_lip_thickness = Tk.Label(root,text='Hydrocarbon Thickness (\u212B) =') 
        label_lip_area_e=Tk.Entry(root, width=10)
        label_lip_area_e.insert(Tk.END, '68.7')
        label_lip_thickness_e=Tk.Entry(root, width=10)
        label_lip_thickness_e.insert(Tk.END, '17')
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:6] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label_lip_area.grid(row=6,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_area_e.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label_lip_thickness.grid(row=7,column=lipid_col,sticky=Tk.W,pady=(2,2), padx=(90,0))
        label_lip_thickness_e.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(0,0))
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))

#Calculate vesicle molar volume and area based on user inputs for lipid area and hydrocarbon thickness
def compute_vol_area(molarea, thickness, injv, lipid_conc):
    num_lipids = injv * lipid_conc * 6.0221409e23
    vol_vesicle = 4*np.pi*((65e-9)**2)*2*thickness*(10**-10)
    lipid_per_ves = vol_vesicle / ((molarea*thickness)*(10**-30))
    num_vesicles = num_lipids / lipid_per_ves
    tot_ves_vol = vol_vesicle * num_vesicles
    molar_ves_vol = tot_ves_vol / (injv * lipid_conc * (10**-3))
    area_vesicle = 4*np.pi*((65e-9)**2) + 4*np.pi*((65e-9 - (2*thickness*(10**-10)))**2)
    tot_ves_area = area_vesicle * num_vesicles
    molar_ves_area = tot_ves_area / (injv * lipid_conc)
    return(molar_ves_vol/6.0221409e23,molar_ves_area/6.0221409e23)

#Functionality for compute button based lipid parameters inputed in GUI
def lipidparams2():
    global lipidselect
    if lipidselect != None:
        x = lipidselect.get()
    global label1005
    global label1006
    global entry1005
    global entry1006
    global label_lip_area
    global label_lip_area_e
    global label_lip_thickness
    global label_lip_thickness_e
    global labelarrow
    global lipid_col
    if x != ' Other (vesicle volume/area)':
        labelarrow.destroy()
        label1005.destroy()
        entry1005.destroy()
        label1006.destroy()
        entry1006.destroy()
    if x == ' Other (lipid area/thickness)':
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        if 'e' not in str(vmL)[:6]:
            vmL_short = str(vmL)[:6] +'e'+str(vmL).split('e')[1]
        else:
            vmL_short = str(vmL)[:4] +'e'+str(vmL).split('e')[1]
        if 'e' not in str(amL)[:5]:
            amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        else:
            amL_short = str(amL)[:4] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' Other (vesicle volume/area)':
        pass
    elif x == ' DC18:1PC':
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:4] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' DC20:1PC':
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:6] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
    elif x == ' DC22:1PC':
        labelarrow=Tk.Label(root,text='-->')
        labelarrow.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
        vmL,amL = compute_vol_area(float(label_lip_area_e.get()), float(label_lip_thickness_e.get()), 2e-6, float(entry1002.get()))
        vmL_short = str(vmL)[:6] +'e'+str(vmL).split('e')[1]
        amL_short = str(amL)[:5] +'e'+str(amL).split('e')[1]
        label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
        label1006=Tk.Label(root,text=' Vesicle Molar Area=')
        entry1005=Tk.Entry(root, width=25)
        entry1005.insert(Tk.END, '('+vmL_short+')*(6.0221409e23)') #7.033e-1
        entry1006=Tk.Entry(root, width=25)
        entry1006.insert(Tk.END, '('+amL_short+')*(6.0221409e23)') #3.9269e5 
        label1005.grid(row=6,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1005.grid(row=6,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))
        label1006.grid(row=7,column=lipid_col+1,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
        entry1006.grid(row=7,column=lipid_col+3,sticky=Tk.W,pady=(2,2), padx=(0,5))

#Initializes the GUI labels and buttons, and the layout
button1=Tk.Button(text='Select Directory',fg='black',command=Command1, width=25)
button2=Tk.Button(text='Select Drug Experiment File',fg='red',command=Drug1,width=25) 
button3=Tk.Button(text='Select Control Background File',fg='blue',command=Background1,width=25)
button99=Tk.Button(text='Run ITC Analysis',fg='black',command=testing,height=5,width=25)
button8=Tk.Button(text='Reset',fg='black',command=Reset1)
label1001=Tk.Label(root,text='Number of initial points to delete=')
label1002=Tk.Label(root,text='Concentration of the Lipid in the syringe [M]=')
label1003=Tk.Label(root,text='Concentration of the Drug in the cell [M]=')
label1004=Tk.Label(root,text='Volume of the Cell [L]=')
label1007=Tk.Label(root,text='Drug Molar Area=')
label1008=Tk.Label(root,text='Salt Concentration [moles/L]=')
label1009=Tk.Label(root,text='Z(Drug)=')
entry1001=Tk.Entry(root, width=25)
entry1001.insert(Tk.END, '1')
entry1002=Tk.Entry(root, width=25)
entry1002.insert(Tk.END, '15e-3')
entry1003=Tk.Entry(root, width=25)
entry1003.insert(Tk.END, '100e-6')
entry1004=Tk.Entry(root, width=25)
entry1004.insert(Tk.END, '200e-6')
entry1007=Tk.Entry(root, width=25)
entry1007.insert(Tk.END, '(3e-19)*(6.0221409e23)')
entry1008=Tk.Entry(root, width=25)
entry1008.insert(Tk.END, '0.15')
entry1009=Tk.Entry(root, width=25)
entry1009.insert(Tk.END, '1')

label1001.grid(row=1,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1001.grid(row=1,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1002.grid(row=2,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1002.grid(row=2,column=4,sticky=Tk.W,pady=(2,2), padx=(0,))
label1003.grid(row=3,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1003.grid(row=3,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1004.grid(row=4,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1004.grid(row=4,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1007.grid(row=8,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1007.grid(row=8,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1008.grid(row=9,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1008.grid(row=9,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1009.grid(row=10,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))
entry1009.grid(row=10,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))

button1.grid(row=0,column=0,sticky=Tk.W)
button2.grid(row=1,column=0,sticky=Tk.W)
button3.grid(row=2,column=0,sticky=Tk.W)
button99.grid(row=3,column=0,sticky=Tk.W,columnspan=2,rowspan=5)
button8.grid(row=0,column=6,sticky=Tk.E)

label10100=Tk.Label(root,text='Select lipid used: ') 
label10100.grid(row=5,column=3,sticky=Tk.W,pady=(2,2), padx=(15,0))

label_lip_area = Tk.Label(root,text='Area/Molecule of Lipid (\u212B\u00B2) =')
label_lip_thickness = Tk.Label(root,text='Hydrocarbon Thickness (\u212B) =') 
label_lip_area_e=Tk.Entry(root, width=10)
label_lip_area_e.insert(Tk.END, '68.7')
label_lip_thickness_e=Tk.Entry(root, width=10)
label_lip_thickness_e.insert(Tk.END, '17')
labelarrow=Tk.Label(root,text='-->')
labelarrow.grid(row=6,column=4,sticky=Tk.W,pady=(2,2), padx=(70,0), columnspan=2, rowspan=2)
label1005=Tk.Label(root,text=' Vesicle Molar Volume=')
label1006=Tk.Label(root,text=' Vesicle Molar Area=')
entry1005=Tk.Entry(root, width=25)
entry1005.insert(Tk.END, '(1168e-27)*(6.0221409e23)') #7.033e-1
entry1006=Tk.Entry(root, width=25)
entry1006.insert(Tk.END, '(6.521e-19)*(6.0221409e23)') #3.9269e5 
label_lip_area.grid(row=6,column=3,sticky=Tk.W,pady=(2,2), padx=(90,0))
label_lip_area_e.grid(row=6,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label_lip_thickness.grid(row=7,column=3,sticky=Tk.W,pady=(2,2), padx=(90,0))
label_lip_thickness_e.grid(row=7,column=4,sticky=Tk.W,pady=(2,2), padx=(0,0))
label1005.grid(row=6,column=4,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
entry1005.grid(row=6,column=6,sticky=Tk.W,pady=(2,2), padx=(0,5))
label1006.grid(row=7,column=4,sticky=Tk.W,pady=(2,2), padx=(95,0), columnspan=2)
entry1006.grid(row=7,column=6,sticky=Tk.W,pady=(2,2), padx=(0,5))

lipid_col = 3

n = Tk.StringVar() 
keepvalue = n.get()
lipidselect = ttk.Combobox(root, width = 22, textvariable = keepvalue, state = 'readonly', justify=Tk.LEFT) 
lipidselect['values'] = (' Other (lipid area/thickness)',
                         ' Other (vesicle volume/area)',
                         ' DC18:1PC', 
                         ' DC20:1PC', 
                         ' DC22:1PC')   
lipidselect.current(4) 
lipidselect.bind("<<ComboboxSelected>>", lipidparams)
lipidselect.grid(column = 4, row = 5, sticky= Tk.W) 

button10 = Tk.Button(text='compute',fg='black',command=lipidparams2, width=7) #10
button10.grid(row=5,column=5,sticky=Tk.W,pady=(2,2), padx=(5,0)) #15

#Run GUI
root.mainloop()