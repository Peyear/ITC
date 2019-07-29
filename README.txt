# ITC
 Charged

Measuring the partitioning of neutral and charged compounds into large unilamellar vesicles (LUVs). Data for ITC experiments are stored in Excel files (.xls,.xslx,.csv) and script uses the heat in the first column and injection volume in the second column.

ITC_Charged.py

Before you begin make sure following packages are installed

Numpy
Pandas
Xlrd
Scipy
Matplotlib
Astropy

1. Convert and save the experimental files to Excel (.xls, xlsx)

2. Line 9. Set directory os.chdir(r'%%%') and replace %%% with the directory
Example: os.chdir(r'H:\Data\ITC') 

3. Line 10. Select the experiment file between the ‘—'
Example: dlsFile='Data7.xls' 

4. Line 11. Select the background experiment file for heat of dilution and blank effects between the ‘—'
Example: dlsFile2='Data1.xls' 
If there is no file for correction or if data has already been corrected delete lines 60-73

5. Line 13. Select the number of points to delete (often it’s the first injection)
Example: Delete=1

6. Lines 18-20. Set the concentration of the compound in the syringe in Clipid; concentration of lipid in cell in Cdrug; and volume of cell in Vw.
Example: Clipid=15e-3  	% M
Cdrug=100e-6		% M
Vw=200e-6		% L

7. Confirm values in lines 22-28

8. Run script and results for neutral charge are in the first row and charged results in the second row produced.



ITC_Charged.m

1. Convert and save the experimental files to Excel (.xls, xlsx)

2. Line 2. Set directory cd('%%%'); and replace %%% with the directory
Example: cd('H:\Data\ITC') 

3. Line 4. Select the experiment file between the ‘—'
Example: ITC=xlsread('Data7.xls') 

4. Line 5. Select the background experiment file for heat of dilution and blank effects between the ‘—'
Example: ITC2=xlsread('Data1.xls') 
If there is no file for correction or if data has already been corrected delete lines 36-44

5. Line 7. Select the number of points to delete (often it’s the first injection)
Example: Delete=1

6. Lines 9-11. Set the concentration of the compound in the syringe in Clipid; concentration of lipid in cell in Cdrug; and volume of cell in Vw.
Example: Clipid=15e-3  	% M
Cdrug=100e-6		% M
Vw=200e-6		% L

7. Confirm values in lines 17-23

8. Run script and results for neutral charge are in the first row and charged results in the second row produced.