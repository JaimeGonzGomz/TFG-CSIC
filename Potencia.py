# importe de librerías y tal
import csv
import glob
import io
import openpyxl
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import scipy.integrate as integrate
from statistics import mean
import sys
from scipy import fftpack


# The data assigned to the list
tiempoDeExec=0
#Variables a definir, resistencia en megaohm y área en m^2
ResisPlot=[]
PotenciaPlot=[]
PotenciaMaxPlot=[]
Area=0.02*0.02
# read_csv function which is used to read the required CSV file, hago también lista para que lea todos los csvs del directorio
path='Potencia'
all_files=glob.glob(path + "/*.csv")
#Outputnames
Headers=["Nombre de archivo",'Vmax (V)','Vmin (V)','PktPk (V)','Fuerza (N)','PotenciaMedia (W/m^2)','PotenciaMax (W/m^2)']
dfexcel=pd.DataFrame(columns=Headers)
#FrecuenciasRicas=[]
with open('Resultados2.csv','w') as Outp:
    Output=csv.writer(Outp)
    Output.writerow(Headers)
    #with pd.ExcelWriter(r'C:\Users\Jaime\Desktop\PruebaPotencia\ResultadosPotencia.xlsx') as writer:
    for filename in all_files:
            filename = str(filename)
            #print(filename)
            Resis=float(filename.split('_',-1)[1])
            #Resis=10
            #print(Resis)
            data=pd.read_csv(filename,header=5)
            #print(data)
            # Se borran todos los que no son números, no sé como borrar el header del csv así que por ahora lo hago a mano
            df1=data.dropna(axis=1,thresh=2)
            df1 = df1.set_axis(['Tiempo(s)', 'TENG(V)', 'Corriente(mA)', 'Sensor(V)'], axis=1,inplace=False)
            #Ordenar las columnas
            dfOrden=['Tiempo(s)','TENG(V)',"Sensor(V)","Corriente(mA)"]
            df1=df1.reindex(dfOrden,axis='columns')

            df1=df1.drop([0,1,2,3,4,5],axis=0)
            # display
            #print("\nCSV Data after deleting the column :\n")
            #print(df1)
            #Calculo de cosas sin integrar

            # Floateo el data en dataframe no en listas anidadas
            df1['Tiempo(s)'] = df1['Tiempo(s)'].astype(float)
            df1['TENG(V)'] = df1['TENG(V)'].astype(float)
            df1['Sensor(V)'] = df1['Sensor(V)'].astype(float)

            Vmax=df1["TENG(V)"][df1["TENG(V)"].idxmax()]
            VmaxSensor = df1["Sensor(V)"][df1["Sensor(V)"].idxmax()]
            Fuerza = float(df1["Sensor(V)"][df1["Sensor(V)"].idxmax()] * 1000 / 22.1)
            TiempoInicial = df1["Tiempo(s)"][df1["Tiempo(s)"].idxmin()]
            df1['Tiempo(s)']=df1['Tiempo(s)']+abs(TiempoInicial)
            df1['Corriente(mA)']=df1['TENG(V)']/(Resis)

            if VmaxSensor>Vmax:
                Vmax=VmaxSensor
                df1=df1[['Tiempo(s)','Sensor(V)','TENG(V)','Corriente(mA)']]
                Fuerza=float(df1["TENG(V)"][df1["TENG(V)"].idxmax()] * 1000 / 22.1)
                df1['Corriente(mA)'] = df1['TENG(V)'] / (Resis)

            #Cálculo para cada ciclo, definiendo tiempo y frecuencia
            frecuencia=10
            Tiempototal=df1["Tiempo(s)"][df1["Tiempo(s)"].idxmax()]
            ciclosTotales=int(round(Tiempototal*frecuencia,0))
            datosIntroducidos=len(df1.index)
            voltajesMaxCiclos=[]
            voltajesMinCiclos=[]
            voltajesMaxCiclosSensor=[]
            for ciclo in range(ciclosTotales):
                corteDato=df1.loc[int(ciclo * datosIntroducidos/ciclosTotales):int((ciclo+1)*datosIntroducidos/ciclosTotales)]
                localVmax = corteDato["TENG(V)"][corteDato["TENG(V)"].idxmax()]
                localVmin = corteDato["TENG(V)"][corteDato["TENG(V)"].idxmin()]
                localVmaxSensor=corteDato["Sensor(V)"][corteDato["Sensor(V)"].idxmax()]
                voltajesMaxCiclos.append(localVmax)
                voltajesMinCiclos.append(localVmin)
                voltajesMaxCiclosSensor.append(localVmaxSensor)
            #Vmax absoluto sería quitando el comentario de debajo
            #print(Vmax)
            VmaxSensor=mean(voltajesMaxCiclosSensor)
            Vmax=mean(voltajesMaxCiclos)
            #Este sería el Vmax que se presenta en el resultado
            #print(Vmax)
            Vmin=mean(voltajesMinCiclos)
            PktPk=Vmax-Vmin
            #Representación, por ahora ignorar


            # Cálculo de la potencia
            df1 = pd.concat([df1, (df1['TENG(V)'] * df1['TENG(V)'] / Resis)], ignore_index=True, axis=1)
            df1 = df1.set_axis(['Tiempo(s)', 'TENG(V)', 'Sensor(V)', 'Corriente(mA)', 'PotenciaInst(uW)'], axis=1,
                               inplace=False)
            if VmaxSensor > Vmax:
                df1 = df1[['Tiempo(s)', 'Sensor(V)', 'TENG(V)', 'Corriente(mA)', 'PotenciaInst(uW)']]
            # añado potencia al dataframe principal

            totalInt=integrate.simpson(df1['PotenciaInst(uW)'], df1['Tiempo(s)'])
            if abs(Vmin)>abs(Vmax):
                Vmax=Vmin
            PotenciaMedia=totalInt/(Tiempototal*Area)
            PotenciaMax=Vmax*Vmax/(Resis*Area*1000)
            Imax=Vmax/Resis

            #print("\nPotencia media:\n")
            #print(PotenciaMedia,'W/m^2')
            #print(filename)
            # intentos de calcular la frecuencia
            #f_s=12500
            #lolo = np.array(df1['TENG(V)'], dtype=float)
            #Xaxa = fftpack.fft(lolo)
            #freqs = fftpack.fftfreq(len(lolo)) * f_s
            #df2=pd.DataFrame((freqs,np.abs(Xaxa)))
            #df2=df2.T
            #df2 = df2.set_axis(['freqs','Xaxa'], axis=1, inplace=False)
            #df2 = df2.drop([0, 1, 2, 3, 4, 5], axis=0)
            #df2 = df2.drop(df2.index[100:20000])

            #FrecuenciaCalc=df2['Xaxa'].idxmax()

            #print(df2['freqs'][FrecuenciaCalc])
            #FrecuenciasRicas.append(df2['freqs'][FrecuenciaCalc])
            #plt.show()
            #Para el csv de salida
            filename = filename.split('\\', -1)[1]
            Resultados=[filename,format(Vmax,'.2E'),format(Vmin,'.2E'),format(PktPk,'.2E'),round(Fuerza,2),format(PotenciaMedia,'.2E'),format(PotenciaMax,'.2E'),format(Resis,'.2E'),format(Imax,'.2E')]
            Output.writerow(Resultados)

            #Para el excel de salida
            #excelResultados=pd.DataFrame([Resultados],columns=Headers)
            #dfexcel=pd.concat([dfexcel,excelResultados],ignore_index=True,axis=0)
            #dfexcel.to_excel(writer, sheet_name='Resumen')
            #df1.to_excel(writer,sheet_name=filename)
            #print(PotenciaMax)
            ResisPlot.append(float(Resis))
            print(Resis)
            PotenciaPlot.append(float(PotenciaMedia))
            PotenciaMaxPlot.append(float(PotenciaMax))

#print(FrecuenciasRicas)

xdata=ResisPlot
print(ResisPlot)
ydata=PotenciaPlot
plt.figure(1,dpi=140)
plt.xscale('log')
plt.xlabel('R(MOhm)')
plt.ylabel('Power (uW/m^2)')
print(sorted(xdata))
Z = [y for _,y in sorted(zip(xdata,ydata))]
plt.plot(sorted(xdata),Z,label='Potencia Media')
plt.show()
print(ResisPlot)
print(Z)
ydata2=PotenciaMaxPlot
plt.figure(2,dpi=140)

Z2 = [y for _,y in sorted(zip(xdata,ydata2))]
plt.plot(sorted(xdata),Z2,label='Potencia Max')
print(PotenciaMaxPlot)

#ploteo de figuras de primera presentación

fig, axs = plt.subplots(2)
fig.suptitle('Potencias')
plt.xscale('log')
plt.xlabel('R(MOhm)')
plt.ylabel('Power (uW/m^2)')

axs[0].plot(sorted(xdata), Z,'o')
axs[0].set_xscale('linear')
axs[0].set_xlabel('R(MOhm)')
axs[0].set_ylabel('Power (uW/m^2)')
axs[0].set_title("Potencia Media")
axs[1].plot(sorted(xdata), Z2,'+')
axs[1].set_xscale('linear')
axs[1].set_xlabel('R(MOhm)')
axs[1].set_ylabel('Power (uW/m^2)')
axs[1].set_title("Potencia Max")
plt.show()