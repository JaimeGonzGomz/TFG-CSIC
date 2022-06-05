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

# The data assigned to the list
tiempoDeExec = 0
figura=0
# Variables a definir, resistencia en megaohm y área en m^2
Resis = 10
Area = 0.02 * 0.02
# read_csv function which is used to read the required CSV file, hago también lista para que lea todos los csvs del directorio
path = 'csvs'
all_files = glob.glob(path + "/*.csv")
# Outputnames
Headers = ["Nombre de archivo", 'Vmax (V)', 'Vmin (V)', 'PktPk (V)', 'Fuerza (N)', 'PotenciaMedia (W/m^2)',
           'PotenciaMax (W/m^2)']
dfexcel = pd.DataFrame(columns=Headers)

with open('Resultados2.csv', 'w') as Outp:
    Output = csv.writer(Outp)
    Output.writerow(Headers)
    with pd.ExcelWriter(r'C:\Users\Jaime\Desktop\ResultadosPrueba2.xlsx') as writer:
        for filename in all_files:
            start = time.time()
            data = pd.read_csv(filename, header=5)
            # print(data)
            # Se borran todos los que no son números, no sé como borrar el header del csv así que por ahora lo hago a mano
            df1 = data.dropna(axis=1, thresh=2)
            #print(df1)
            df1 = df1.set_axis(['Tiempo(s)', 'TENG(V)', 'Corriente(mA)', 'Sensor(V)'], axis=1, inplace=False)
            # Ordenar las columnas
            dfOrden = ['Tiempo(s)', 'TENG(V)', "Sensor(V)", "Corriente(mA)"]
            df1 = df1.reindex(dfOrden, axis='columns')

            df1 = df1.drop([0, 1, 2, 3, 4, 5], axis=0)
            # display
            # print("\nCSV Data after deleting the column :\n")
            # print(df1)
            # Calculo de cosas sin integrar

            # Floateo el data en dataframe no en listas anidadas
            df1['Tiempo(s)'] = df1['Tiempo(s)'].astype(float)
            df1['TENG(V)'] = df1['TENG(V)'].astype(float)
            df1['Sensor(V)'] = df1['Sensor(V)'].astype(float)

            Vmax = df1["TENG(V)"][df1["TENG(V)"].idxmax()]
            VmaxSensor = df1["Sensor(V)"][df1["Sensor(V)"].idxmax()]
            Fuerza = float(df1["Sensor(V)"][df1["Sensor(V)"].idxmax()] * 1000 / 22.1)
            TiempoInicial = df1["Tiempo(s)"][df1["Tiempo(s)"].idxmin()]
            df1['Tiempo(s)'] = df1['Tiempo(s)'] + abs(TiempoInicial)
            df1['Corriente(mA)'] = df1['TENG(V)'] / (Resis)

            if VmaxSensor > Vmax:
                Vmax = VmaxSensor
                df1 = df1[['Tiempo(s)', 'Sensor(V)', 'TENG(V)', 'Corriente(mA)']]
                Fuerza = float(df1["TENG(V)"][df1["TENG(V)"].idxmax()] * 1000 / 22.1)
                df1['Corriente(mA)'] = df1['TENG(V)'] / (Resis)

            # Cálculo para cada ciclo, definiendo tiempo y frecuencia
            frecuencia = 10
            Tiempototal = df1["Tiempo(s)"][df1["Tiempo(s)"].idxmax()]
            ciclosTotales = int(round(Tiempototal * frecuencia, 0))
            datosIntroducidos = len(df1.index)
            voltajesMaxCiclos = []
            voltajesMinCiclos = []
            voltajesMaxCiclosSensor = []
            for ciclo in range(ciclosTotales):
                corteDato = df1.loc[int(ciclo * datosIntroducidos / ciclosTotales):int(
                    (ciclo + 1) * datosIntroducidos / ciclosTotales)]
                localVmax = corteDato["TENG(V)"][corteDato["TENG(V)"].idxmax()]
                localVmin = corteDato["TENG(V)"][corteDato["TENG(V)"].idxmin()]
                localVmaxSensor = corteDato["Sensor(V)"][corteDato["Sensor(V)"].idxmax()]
                voltajesMaxCiclos.append(localVmax)
                voltajesMinCiclos.append(localVmin)
                voltajesMaxCiclosSensor.append(localVmaxSensor)
            # Vmax absoluto sería quitando el comentario de debajo
            print(Vmax)
            VmaxSensor = mean(voltajesMaxCiclosSensor)
            Vmax = mean(voltajesMaxCiclos)
            # Este sería el Vmax que se presenta en el resultado
            # print(Vmax)
            Vmin = mean(voltajesMinCiclos)
            PktPk = Vmax - Vmin


            # Cálculo de la potencia
            df1 = pd.concat([df1, (df1['TENG(V)'] * df1['TENG(V)'] / Resis)], ignore_index=True, axis=1)
            df1 = df1.set_axis(['Tiempo(s)', 'TENG(V)', 'Sensor(V)', 'Corriente(mA)', 'PotenciaInst(uW)'], axis=1,
                               inplace=False)
            if VmaxSensor > Vmax:
                df1 = df1[['Tiempo(s)', 'Sensor(V)', 'TENG(V)', 'Corriente(mA)', 'PotenciaInst(uW)']]

            # añado potencia al dataframe principal

            totalInt = integrate.simpson(df1['PotenciaInst(uW)'], df1['Tiempo(s)'])
            PotenciaMedia = totalInt / (1.6 * 0.02 * 0.02 * 1000000)
            PotenciaMax = Vmax * Vmax / (Resis * 1000000 * 0.02 * 0.02)

            # print("\nPotencia media:\n")
            # print(PotenciaMedia,'W/m^2')
            # print(filename)

            # Para el csv de salida
            filename = str(filename)
            filename = filename.split('\\', -1)[1]
            #filename = filename.split('_', -1)[0]
            Resultados = [filename, format(Vmax, '.2E'), format(Vmin, '.2E'), format(PktPk, '.2E'), round(Fuerza, 2),
                          format(PotenciaMedia, '.2E'), format(PotenciaMax, '.2E')]
            Output.writerow(Resultados)
            # Representación, por ahora ignorar
            exampledata = np.array(df1[10:], dtype=float)
            xdata = exampledata[:, 0]
            ydata = exampledata[:, 1]
            ydata2 = exampledata[:, 2]
            # ploteo de figuras de primera presentación
            figura=figura+1
            plt.figure(figura, dpi=140)
            plt.xlabel('t(s)')
            plt.ylabel('Voltage(V)')
            plt.plot(xdata, ydata, label='Voltaje del teng')
            plt.plot(xdata, ydata2, label='Voltaje del sensor')
            #plt.savefig(filename)
            # Para el excel de salida
            excelResultados = pd.DataFrame([Resultados], columns=Headers)
            dfexcel = pd.concat([dfexcel, excelResultados], ignore_index=True, axis=0)
            dfexcel.to_excel(writer, sheet_name='Resumen')
            df1.to_excel(writer, sheet_name=filename)
            end = time.time()
            tiempoDeExec = end - start + tiempoDeExec
print('El tiempo de ejecución fue de:', float(tiempoDeExec))
