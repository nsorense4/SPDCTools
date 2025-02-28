# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:52:42 2023

@author: nicho
"""

import numpy as np
import re
from plot_custom import plot_custom, makeTopLeftSpinesInvisible
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import os
import scipy.integrate as integrate
import scipy.special as special
import seaborn as sns
# from SPDCalc import all
import pandas as pd

import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score

from matplotlib.ticker import FormatStrFormatter

from SPDCalc import *

from scipy.signal import fftconvolve

PRQuantumWidthFull = 6.75

labelfontsize = 12
numfontsize = 10 
textwidth = 8.5 - 2 # in inches

labelfontsize = 9
numfontsize = 9
textwidth = 8.5 - 2 # in inches
lw = 1
s = 10

colorRed = '#ff3a63'
colorBlue = '#5d31ff'
colorPurple = '#67009f'
colorOrange = '#ff9b00'

colorRedLight = '#ffb2c2'
colorBlueLight = '#cec0ff'
colorPurpleLight = '#d68cff'

class timeTaggerSpecs():
    jitter = 400 # ps
    
def extract_file_name(file_path):
    file_name_with_extension = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    return file_name_without_extension

def parseTextFile(filePath):
    result_dict = {}
    correlation_data_lines = []
    count_data_lines = []
    with open(filePath, 'r') as file:
        lines = file.readlines()
        skip_lines = False
        corrData = False
        correlation_data_str = ""
        for line in lines:
            if skip_lines and corrData is True:
                correlation_data_lines.append(line.strip())
                if ')' in line or line == 'None\n':
                    skip_lines = False
            elif skip_lines and corrData is False:
                count_data_lines.append(line.strip())
                if ')' in line or line == 'None\n':
                    skip_lines = False
            elif line.startswith('Correlation data:'):
                skip_lines = True
                corrData = True
            elif line.startswith('Counter data:'):
                skip_lines = True
                corrData = False
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                value, unit = extractValueAndUnit(value)
                if unit:
                    result_dict[key + ' (' + unit + ')'] = value
                else:
                    if key == "g(2)(0)":
                        result_dict[key] = float(value)
                    else:
                        result_dict[key] = value
        correlation_data_str = ''.join(correlation_data_lines)
        count_data_str = ''.join(count_data_lines)
    result_dict['Correlation data'] = (eval('np.' + correlation_data_str.replace(" ", "")))
    try:
        countArray1 = (eval('np.' + count_data_str.replace(" ", "").replace("...,", "")))[0]
        result_dict['Channel 1 count rate:'] = np.mean(countArray1[countArray1 != 0])/result_dict['Count interval (s)']
        countArray2 = (eval('np.' + count_data_str.replace(" ", "").replace("...,", "")))[1]
        result_dict['Channel 2 count rate:'] = np.mean(countArray2[countArray2 != 0])/result_dict['Count interval (s)']
        countArray1Bool = np.asarray([element != 0 for element in countArray1])
        countArray2Bool = np.asarray([element != 0 for element in countArray2])
        exposureTimeArray = countArray1Bool * countArray2Bool
        exposureTime = np.sum(np.array([1 if value else 0 for value in exposureTimeArray]))*result_dict['Count interval (s)']
        if abs(1 - exposureTime/result_dict["Correlation time (s)"]) > 0.01:
            result_dict["Correlation time (s)"] = exposureTime
            print("Calculating correlation time; detector was not stable.")
    except:
        try:
            result_dict['Channel 1 count rate:'] = np.mean((eval('np.' + count_data_str.replace(" ", "").replace("...,", "")))[0])/0.1
            result_dict['Channel 2 count rate:'] = np.mean((eval('np.' + count_data_str.replace(" ", "").replace("...,", "")))[1])/0.1
        except:
            result_dict['Channel 1 count rate:'] = 0
            result_dict['Channel 2 count rate:'] = 0
    N = int(result_dict['Correlation bins'])
    result_dict['Time data (ps)'] = (np.arange(-np.ceil(N/2)*result_dict['Correlation bin width (ps)'], 
                                         np.floor(N/2)*result_dict['Correlation bin width (ps)'], 
                                         result_dict['Correlation bin width (ps)']) + result_dict['Input delay A (ps)'] 
                                     - result_dict['Input delay B (ps)'] + 500)
    return result_dict

def parseTabularTextFile(filePath):
    result_dict = {}
    with open(filePath, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split('\t')
        data_lines = lines[1:]

        # Extract data columns
        data = [line.strip().split('\t') for line in data_lines]
        data = np.array(data, dtype=float).T

        # Rename fields
        result_dict['Time data (ps)'] = data[0]
        result_dict['Correlation data'] = data[1][::-1]
        result_dict["Correlation bin width (ps)"] = data[0][1] - data[0][0]

    return result_dict

def parseTabularCRTextFile(filePath):
    result_dict = {}
    
    with open(filePath, 'r') as file:
        lines = file.readlines()
        
        # Get headers (assuming the first row contains headers)
        headers = lines[0].split('\t')
        headers = [header.strip() for header in headers]  # Remove newline characters
        
        # Initialize data lists for each header
        data = {header: [] for header in headers}
        
        # Parse data rows
        for line in lines[1:]:
            values = line.split('\t')
            for header, value in zip(headers, values):
                data[header].append(value.strip())
    
    # Convert data lists to numpy arrays
    for header, values in data.items():
        result_dict[header] = np.array(values, dtype=float)
    
    return result_dict

def extractValueAndUnit(value):
    match = re.search(r'([+-]?\d*\.?\d+)\s?(\w.*)?', value)
    if match:
        value = match.group(1)
        unit = match.group(2)
        return float(value), unit
    else:
        return None, None

def parseFile(filePath):
    if filePath.endswith('.txt'):
        if "webCR" in filePath:
            return parseTabularCRTextFile(filePath)
        elif "web" in filePath:
            return parseTabularTextFile(filePath)
        return parseTextFile(filePath)
    else:
        raise ValueError("Unsupported file format")
        
def combineDataDicts(dataDict1, dataDict2):
    if (dataDict1['Time data (ps)'] == dataDict2['Time data (ps)']).all():
        dataDict = dataDict1.copy()
        dataDict['Correlation data'] += dataDict2['Correlation data'].astype('int32')
    
        return dataDict
    
    else:
        raise ValueError("Time data does not match.")
        
def writeData2File(dataDict, filePath):
    np.set_printoptions(threshold=len(dataDict['Correlation data']))
    if filePath and not(os.path.isfile(filePath)):
        with open(filePath, 'w') as f:
            f.write('Input channel A: %d\n' % dataDict['Input channel A'])
            f.write('Input channel B: %d\n' % dataDict['Input channel B'])
            f.write('Input delay A: %d ps\n' % dataDict['Input delay A (ps)'])
            f.write('Input delay B: %d ps\n' % dataDict['Input delay B (ps)'])
            f.write('Trigger level A: %.3f V\n' % dataDict['Trigger level A (V)'])
            f.write('Trigger level B: %.3f V\n' % dataDict['Trigger level B (V)'])
            f.write('Test signal A: %d\n' % dataDict['Test signal A'])
            f.write('Test signal B: %d\n' % dataDict['Test signal B'])

            f.write('Coincidence window: %d ps\n' % dataDict['Coincidence window (ps)'])
            f.write('Correlation bin width: %d ps\n' % dataDict['Correlation bin width (ps)'])
            f.write('Correlation bins: %d\n' % dataDict['Correlation bins'])
            
            f.write("Correlation time: %0.2f s\n" % dataDict['Correlation time (s)'])
            
            f.write("Average total count rate: %0.2f kc/s\n" % dataDict['Average total count rate (kc/s)'])
            f.write("Average correlation rate: %0.2f Hz\n" % dataDict['Average correlation rate (Hz)'])
            f.write("g(2)(0): %0.4f \n\n" % dataDict['g(2)(0)'])

            f.write('Counter data:\n%s\n\n' % dataDict['Counter data'].__repr__())
            f.write('Correlation data:\n%s\n\n' % dataDict['Correlation data'].__repr__())
    else:
        print("File either already exists and cannot be written over or a filepath was not given.")
        
def combineWebAndPyData(fileNameWeb, fileNamePy, dataDir, fileNameNew):
    dataDictWeb = parseFile(dataDir + fileNameWeb)
    dataDictPy = parseFile(dataDir + fileNamePy)
    
    newDict = combineDataDicts(dataDictPy, dataDictWeb)
    
    writeData2File(newDict, dataDir + fileNameNew)
    
    
    
def fitOffsetGaussian(time, data, timeCont):
    
    def offsetGaussian(x, A, B, mu, off):
        y = A*np.exp(-(x-mu)**2/B**2)+off
        return y
    
    def g2Gauss(x, A, B, mu, off):
        return A/off*np.exp(-(x-mu)**2/B**2)+1
    
    def corr2g2(data, off):
        return data/off
    
    try:
        parameters, covariance = curve_fit(offsetGaussian, time, data, p0 = [10*data[0], 500, 0, data[0]])
        (peak, hOffset, stddev, vOffset) = (parameters[0], parameters[1], parameters[2], parameters[3])
        perr = np.sqrt(np.diag(covariance))
        (delPeak, delvOff) = (perr[0], perr[3])
        gauss = g2Gauss(timeCont, parameters[0], hOffset, stddev, parameters[3])
    except:
        gauss = timeCont*0
        peak = 1
        hOffset = 0
        vOffset = 0
        stddev = 0
        
        
    # fig, ax = plot_custom(
    #     textwidth, textwidth/2.4, r"time delay, $\tau$ (ns)", r"coincidence rate (Hz/mW)", 
    #     equal=False, labelPadX = 5, labelPadY = 10, axefont = labelfontsize, numsize = numfontsize, 
    #     legfont = labelfontsize
    # )
    # ax.plot(time, offsetGaussian(time, peak, hOffset, stddev, vOffset))
    # ax.plot(time, data)
    return gauss, peak, hOffset, stddev, vOffset, delPeak, delvOff

def calcNormData(dictData):
    
    corrData = dictData["Correlation data"]
    timeData = dictData["Time data (ps)"]
    
    gaussCont, peak, hOffset, stddev, vOffset, delPeak, delvOff = fitOffsetGaussian(timeData, corrData, timeCont=timeData)
    
    corrDataNorm = corrData/vOffset
    return timeData, corrData, corrDataNorm
    
def removeBackgroundData(dictData):
    
    corrData = dictData["Correlation data"]
    timeData = dictData["Time data (ps)"]
    
    gaussCont, peak, hOffset, stddev, vOffset, delPeak, delvOff = fitOffsetGaussian(timeData, corrData, timeCont=timeData)
    
    corrDataOff = (corrData - vOffset)/np.max(corrData - vOffset)
    corrDataOff = (corrData/vOffset)
    return timeData, corrData, corrDataOff, vOffset

def calcRealCoincidenceRate(filePath, log = False, fitg2 = True, saveBool = False, CRFilePath = "", CRFileBool = False, title = "", 
                    fileName = "", exposureTime = 1, PpumpmW = 10, timeOffset = 0):

    if CRFileBool is True:
        dictCRData = parseFile(CRFilePath)
    dictData = parseFile(filePath)
    timeData, corrData, corrDataOff, vOffset = removeBackgroundData(dictData)
    
    try:
        rate1 = dictData["Channel 1 count rate:"]
        rate2 = dictData["Channel 2 count rate:"]
        timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
        exposureTime = dictData["Correlation time (s)"]
        RAcc = rate1*rate2*exposureTime*timeBinWidth

    except:
        RAcc = vOffset
        # rate1 = 200e3
        # rate2 = 165e3
        # timeBinWidth = 100e-12
        # exposureTime = RAcc/(rate1*rate2*timeBinWidth)
        # print('exposureTime: ', exposureTime)
        
    RAcc = vOffset
    
    corrDataOffset = dictData["Correlation data"] - RAcc
    timeData = dictData["Time data (ps)"]
    dataDel = np.sqrt(corrData)

    # integrate peak
    peakArr = corrDataOffset/exposureTime
    sorted_index_array = np.argsort(peakArr)

    
    sorted_array = peakArr[sorted_index_array]
    n = 5
    area = sum(sorted_array[-n:])
    print('countrate: ', area/PpumpmW, '  power: ', PpumpmW)
    return area # in counts/s
    
        
def plotCorrelation(filePath, log = False, fitg2 = True, saveBool = False, CRFilePath = "", CRFileBool = False, title = "", 
                    fileName = "", exposureTime = 1, PpumpmW = 10, timeOffset = 0):
    colorArray = sns.color_palette("hot", n_colors=4)
    if CRFileBool is True:
        dictCRData = parseFile(CRFilePath)
    dictData = parseFile(filePath)
    timeData, corrData, corrDataOff, vOffset = removeBackgroundData(dictData)
    
    corrData = dictData["Correlation data"]
    timeData = dictData["Time data (ps)"]
    dataDel = np.sqrt(corrData)

    # calculate g2 scaling
    if CRFileBool is True:
        rate1 = np.mean(dictCRData["ch1CR(1/s)"])
        rate2 = np.mean(dictCRData["ch2CR(1/s)"])
        timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
        
        g2Scale = 1/(rate1*rate2*timeBinWidth*exposureTime)
        
    else:
        try:
            rate1 = dictData["Channel 1 count rate:"]
            rate2 = dictData["Channel 2 count rate:"]
            timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            exposureTime = dictData["Correlation time (s)"]
                
            g2Scale = 1/(rate1*rate2*timeBinWidth*exposureTime)
    
        except:
            g2Scale = 1/vOffset
    
    fig, ax = plot_custom(
        textwidth, textwidth/2.4, r"time delay, $\tau$ (ns)", r"coincidence rate (Hz/mW)", 
        equal=False, labelPadX = 5, labelPadY = 10, axefont = labelfontsize, numsize = numfontsize, 
        legfont = labelfontsize
    )
               
    ax.plot((timeData - timeOffset)/1000, corrData/exposureTime/PpumpmW, color = colorArray[0], zorder = 10)
    ax.fill_between((timeData - timeOffset)/1000, (corrData - dataDel)/exposureTime/PpumpmW, (corrData+dataDel)/exposureTime/PpumpmW, color = colorArray[0], alpha = 0.3, zorder = 1)

    # ax.set_title(title, fontsize = 30, pad = 20)
    
    # integrate peak
    peakArr = corrData/exposureTime/PpumpmW
    sorted_index_array = np.argsort(peakArr)
    sorted_array = peakArr[sorted_index_array]
    n = 4
    area = sum(sorted_array[-n:])
    print(area)
    
    if log == True:
        plt.yscale('log')
    ax.set_xlim([-10, 10])
    ylim = np.array([np.min(corrData)*0.8, np.max(corrData)*1.4])
    ax.set_ylim(ylim/exposureTime/PpumpmW)
    
    axg = ax.twinx()
    plt.yticks(fontsize = numfontsize)
    
    if log == True:
        plt.yscale('log')
        
    axg.set_ylim(ylim*g2Scale)
    
    axg.set_ylabel(r"second-order coherence, $g^{(2)}(\tau)$", fontsize = labelfontsize)
        
    plt.tight_layout()
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_correlation" + fileName + ".png", dpi=200, bbox_inches="tight",transparent=True
        )
        
def plotCorrelations(filePaths, logBool = False, saveBool = False, 
                     normBool = True, legBool = False,
                     title = "", figName = "defaultFigureName", exposureTimeArr = [], 
                     laserPowerArr = [], timeOffsetArr = [], legendLabels = []):
    
    colorArray = sns.color_palette("hot", n_colors=len(filePaths)+1)
    corrDataOff = [None]*(len(filePaths))
    timeData = corrDataOff.copy()
    corrData = corrDataOff.copy()
    figHandles = [None] * len(filePaths)
    
    # instantiate figure
    fig, ax = plot_custom(
        textwidth, textwidth/2.4, r"time delay, $\tau$ (ns)", r"coincidence rate (Hz/mW)", 
        equal=False, labelPadX = 5, labelPadY = 10, axefont = labelfontsize, numsize = numfontsize, 
        legfont = labelfontsize
    )
    
    i = 0

    for file in filePaths:
        dictData = parseFile(dataDir + file)
        timeData[i], corrData[i], corrDataOff[i], vOffset = removeBackgroundData(dictData)
        
        if normBool is True:
            try:
                exposureTime = dictData["Correlation time (s)"]
            except:
                exposureTime = exposureTimeArr[i]
            try:
                laserPower = dictData["Laser power (mW)"]
            except:
                laserPower = laserPowerArr[i]
            data = corrData[i] 
            dataDel = (np.sqrt(data)/exposureTime/laserPower) # assuming poissonian stats
            data = (data/exposureTime/laserPower)
        else:
            data = corrData[i]
            dataDel = np.sqrt(corrData[i]) # assuming poissonian stats
          
        # calculate x axis
        
        timeScaled = np.array(timeData[i])/1000
        # print("Integrating +/- 200 ps around center:")
        # print(sum(sorted(np.array(data), reverse=True)[:4])-4*min(data))
        lw = 2
        # plot data
        figHandles[i] = ax.plot((timeScaled-timeOffsetArr[i]), np.array(data), color = colorArray[i], linewidth = 2, zorder = i)[0]
        plt.fill_between((timeScaled-timeOffsetArr[i]), data-dataDel, data+dataDel, color = colorArray[i], alpha = 0.3, zorder = i)
        
        i += 1
    
    # set the figure parameters
    
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    ax.set_title(title, fontsize = labelfontsize,pad=20)
    ax.set_ylim([2e-6, 10])
    ax.set_xlim([-5, 5])
    
    if logBool == True:
        plt.yscale('log')
    
    if legBool == True:
        leg = ax.legend(figHandles, legendLabels, fontsize = labelfontsize)
        
        leg.get_frame().set_linewidth(0.0)
    
    plt.tight_layout()
    ax.grid()
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + figName + ".png", dpi=400, bbox_inches="tight",transparent=True
        )
        
def plotNormalizedCorrelation(filePath, log = False, saveBool = False):
    colorArray = sns.color_palette("hot", n_colors=4)
    dictData = parseFile(filePath)
    timeData, corrData, corrDataNorm = calcNormData(dictData)

    N = int(dictData['Correlation bins'])
    timeCont = np.linspace(-np.ceil(N/2)*dictData['Correlation bin width (ps)'], np.ceil(N/2)*dictData['Correlation bin width (ps)'], 1000)
    
    gaussCont, peak, hOffset, stddev, vOffset, delPeak, delvOff = fitOffsetGaussian(timeData, corrData, timeCont)
    
    fig, ax = plot_custom(
        9, 6, r"time delay, $\tau$ (ns)", r"norm. coincidences", equal=False, labelPad = 25
    )
    
    ax.plot(timeData/1000, corrData/vOffset)
    ax.plot(timeCont/1000, gaussCont)
    
    if log == True:
        plt.yscale('log')
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    # ax.set_title(r"$g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}", fontsize = 25,pad=20)
    plt.tight_layout()
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_normCorrelation.png", dpi=300, bbox_inches="tight",transparent=True
        )
        
def calcg2FromNorm(corrDataNorm, timeData):
    M = len(corrDataNorm)
    g2 = np.zeros([M])
    
    for i in range(M):
        intWidth = np.sum((np.abs(timeData - timeData[i]) <= timeTaggerSpecs.jitter/2)*1)/(timeTaggerSpecs.jitter/(timeData[1]-timeData[0]))
        g2[i] = (integrate.simpson(corrDataNorm*(np.abs(timeData - timeData[i]) <= timeTaggerSpecs.jitter/2), 
                                          timeData)/(intWidth*timeTaggerSpecs.jitter))
    return g2

def calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime):
    return corrData/(rate1*rate2*timeBinWidth)/exposureTime

def calcg2Err(corrData, timeBinWidth, rate1, rate2, exposureTime):
    return np.sqrt(corrData)/(rate1*rate2*timeBinWidth)/exposureTime

colorArray = sns.color_palette("hot", n_colors=4)
def plotg2(filePath, logBool = False, saveBool = False, figName = "", fitg2 = False):
    dictData = parseFile(filePath)
    timeData, corrData, corrDataNorm = calcNormData(dictData)

    if fitg2 == True:
        rate1 = dictData["Channel 1 count rate:"]
        rate2 = dictData["Channel 2 count rate:"]
        timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
        exposureTime = dictData["Correlation time (s)"]
        
        g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
    else:
        g2 = calcg2FromNorm(corrDataNorm, timeData)
    
    fig, ax = plot_custom(
        9, 6, r"time delay, $\tau$ (ns)", r"$g^{(2)}(\tau)$", equal=False, labelPad = 25
    )
    
    ax.plot(timeData/1000, g2)
    # ax.fill_between(timeData/1000, g2 - g2del, data+dataDelg2 + g2del, color = colorArray[i%2*2], alpha = 0.3)
    ax.set_title(extract_file_name(filePath), fontsize = 25, pad = 15)
    
    if logBool == True:
        plt.yscale('log')
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    # ax.set_title(r"$g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}", fontsize = 25,pad=20)
    plt.tight_layout()
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_g2" + figName + ".png", dpi=200, bbox_inches="tight",transparent=True
        )
        
    
        
        
def plotg2Power(dataDir, filePaths, powers, powersDel, saveBool = False, figName = ""):
    
    g20 = np.zeros(len(filePaths))
    
    i = 0
    for filePath in filePaths:
        dictData = parseFile(dataDir + filePath)
        timeData, corrData, corrDataNorm = calcNormData(dictData)
        
        try:
            rate1 = dictData["Channel 1 count rate:"]
            rate2 = dictData["Channel 2 count rate:"]
            timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            exposureTime = dictData["Correlation time (s)"]
            
            g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
            
            g20[i] = np.max(g2)
        except:
            g20[i] = np.max(calcg2FromNorm(corrDataNorm, timeData))
            

        
        i += 1
        
    def fitInverse(x, A):
        return A/x + 1

    param, cov = curve_fit(fitInverse, powers, g20)
    ppowerCont = np.linspace(4, 1.1* np.max(powers), 500)
    
        
    fig, ax = plot_custom(
        9, 6, r"pump power, $P_p$ (mW)", r"$g^{(2)}(0)$", equal=False, labelPad = 25
    )
    
    ax.errorbar(powers, g20, xerr = powersDel, yerr = np.sqrt(g20), marker = 'o', linestyle = 'None')
    pfit1, = ax.plot(ppowerCont, fitInverse(ppowerCont, param[0]))
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_g2Power" + figName + ".png", dpi=300, bbox_inches="tight"
        )
        
def plotg2Powers(dataDir, filePathArr, powersArr, powersDel, saveBool = False, fitg2 = False,
                 figName = "", legendBool = False, legendLabels = ["", "", ""], fileName = ""):
    
    def fitInverse(x, A):
        return A/x + 1
    
    colorScalar = 1.1
    colorArray = sns.color_palette("hot", n_colors=len(filePathArr)+1)
    colorArrayDark = tuple(tuple(i / colorScalar for i in color) for color in colorArray)
    
    
    fig, ax = plot_custom(
        7.5, 7, r"pump power, $P_p$ (mW)", r"$g^{(2)}(0)$", equal=False, labelPad = 25
    )
    j = 0
    
    for filePaths in filePathArr:
        g20 = np.zeros(len(filePaths))
        
        i = 0
        for filePath in filePaths:
            dictData = parseFile(dataDir + filePath)
            timeData, corrData, corrDataNorm = calcNormData(dictData)
        
            if fitg2 == True:
                try:
                    rate1 = dictData["Channel 1 count rate:"]
                    rate2 = dictData["Channel 2 count rate:"]
                    timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
                    exposureTime = dictData["Correlation time (s)"]
                    
                    g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
                    
                    g20[i] = np.max(g2)
                except:
                    g20[i] = np.max(calcg2FromNorm(corrDataNorm, timeData))
            else:
                g20[i] = np.max(calcg2FromNorm(corrDataNorm, timeData))
            
            i += 1

        # param, cov = curve_fit(fitInverse, powersArr[j], g20)
        # ppowerCont = np.linspace(4, 1.1* np.max(powersArr[j]), 500)

    
        ax.errorbar(powersArr[j], g20, xerr = powersDel[j], yerr = np.sqrt(g20), 
                    marker = 'o', linestyle = 'None', color = colorArray[j], zorder = 10)

        # pfit1, = ax.plot(ppowerCont, fitInverse(ppowerCont, param[0]), color = colorArrayDark[j], zorder = 1)

        ax.grid()
        j += 1
    
    if legendBool is True:
        ax.legend(legendLabels, fontsize = 25)
        
    # ax.set_ylim([0, 160])
    # ax.set_xlim([2,75])
    # ax.set_yscale("log")
    # ax.set_xscale("log")
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_g2Power" + fileName + ".png", dpi=300, bbox_inches="tight"
        )
        

    
    return ax, fig

fFitcut2wl = lambda x, a, b, c: a + b*x + c*x**2
cutoff1500 = 1497
def defcut2wl():
    cutoffs = np.array([-8.9, -2.4, 2.1, 9.5])
    
    # wavelengths = np.array([1500, 1550, 1600, (1/788-1/1500)**-1])
    wavelengths = np.array([cutoff1500, 1550, 1600, (1/788-1/cutoff1500)**-1])
    par, cov = curve_fit(fFitcut2wl, cutoffs, wavelengths)
    return par, cutoffs, wavelengths
par1, _, _ = defcut2wl()
fcut2wl = lambda delay: fFitcut2wl(delay, par1[0], par1[1], par1[2])

def defcut2wlHalf():
    cutoffs = np.array([-4.9, -1.8 ,  0.3,  4.55])
    wavelengths = np.array([cutoff1500, 1550, 1600, (1/788-1/cutoff1500)**-1])
    par, cov = curve_fit(fFitcut2wl, cutoffs, wavelengths)
    return par, cutoffs, wavelengths
par2, _, _ = defcut2wlHalf()

fcut2wlHalf = lambda delay: fFitcut2wl(delay, par2[0], par2[1], par2[2])

def countsToSpectra(time, counts, offset):
    time = time - offset
    timeAbs = np.abs(time)
    idx = np.argsort(timeAbs)
    # counts = np.flip(counts)
    time = np.concatenate((-timeAbs[np.flip(idx)], timeAbs[idx]))
    counts = np.concatenate((counts[np.flip(idx)], counts[idx]))
    # plt.plot(time, counts)
    return time, counts

def plotCut2Wl(saveBool = False):
    
    
    colorArray = [colorBlue, colorRed, colorPurple]

    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.45, 0.3*PRQuantumWidthFull, r"time delay, $\tau$ (ns)", r"wavelength, $\lambda$ (nm)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_x_value = -10
    max_x_value = 10
    min_y_value = 1475
    max_y_value = 1680
    
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.08
    limScaley = 0.1
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    _, cutoffs, wavelengths = defcut2wl()
    
    
    
    ax.scatter(cutoffs, wavelengths, color = colorArray[0], s = 2)
    delays = np.linspace(min_x_value, max_x_value, 100)
    ax.plot(delays, fcut2wl(delays), color = colorArray[0], zorder = 0,alpha = 0.5, lw=lw)
    

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    
    
    ax.set_xticks(filtered_xticks)
    ax.set_yticks([1500, 1550, 1600, 1650])
    # ax.set_xticks([1500, 1550, 1600, 1650])
    # ax.set_xticklabels([r'$2\lambda_{\text{pump}}$'])
    # ax.set_yticks([0,1])
    
    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value])
    
    ax.tick_params(axis='x', pad=3)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick label
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_fiberDispersionRegression.pdf", dpi=200, bbox_inches="tight"
        )

def envelopeFunctionDetectionEff():
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/"
    
    file = "LN-Fe/" + "fiberSpectroscopy/20230801_2210_LN-Fe_70mW_vPolPump_788nm_transmission_fiber_1400nm.txt"
    file = 'GaAs/' + "photolumDecayBurningFiberSpec/20230802_1003_GaAs111_10mW_vPolPump_788nm_transmission_fiber_1400nm_expdecay.txt"
    
    dictData = parseFile(dataDir + file)
    timeData, corrData, corrDataOff, vOffset = removeBackgroundData(dictData)

    corrDataErr = np.sqrt(corrData)/max(corrData)
    corrData = corrData/max(corrData)
    x_data = fcut2wl(timeData/1000)
    
    # Lorentzian function definition
    def lorentzian(x, A, x0, gamma):
        return (A / np.pi) * (gamma / 2) / ((x - x0)**2 + (gamma / 2)**2)
    
    # Gaussian function definition
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    def top_hat(x, A, x1, x2, sigma1, sigma2):
        return (A / 2) * (np.arctan((x - x1) / sigma1) - np.arctan((x - x2) / sigma2))

    
    # Initial guesses for the fitting parameters
    initial_guess_gaussian = [1, 1575, 1]    # [A, x0, sigma]

    # Perform the curve fit for Gaussian
    popt_gaussian, pcov_gaussian = curve_fit(gaussian, x_data, corrData, p0=initial_guess_gaussian)

    # Extract fitting parameters
    A_gauss, x0_gauss, sigma_gauss = popt_gaussian
    
    filterData = (x_data > 1200) & (x_data < 2000)
    x_data = x_data[filterData]
    corrData = corrData[filterData]
    corrDataErr = corrDataErr[filterData]
    
    return popt_gaussian


def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

popt_gaussian = envelopeFunctionDetectionEff()
envelopeFunc = lambda lamb: gaussian(lamb, popt_gaussian[0], popt_gaussian[1], popt_gaussian[2])
        

def plotEnvelopeFunction(saveBool = False):
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/"
    
    file = "LN-Fe/" + "fiberSpectroscopy/20230801_2210_LN-Fe_70mW_vPolPump_788nm_transmission_fiber_1400nm.txt"
    # file = 'GaAs/' + "photolumDecayBurningFiberSpec/20230802_1003_GaAs111_10mW_vPolPump_788nm_transmission_fiber_1400nm_expdecay.txt"
    
    dictData = parseFile(dataDir + file)
    timeData, corrData, corrDataOff, vOffset = removeBackgroundData(dictData)

    corrDataErr = np.sqrt(corrData)/max(corrData)
    corrData = corrData/max(corrData)
    x_data = fcut2wl(timeData/1000)
    
    # Lorentzian function definition
    def lorentzian(x, A, x0, gamma):
        return (A / np.pi) * (gamma / 2) / ((x - x0)**2 + (gamma / 2)**2)
    
    # Gaussian function definition
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    def top_hat(x, A, x1, x2, sigma1, sigma2):
        return (A / 2) * (np.arctan((x - x1) / sigma1) - np.arctan((x - x2) / sigma2))

    
    # Initial guesses for the fitting parameters
    # initial_guess_lorentzian = [1, 1575, 2]  # [A, x0, gamma]
    initial_guess_gaussian = [1, 1575, 1]    # [A, x0, sigma]
    # initial_guess_tophat = [5, 1500, 1750, 0.5, 0.5]
        
    # Perform the curve fit for Lorentzian
    # popt_lorentzian, pcov_lorentzian = curve_fit(lorentzian, x_data, corrData, p0=initial_guess_lorentzian)
    
    # Perform the curve fit for Gaussian
    popt_gaussian, pcov_gaussian = curve_fit(gaussian, x_data, corrData, p0=initial_guess_gaussian)

    # perform the curve fit for a tophat
    # popt_tophat, pcov_tophat = curve_fit(top_hat, x_data, corrData, p0=initial_guess_tophat)

    # Extract fitting parameters
    # A_lorentz, x0_lorentz, gamma_lorentz = popt_lorentzian
    A_gauss, x0_gauss, sigma_gauss = popt_gaussian
    
    filterData = (x_data > 1200) & (x_data < 2000)
    x_data = x_data[filterData]
    corrData = corrData[filterData]
    corrDataErr = corrDataErr[filterData]
    
    # y_fit_lorentzian = lorentzian(x_data, *popt_lorentzian)
    y_fit_gaussian = gaussian(x_data, *popt_gaussian)
    # y_fit_tophat = top_hat(x_data, *popt_tophat)

    # Plot the results
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.45, 0.33*PRQuantumWidthFull, r"wavelength, $\lambda$ (nm)",r"normalized countrate (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 6, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize,
    )
    
    
    ax.plot(x_data, corrData, color = colorBlue,lw = lw/2)
    ax.fill_between(x_data, np.array(corrData-corrDataErr), 
                                        np.array(corrData+corrDataErr), color = colorBlue, alpha = 0.3)
            
    ax.plot(x_data, y_fit_gaussian, color='k', linestyle='-', lw=lw*1.5)
    makeTopLeftSpinesInvisible(ax)
    
    ax.set_xticks([1200,1400,1600,1800,2000])
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        plt.savefig(
            "figures/" + preamble + "_fiberDispersionLNFe.svg", dpi=400, bbox_inches="tight"
        )
        
def plotFiberCoincidences(dataDir, filePaths, logBool = False, saveBool = False, legendBool = False,
                          avgBool = False, avgSample = 5, normBool = False, offsetBool = True, figName = "",
                          legendLabels = [], cutoffBool = False, cutoffs = [], plotDelayBool = True,
                          ):
    
    colorArray = sns.color_palette("hot", n_colors=len(filePaths)+2)
    
    corrDataOff = [None]*(len(filePaths))
    timeData = corrDataOff.copy()
    corrData = corrDataOff.copy()
    
    if plotDelayBool is True:
        fig, ax = plot_custom(
            textwidth/1, textwidth/2, r"time delay, $\tau$ (ns)", "coincidence rate \n (mHz/mW)",  
            numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
            labelPadX = 6, labelPadY = 16, nrow = 1, ncol = 1
        )
    else:
        fig, ax = plot_custom(
            textwidth, textwidth/3, r"wavelength, $\lambda$ (nm)", "coincidence rate \n (mHz/mW)",  
            numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
            labelPadX = 6, labelPadY = 16, nrow = 1, ncol = 1
        )
    
    figHandles = [None] * len(filePaths)
    
    i = 0
    for file in filePaths:
        dictData = parseFile(dataDir + file)
        timeData[i], corrData[i], corrDataOff[i], vOffset = removeBackgroundData(dictData)
        
        if normBool is False:
            rate1 = dictData["Channel 1 count rate:"]
            rate2 = dictData["Channel 2 count rate:"]
            timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            exposureTime = dictData["Correlation time (s)"]
            laserPower = dictData["Laser power (mW)"]
            data = corrData[i] 
            dataDel = (np.sqrt(data)/exposureTime/laserPower + 1e-10)*1e3 # assuming poissonian stats
            offset = (rate1*rate2*timeBinWidth)/laserPower
            if offset == 0:
                offset = np.min(data)
            data = (data/exposureTime/laserPower + 1e-10)*1e3 - offset * 1e3
        else:
            data = corrDataOff[i]
            dataDel = np.sqrt(corrData[i])/vOffset # assuming poissonian stats
        
        if avgBool == True:
            corr = data.copy()
            corrDel = data.copy()
            k = avgSample
            for j in range(len(data)-avgSample):
                corr[j] = np.mean(data[j:j+avgSample])
                corrDel[j] = np.mean(dataDel[j:j+avgSample])
            timeScaled = np.array(timeData[i][:-avgSample])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                x = fcut2wl(timeScaled)
                
            figHandles[i] = ax.plot(x, np.array(corr)[:-avgSample], color = colorArray[i])[0]
            plt.fill_between(x, np.array(corr-corrDel)[:-avgSample], np.array(corr+corrDel)[:-avgSample], color = colorArray[i], alpha = 0.3)
            
                
        else:
            timeScaled = np.array(timeData[i])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                x = fcut2wl(timeScaled)
            ax.plot(x, np.array(data), color = colorArray[i])
            plt.fill_between(x, (data-dataDel), (data+dataDel), color = colorArray[i], alpha = 0.3)
        i += 1
    
    
    if logBool == True:
        plt.yscale('log')
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    # ax.set_title(figName, fontsize = labelfontsize,pad=20)
    
    if legendBool == True:
        ax.legend(figHandles, legendLabels, fontsize = labelfontsize-2, loc = "upper right")
        
    if cutoffBool == True:
        for cutoff in cutoffs:
            ax.plot(cutoff * np.array([1,1]), np.array([0, 500]), 'k-')
    
    plt.tight_layout()
    
    # ax.set_ylim([0, 15])
    # ax.set_xlim([1300, (1/788 - 1/1300)**-1])
    
    ax.grid()
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        if normBool is True:
            plt.savefig(
                "figures/" + preamble + "_fiberDispersionNorm.png", dpi=200, bbox_inches="tight"
            )
        else:
            plt.savefig(
                "figures/" + preamble + "_fiberDispersion.png", dpi=200, bbox_inches="tight"
            )
            
def plotFiberCoincidencesSeparate(dataDir, filePaths, titles = [r"transmission", r"reflection", r"tranmission/reflection"], 
                                  logBool = False, saveBool = False, avgBool = False, avgSample = 5, 
                                  normBool = False, delayDoubleBool = [], figName = "_fiberDispersionNorm", plotDelayBool = True,
                                  exposureTimeArr = [], laserPowerArr = [], rate1Arr = [], rate2Arr = [],
                                  timeBinWidthDefault = 100e-12, ):
    colorArray = [colorBlue, colorRed, colorPurple]
    
    fig, ax = plot_custom(
        1*PRQuantumWidthFull,  0.35*PRQuantumWidthFull, [r"time delay, $\tau$ (ns)", r"time delay, $\tau$ (ns)", r"time delay, $\tau$ (ns)"],
        [r'coincidence rate (cps/mW)', '', ''], labelPadX = 4, labelPadY = 4, 
        commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 3, nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize+1,
    
    )
    
    
    corrDataOff = [None]*(len(filePaths))
    plotHandle = [None]*(len(filePaths))
    timeData = corrDataOff.copy()
    corrData = corrDataOff.copy()
    
    
    min_x_value = -20
    max_x_value = 20
    rangex = max_x_value - min_x_value
       
    legendHandles = [r"1400 nm", r"1500 nm"]
    
    i = 0
    for file in filePaths:
        
        dictData = parseFile(dataDir + file)
        timeData[i], corrData[i], corrDataOff[i], vOffset = removeBackgroundData(dictData)
        
        if normBool is True:
            try:
                exposureTime = dictData["Correlation time (s)"]
            except:
                exposureTime = exposureTimeArr[i]
            try:
                laserPower = dictData["Laser power (mW)"]
            except:
                laserPower = laserPowerArr[i]
            
            data = corrData[i]
            dataDel = (np.sqrt(data)/exposureTime/laserPower + 1e-10)*1e3 # assuming poissonian stats
            
            try:
                rate1 = dictData["Channel 1 count rate:"]
                rate2 = dictData["Channel 2 count rate:"]
                timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            except:
                rate1 = rate1Arr[i]
                rate2 = rate2Arr[i]
                timeBinWidth = timeBinWidthDefault
            offset = (rate1*rate2*timeBinWidth)/laserPower
            if offset == 0:
                offset = np.min(data)/exposureTime/laserPower
            # print(i)
            # print(offset)
            
            # offset = 0
            data = (data/exposureTime/laserPower)*1e3 - offset * 1e3* 1.015
            
        else:
            data = corrDataOff[i]
            dataDel = np.sqrt(corrData[i])/vOffset # assuming poissonian stats
            
        
        if avgBool == True:
            corr = data.copy()
            corrDel = data.copy()
            k = avgSample
            for j in range(len(data)-avgSample):
                corr[j] = np.mean(data[j:j+avgSample])
                corrDel[j] = np.mean(dataDel[j:j+avgSample])
            timeScaled = np.array(timeData[i][:-avgSample])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                if delayDoubleBool[i] == True:
                    x = fcut2wl(timeScaled)
                else:
                    x = fcut2wlHalf(timeScaled)
            
            filterArr = (x > min_x_value) & (x < max_x_value)
            x = x[filterArr]
            corr = np.array(corr)[:-avgSample]
            corr = corr[filterArr]
            corrDel = np.array(corrDel)[:-avgSample]
            corrDel = corrDel[filterArr]
            
            plotHandle[i], = ax[i].plot(x, np.array(corr), color = colorArray[i], label=legendHandles[i%2], lw = lw/2)
            ax[i].fill_between(x, np.array(corr-corrDel), 
                                                np.array(corr+corrDel), color = colorArray[i], alpha = 0.3)
            
                
        else:
            timeScaled = np.array(timeData[i])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                if delayDoubleBool[i] == True: #int(np.floor(i/2))
                    x = fcut2wl(timeScaled)
                else:
                    x = fcut2wlHalf(timeScaled)
            ax[i].plot(x, np.array(data), color = colorArray[0])
            ax[i].fill_between(x, data-dataDel, data+dataDel, color = colorArray[0], alpha = 0.3)
        i += 1
        
    for i in range(3):
        if plotDelayBool:
            
            ax[i].set_xlim([-22, 22])
            
            ax[i].set_ylim([0, None])
            # ax[i].set_xlim([1405, (1/788 - 1/1405)**-1])
        else:
            ax[i].set_xlim([-22, 22])
            ax[i].set_title(titles[i], fontsize = 30, pad = 10)
            
            ax[i].set_ylim([0, None])
            ax[i].set_xlim([1405, (1/788 - 1/1405)**-1])
            ax[i].grid()
        
    # label the lines
    # ax[1].plot([0, 8], [1.25, 1.25], color = colorArray[2], linewidth= 2)
    # ax[1].plot([1, 9], [1.13, 1.13], color = colorArray[0], linewidth = 2)
    # ax[1].text(9, 1.25, r'1500 nm', color = 'black', fontsize = 20, ha = 'left', va = 'center' )
    # ax[1].text(10, 1.13, r'1400 nm', color = 'black', fontsize = 20, ha = 'left', va = 'center' )
        
    if logBool == True:
        plt.yscale('log')
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    # ax.set_title(r"$g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}", fontsize = 25,pad=20)
    plt.tight_layout()
    
    
    limScale = 0.11
    
    for axx in ax:
        axx.set_ylim([0, None])
        axx.set_xticks([min_x_value, 0, max_x_value])
        
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)    
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=True)    
        axx.spines['bottom'].set_bounds(min_x_value,max_x_value)
        axx.xaxis.set_label_position("bottom")
        axx.xaxis.set_ticks_position('bottom')
        axx.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
        
        # fig.subplots_adjust(wspace=0.4)
    
    

    min_y_value = 0
    max_y_value = 4
    limScaley = 0.11
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley    
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 2
    # ax[1].set_yticks([0, 0.04, 0.08, 0.12])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 0.6
    # ax[2].set_yticks([0, 0.08, 0.16, 0.24])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[2].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[2].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_fiberDispersion.pdf", dpi=200, bbox_inches="tight"
        )
        
def plotFiberCoincidencesSeparateWithModel(dataDir, filePaths, titles = [r"transmission", r"reflection", r"tranmission/reflection"], 
                                  logBool = False, saveBool = False, avgBool = False, avgSample = 5, 
                                  normBool = False, delayDoubleBool = [], figName = "_fiberDispersionNorm", plotDelayBool = True,
                                  exposureTimeArr = [], laserPowerArr = [], rate1Arr = [], rate2Arr = [],
                                  timeBinWidthDefault = 100e-12, ):
    colorArray = [colorBlue, colorRed, colorPurple]
    
    fig, ax = plot_custom(
        1*PRQuantumWidthFull,  0.35*PRQuantumWidthFull, [r"wavelength, $\lambda$ (nm)", r"wavelength, $\lambda$ (nm)", r"wavelength, $\lambda$ (nm)"],
        [r'coincidence rate (cps/mW)', '', ''], labelPadX = 4, labelPadY = 4, 
        commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 3, nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize+1,
    
    )
    
    
    corrDataOff = [None]*(len(filePaths))
    plotHandle = [None]*(len(filePaths))
    timeData = corrDataOff.copy()
    corrData = corrDataOff.copy()
    
    
    min_x_value = 1400
    max_x_value = 1800
    rangex = max_x_value - min_x_value
       
    legendHandles = [r"1400 nm", r"1500 nm"]
    
    i = 0
    maxData = 0
    for file in filePaths:
        
        dictData = parseFile(dataDir + file)
        timeData[i], corrData[i], corrDataOff[i], vOffset = removeBackgroundData(dictData)
        
        if normBool is True:
            try:
                exposureTime = dictData["Correlation time (s)"]
            except:
                exposureTime = exposureTimeArr[i]
            try:
                laserPower = dictData["Laser power (mW)"]
            except:
                laserPower = laserPowerArr[i]
                
            # if i != 2:
            #     time, data = countsToSpectra(timeData[i], corrData[i], offset=0)
            
            data = corrData[i]
            dataDel = (np.sqrt(data)/exposureTime/laserPower + 1e-10)*1e3 # assuming poissonian stats
            
            try:
                rate1 = dictData["Channel 1 count rate:"]
                rate2 = dictData["Channel 2 count rate:"]
                timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            except:
                rate1 = rate1Arr[i]
                rate2 = rate2Arr[i]
                timeBinWidth = timeBinWidthDefault
            offset = (rate1*rate2*timeBinWidth)/laserPower
            if offset == 0:
                offset = np.min(data)/exposureTime/laserPower
            # print(i)
            # print(offset)
            
            # offset = 0
            if i == 0:
                scale = 1.035
                
            elif i == 1:
                scale = 1.015
                
            else:
                scale = 1.04
                        
            data = (data/exposureTime/laserPower)*1e3 - offset * 1e3* scale
            
        else:
            data = corrDataOff[i]
            dataDel = np.sqrt(corrData[i])/vOffset # assuming poissonian stats
            
        
        if avgBool == True:
            corr = data.copy()
            corrDel = data.copy()
            k = avgSample
            for j in range(len(data)-avgSample):
                corr[j] = np.mean(data[j:j+avgSample])
                corrDel[j] = np.mean(dataDel[j:j+avgSample])
            timeScaled = np.array(timeData[i][:-avgSample])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                if delayDoubleBool[i] == True:
                    x = fcut2wl(timeScaled)
                else:
                    x = fcut2wlHalf(timeScaled)
            
            filterArr = (x > min_x_value) & (x < max_x_value)
            x = x[filterArr]
            corr = np.array(corr)[:-avgSample]
            corr = corr[filterArr]
            corrDel = np.array(corrDel)[:-avgSample]
            corrDel = corrDel[filterArr]
            
            plotHandle[i], = ax[i].plot(x, np.array(corr), color = colorArray[i], label=legendHandles[i%2], lw = lw/2)
            ax[i].fill_between(x, np.array(corr-corrDel), 
                                                np.array(corr+corrDel), color = colorArray[i], alpha = 0.3)
            
            if maxData < np.max(np.array(corr)):
                maxData = np.max(np.array(corr))
                
        else:
            timeScaled = np.array(timeData[i])/1000
            if plotDelayBool == True:
                x = timeScaled
            else:
                if delayDoubleBool[i] == True: #int(np.floor(i/2))
                    x = fcut2wl(timeScaled)
                else:
                    x = fcut2wlHalf(timeScaled)
            ax[i].plot(x, np.array(data), color = colorArray[0])
                
            ax[i].fill_between(x, data-dataDel, data+dataDel, color = colorArray[0], alpha = 0.3)
    
        i += 1
    print(maxData)
    
    
    def gaussian(x, A, x0, sigma):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    popt_gaussian = envelopeFunctionDetectionEff()
    envelopeFunc = lambda lamb: gaussian(lamb, popt_gaussian[0], 1590, 70)
    
    def gaussianNew(w, w0, FWHM):
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
        return np.exp(-0.5 * ((w - w0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
    

    llamb = np.linspace(1.4e-6, 1.8e-6, 1000)
    L = 10.13e-6
    TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llamb, L = 10.15e-6, interferenceBool = True, collectionBool = True, envelopeFunc = envelopeFunc(llamb*1E9))
    _, _, TRRT = SPDCResonator.calcEtalonEffect(llamb, L = 10.15e-6 , interferenceBool = True, collectionBool = True, envelopeFunc = envelopeFunc(llamb*1E9))
  
            
    t0 = 1600                       # Center of the Gaussian
    FWHM = 15                    # Full width at half maximum of the Gaussian

    # Generate the Gaussian and function values
    gaussFunc = lambda t: gaussianNew(t, t0, FWHM)

    # Perform the convolution using FFT for speed
    conv = lambda f, t: fftconvolve(f, gaussFunc(t), mode='same') * (t[1] - t[0])  # Scale by the spacing
    
    zorder = 0
    modelColor = 'k'
    R = 2.5
    T = 0.95
    # ax[0].plot(llamb*1e9, gaussFunc(llamb**))
    ax[0].plot(llamb*1e9, conv(TT, llamb*1e9)*maxData/np.max(conv(TT, llamb*1e9))*T*T, zorder = zorder, color = modelColor, lw = lw)
    ax[1].plot(llamb*1e9, conv(RR, llamb*1e9)*maxData/np.max(conv(TT, llamb*1e9))*R*R, zorder = zorder, color = modelColor, lw = lw)
    ax[2].plot(llamb*1e9, conv(TRRT, llamb*1e9)*maxData/np.max(conv(TT, llamb*1e9))*T*R, zorder = zorder, color = modelColor, lw = lw)
        
        
        
    for i in range(3):
        if plotDelayBool:
            
            ax[i].set_xlim([-22, 22])
            
            ax[i].set_ylim([0, None])
            # ax[i].set_xlim([1405, (1/788 - 1/1405)**-1])
        else:
            ax[i].set_xlim([-22, 22])
            # ax[i].set_title(titles[i], fontsize = 30, pad = 10)
            
            ax[i].set_ylim([0, None])
            ax[i].set_xlim([1405, (1/788 - 1/1405)**-1])
            # ax[i].grid()
        
    # label the lines
    # ax[1].plot([0, 8], [1.25, 1.25], color = colorArray[2], linewidth= 2)
    # ax[1].plot([1, 9], [1.13, 1.13], color = colorArray[0], linewidth = 2)
    # ax[1].text(9, 1.25, r'1500 nm', color = 'black', fontsize = 20, ha = 'left', va = 'center' )
    # ax[1].text(10, 1.13, r'1400 nm', color = 'black', fontsize = 20, ha = 'left', va = 'center' )
        
    if logBool == True:
        plt.yscale('log')
    # ax.set_title(r"$P_{\text{pump}}=10$ mW;  $g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}" + r";  CR = 64 kc/s", fontsize = 25,pad=20)
    # ax.set_title(r"$g^{(2)}(0)=$ " + f"{A/vertOffset+ 1:0.1f}", fontsize = 25,pad=20)
    plt.tight_layout()
    
    
    limScale = 0.11
    
    for axx in ax:
        axx.set_ylim([0, None])
        axx.set_xticks([min_x_value, 1600, max_x_value])
        
        
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)    
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=True)    
        axx.spines['bottom'].set_bounds(min_x_value,max_x_value)
        axx.xaxis.set_label_position("bottom")
        axx.xaxis.set_ticks_position('bottom')
        axx.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
        
        fig.subplots_adjust(wspace=0.05)
    
    
    ax[1].spines["left"].set_visible(False)
    ax[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax[2].spines["left"].set_visible(False)
    ax[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    min_y_value = 0
    max_y_value = 3.5
    limScaley = 0.11
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley    
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 3.5
    # ax[1].set_yticks([0, 0.04, 0.08, 0.12])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 3.5
    # ax[2].set_yticks([0, 0.08, 0.16, 0.24])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[2].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[2].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
        
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_modelToDataCR.pdf", dpi=200, bbox_inches="tight"
        )
            
def plotPolTomography(filePaths, saveBool = False, figName = ""):
    g20 = np.zeros([2, 2])
    
    i=0
    for filePath in filePaths:
        dictData = parseFile(dataDir + filePath)
        timeData, corrData, corrDataNorm = calcNormData(dictData)
        
        g20[i%2, int(1*(i>1))] = np.max(calcg2FromNorm(corrDataNorm, timeData))
        
        i += 1
        
    fig, ax = plot_custom(
        6, 6, r"", r"", equal=False, labelPad = 0
    )
    
    p1 = ax.matshow(g20, cmap = 'hot', vmin = 0, vmax = 30)
    ax.set_yticks([0,1])
    ax.set_xticks([0,1])
    ax.set_yticklabels([r"$|ee\rangle$", r"$|oo\rangle$"])
    ax.set_xticklabels([r"$|e\rangle$", r"$|o\rangle$"])
    ax.set_title(r"pump polarization", fontsize = 30, pad = 20)
    ax.set_ylabel(r"photon polarization, $|si\rangle$", fontsize = 30, labelpad = 25 )
    
    cb = fig.colorbar(p1, fraction=0.045)
    cb.ax.tick_params(labelsize=20)
    cb.ax.set_ylabel(r'$g^{(2)}(0)$', fontsize = 30, rotation = 90, labelpad = 20, ha='center', va='center')
    # cb.ax.set_yticks([])
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_polTomography" + figName + ".png", dpi=200, bbox_inches="tight"
        )
        
# by polarization sweep I mean a HWP swept from 0 to 360 degrees before a glan-polarizer
        
def importPolarizationSweep(filepath):
    df = pd.read_csv(filepath)
    return df

def dfRow2dict(dfRow):
    result_dict = {}
    
    correlation = eval("np." + dfRow["correlation"])
    angle = dfRow["Angle"]
    exposureTime = dfRow["exposure time (s)"]
    binWidth = dfRow["bin width (ps)"]
    binNumber = dfRow["bin number"]
    countrate = eval("np." + dfRow["countrate"])
    
    timeCont = np.linspace(-np.ceil(binNumber/2)*binWidth, np.ceil(binNumber/2)*binWidth, binNumber)
        
    result_dict['Time data (ps)'] = timeCont
    result_dict['Correlation data'] = correlation
    result_dict["Channel 1 count rate:"] = countrate[0]
    result_dict["Channel 2 count rate:"] = countrate[1]
    result_dict["Correlation bin width (ps)"] = binWidth
    result_dict['Correlation time (s)'] = exposureTime
    
    return result_dict

def plotPolarizationCorrData(filepath):
    df = importPolarizationSweep(filepath)
    
    # initially loop through and plot all correlations on a single plot
    fig, ax = plot_custom(
        8, 6, r"", r"", equal=False, labelPad = 0
    )
    
    for index, row in df.iterrows():
        correlation = eval("np." + row["correlation"])
        angle = row["Angle"]
        exposureTime = row["exposure time (s)"]
        binWidth = row["bin width (ps)"]
        binNumber = row["bin number"]
        countrate = eval("np." + row["countrate"])
            
        timeCont = np.linspace(-np.ceil(binNumber/2)*binWidth, np.ceil(binNumber/2)*binWidth, binNumber)
        
        ax.plot(timeCont/1000, correlation)
        

def analyzePolarizationData(filepath, fitg2 = False, title = "", saveBool = False, offset = 0,  figName = "", HWPAngle = True, polType = 0):
    colorArray = sns.color_palette("hot", n_colors=10)
    
    df = importPolarizationSweep(filepath)
    
    # initially loop through and plot all correlations on a single plot
    
    g20 = np.zeros(len(df))
    g20Err = g20.copy()
    angles = g20.copy()
    
    
    for index, row in df.iterrows():
        dictData = dfRow2dict(row)
        timeData, corrData, corrDataNorm = calcNormData(dictData)

        if fitg2 == True:
            rate1 = dictData["Channel 1 count rate:"]
            rate2 = dictData["Channel 2 count rate:"]
            timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            exposureTime = dictData["Correlation time (s)"]
            
            g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
            g2Err = calcg2Err(corrData, timeBinWidth, rate1, rate2, exposureTime)
        else:
            g2 = calcg2FromNorm(corrDataNorm, timeData)
            g2Err = g2*0
            
        
        g20[index] = np.max(g2)
        g20Err[index] = np.max(g2Err)
        angles[index] = row["Angle"]
    
    anglesAnalyzer = angles[::2].copy()
    g20Analyzer = (g20[0:int(np.floor(len(df)/2))+ 1] + g20[int(np.floor(len(df)/2)):])/2
    g20AnalyzerErr = np.sqrt(g20Err[0:int(np.floor(len(df)/2))+ 1]**2 + g20Err[int(np.floor(len(df)/2)):]**2)
    
    if polType == 0:
        angleF = lambda theta: np.cos(theta)**4*75 + 1
    elif polType == 2:
        angleF = lambda theta: np.cos(theta)**2*np.sin(theta)**2*53 + 1
        
    fig, ax = plot_custom(
        textwidth/2.2, textwidth/2.2, r"", r"", labelPadX = 5, labelPadY = 10, axefont = labelfontsize, 
        numsize = numfontsize, 
        legfont = labelfontsize, radialBool = True
    )
    # ax.set_yticks(np.arange(5, 20, 5))
    if HWPAngle is True:
        e0 = ax.errorbar((angles-offset)*np.pi/180, g20, xerr = 1*np.pi/180, yerr = g20Err,
                    marker = 'o', zorder = 10, color = colorArray[2], capsize=3)
        ax.set_title(title, fontsize = labelfontsize, pad = 100)
        #theory
        theta = np.linspace(0, 2*np.pi, 300)
        # p0 = ax.plot(theta, np.cos(2*theta)**4*75+1, linewidth = 2, color = colorArray[5]) # Type 0
        p0 = ax.plot(theta, angleF(theta), linewidth = 2, color = colorArray[5]) # Type II
    else:
        e0 = ax.errorbar((anglesAnalyzer-2*offset)*np.pi/180, g20Analyzer, xerr = 1*np.pi/180, yerr = g20AnalyzerErr,
                    marker = 'o', zorder = 10, color = colorArray[2], capsize=3, markersize = 2, lw = 1)
        # ax.set_title(title, fontsize = 25, pad = 100)
        #theory
        theta = np.linspace(0, 2*np.pi, 300)
        # p0 = ax.plot(theta, np.cos(theta)**4*75 + 1, linewidth = 2, color = colorArray[5]) # Type 0
        p0 = ax.plot(theta, angleF(theta), linewidth = 2, color = colorArray[5]) # type II
    
    leg = ax.legend([ r"theory",r"experiment"], fontsize = labelfontsize, loc = [-0.3,1])
    leg.get_frame().set_linewidth(0.0)
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_polTomographySwept" + figName + ".png", dpi=200, bbox_inches="tight"
        )
        
def analyzePolarizationDataPaper(filepath, fitg2 = False, title = "", saveBool = False, offset = 0,  figName = "", HWPAngle = True, polType = 0):
    colorArray = [colorBlue, colorRed, colorPurple]
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.3, 0.3*PRQuantumWidthFull, r"", r"" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 6, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black', radialBool = True,
       nrow = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
        
    ax.set_rlabel_position(75)
    ax.set_yticks([40, 80])
    ax.set_ylim([0, 80])
    
    df = importPolarizationSweep(filepath)
    
    # initially loop through and plot all correlations on a single plot
    
    g20 = np.zeros(len(df))
    g20Err = g20.copy()
    angles = g20.copy()
    
    
    for index, row in df.iterrows():
        dictData = dfRow2dict(row)
        timeData, corrData, corrDataNorm = calcNormData(dictData)

        if fitg2 == True:
            rate1 = dictData["Channel 1 count rate:"]
            rate2 = dictData["Channel 2 count rate:"]
            timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            exposureTime = dictData["Correlation time (s)"]
            
            g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
            g2Err = calcg2Err(corrData, timeBinWidth, rate1, rate2, exposureTime)
        else:
            g2 = calcg2FromNorm(corrDataNorm, timeData)
            g2Err = g2*0
            
        
        g20[index] = np.max(g2)
        g20Err[index] = np.max(g2Err)
        angles[index] = row["Angle"]
    
    anglesAnalyzer = angles[::2].copy()
    g20Analyzer = (g20[0:int(np.floor(len(df)/2))+ 1] + g20[int(np.floor(len(df)/2)):])/2
    g20AnalyzerErr = np.sqrt(g20Err[0:int(np.floor(len(df)/2))+ 1]**2 + g20Err[int(np.floor(len(df)/2)):]**2)
    
    if polType == 0:
        angleF = lambda theta: np.cos(theta)**4*75 + 1
    elif polType == 2:
        angleF = lambda theta: np.cos(theta)**2*np.sin(theta)**2*53 + 1
    

    e0 = ax.errorbar((anglesAnalyzer-2*offset)*np.pi/180, g20Analyzer, xerr = 1*np.pi/180, yerr = g20AnalyzerErr,
                marker = 'o', zorder = 10, color = colorArray[2], capsize=3, markersize = 1, lw = 1)

    #theory
    theta = np.linspace(0, 2*np.pi, 300)
    p0 = ax.plot(theta, angleF(theta), linewidth = 2, color = colorArray[0], alpha = 0.5) # type II
    
    # leg = ax.legend([ r"fit",r"experiment"], fontsize = labelfontsize, loc = [-0.3,1])
    # leg.get_frame().set_linewidth(0.0)
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_polTomographySwept" + figName + ".svg", dpi=200, bbox_inches="tight", transparent=True
        )
        
def plotFigure2Top(saveBool):       
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.7, 0.33*PRQuantumWidthFull, [r"time delay, $\tau$ (ns)",r"pump power (mW)"],
       ["coin. rate (cps/mW)","real coin. rate (cps)"], labelPadX = 3, labelPadY = 4, commonY = False, wSpaceSetting=100,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 2, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = True,
       widthRatios = [1,1], heightRatios=[1]
    )
    
    colorArray = [colorBlue, colorRed, colorPurple]
    
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/"
    filePathExtraT = 'LN-Si/pumpPowerVar/transmission/'
    filePathsPowerTrans = [
        filePathExtraT+'20230705_1053_9mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1102_18mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1108_37mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1130_55mW_coincidences_zpol_transmission_web.txt',
        filePathExtraT+'20230705_1135_82mW_coincidences_zpol_transmission_web.txt',
        ]
    
    filePathExtraR = 'LN-Si/pumpPowerVar/reflection/'    
    filePathsPowerRefl = [
        filePathExtraR+'20230622_1255_5mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1305_10mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1312_20mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1319_30mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1323_50mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1328_70mW_coincidences_zpol.txt',
        ]
    
    filePathExtraFB = 'LN-Si/pumpPowerVar/frontBack/'
    filePathsPowerTransRefl = [
        filePathExtraFB+'20230705_1640_9mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1635_18mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1628_37mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1613_55mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1618_82mW_coincidences_zpol_transmission_reflection_web.txt',
        ]
    
    powersRefl = np.array([5, 10, 20, 30, 50, 70])
    powersReflDel = 0.05*powersRefl # uncertainty of 5%
    powersTransRefl = np.array([9.47, 18.16, 37.44, 55.37, 81.55]) # uncertainty of 5%
    powersTransReflDel = 0.05*powersTransRefl
    powersTrans = powersTransRefl
    powersTransDel = powersTransReflDel
    
    
    i = 0
    dictData = parseFile(dataDir+filePathsPowerTrans[i])
    powers = [9,18,37,55,82]
    PPumpmW = powers[i]
    timeData, corrData, corrDataNorm = calcNormData(dictData)
    fitg2 = True
    if fitg2 == True:
        rate1 = dictData["Channel 1 count rate:"]
        rate2 = dictData["Channel 2 count rate:"]
        timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
        exposureTime = dictData["Correlation time (s)"]
        
        g2 = calcg2(corrData, timeBinWidth, rate1, rate2, exposureTime)
    else:
        g2 = calcg2FromNorm(corrDataNorm, timeData)
        
    min_x_value = -5
    max_x_value = 5
    min_y_value = 0
    max_y_value = 0.125
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    timeData = timeData/1000 
    corrDataDel = np.sqrt(corrData)
    
    filterData = np.logical_and(np.array(timeData < max_x_value), np.array(timeData > min_x_value))
    corrData = corrData[filterData]
    corrDataDel = corrDataDel[filterData]
    timeData = timeData[filterData]
    
    # ax[0].plot(timeData, corrData/exposureTime/PPumpmW, color = colorArray[0], lw = 0.1)

    # ax[0].fill_between(timeData, np.array(corrData-corrDataDel)/exposureTime/PPumpmW, 
    #                                     np.array(corrData+corrDataDel)/exposureTime/PPumpmW, color = colorArray[0], alpha = 0.3)
    ax[0].errorbar(timeData, corrData/exposureTime/PPumpmW, yerr = corrDataDel/exposureTime/PPumpmW, 
                marker = '.', linestyle = 'None', color = colorArray[0], zorder = 10, ms = 1, capsize=1, capthick=0.5, elinewidth  = 0.5)
    ax[0].plot(timeData, corrData/exposureTime/PPumpmW, color = colorArray[0], lw = 0.5, alpha = 0.5)
    
    limScale = 0.08
    limScaley = 0.08
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax[0].set_xticks([-5, 0, 5])
       
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].yaxis.set_ticks_position('left')
    ax[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[0].tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax[0].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[0].xaxis.set_label_position("bottom")
    ax[0].xaxis.set_ticks_position('bottom')
    
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    
    ax[0].set_yticks([0, 0.05, 0.1])
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax[0].set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
    
    min_x_value = 0
    max_x_value = 100
    min_y_value = 0
    max_y_value = 100
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.1
    limScaley = 0.08
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    

    def returnFittedData(powers, data, powerMin = min_x_value, powerMax = max_x_value, N = 2):
        def linFit(x,a):
            return a * x
        popt, pcov = curve_fit(linFit, powers, data)
        
        powerCont = np.linspace(powerMin, powerMax, N)
        dataFit = linFit(powerCont, *popt)
        return powerCont, dataFit
        
    
    # plot the rate of real coincidences 
    exposureTimes = [0, 0, 0, 150, 75]
    countrate = [None]*5
    for i in range(5):
        countrate[i] = calcRealCoincidenceRate(dataDir+filePathsPowerTrans[i],PpumpmW = powersTrans[i], exposureTime=exposureTimes[i])
    powerFit, dataFit = returnFittedData(powersTrans, countrate)
    ax[1].plot(powerFit, dataFit, color = colorArray[0], lw = lw, alpha = 0.5)
        
    ax[1].scatter(powersTrans, countrate,marker = '.', color = colorArray[0], s = 5)
        
        
    countrate = [None]*6
    for i in range(6):
        countrate[i] = calcRealCoincidenceRate(dataDir+filePathsPowerRefl[i], exposureTime=1, PpumpmW = powersRefl[i])
    powerFit, dataFit = returnFittedData(powersRefl, countrate)
    ax[1].plot(powerFit, dataFit, color = colorArray[1], lw = lw, alpha = 0.5)
    
    ax[1].scatter(powersRefl, countrate,marker = '.', color = colorArray[1], s = 5)

    countrate = [None]*5
    exposureTimes = [200, 200, 300, 100 ,100] #s
    for i in range(5):
        countrate[i] = calcRealCoincidenceRate(dataDir+filePathsPowerTransRefl[i], exposureTime=exposureTimes[i],PpumpmW = powersTransRefl[i])
    powerFit, dataFit = returnFittedData(powersTransRefl, countrate)
    ax[1].plot(powerFit, dataFit, color = colorArray[2], lw = lw, alpha = 0.5)
    
    ax[1].scatter(powersTransRefl, countrate,marker = '.', color = colorArray[2], s = 5)

    

    
       
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].yaxis.set_ticks_position('left')
    ax[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[1].tick_params(axis='x', which='both', top=False, bottom=True)
    ax[1].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[1].xaxis.set_label_position("bottom")
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    
    ax[1].set_yticks([0,20,40,60, 80, 100])
    
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax[1].set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])

    
    ax[1].tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax[1].tick_params(axis='y', pad=4)  # Padding for y-axis tick labels
    ax[0].tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax[0].tick_params(axis='y', pad=4)  # Padding for y-axis tick labels
    fig.subplots_adjust(wspace=0.35)
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        
        plt.savefig(
            "figures/" + preamble + "_SPDCg2tPaper" + ".svg", dpi=200, bbox_inches="tight",transparent=True
        )
    
def plotFigure2g20(saveBool= False):
    # (0,0) empty figure for schematic
    # (0,1) g(2)(t) for the Si LN device
    # (0,2) CAR and coincidence rate as a function of power
    # (1,1:3) spectra of etalon generation
    
    colorArray = [colorBlue, colorRed, colorPurple]
    
    fig, ax = plot_custom(
       0.24*PRQuantumWidthFull, 0.18*PRQuantumWidthFull, r"$P_{\text{pump}}$ (mW)",
       r"$g^{(2)}(0)$", labelPadX = 2, labelPadY = 5, commonY = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=labelfontsize,
       widthBool = True,
       widthRatios = [1], heightRatios=[1]
    )
    
    def fitInverse(x, A):
        return A/x + 1
    
    filePathExtraFB = 'LN-Si/pumpPowerVar/frontBack/'
    filePathsPowerTransRefl = [
        filePathExtraFB+'20230705_1640_9mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1635_18mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1628_37mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1613_55mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1618_82mW_coincidences_zpol_transmission_reflection_web.txt',
        ]
    
    filePathExtraR = 'LN-Si/pumpPowerVar/reflection/'    
    filePathsPowerRefl = [
        filePathExtraR+'20230622_1255_5mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1305_10mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1312_20mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1319_30mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1323_50mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1328_70mW_coincidences_zpol.txt',
        ]
    
    filePathExtraT = 'LN-Si/pumpPowerVar/transmission/'
    filePathsPowerTrans = [
        filePathExtraT+'20230705_1053_9mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1102_18mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1108_37mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1130_55mW_coincidences_zpol_transmission_web.txt',
        filePathExtraT+'20230705_1135_82mW_coincidences_zpol_transmission_web.txt',
        ]
    
    powersRefl = np.array([5, 10, 20, 30, 50, 70])
    powersReflDel = 0.05*powersRefl # uncertainty of 5%
    powersTransRefl = np.array([9.47, 18.16, 37.44, 55.37, 81.55]) # uncertainty of 5%
    powersTransReflDel = 0.05*powersTransRefl
    powersTrans = powersTransRefl
    powersTransDel = powersTransReflDel
    
    # plotg2Power(dataDir, filePathsPowerTransRefl, powersTransRefl, powersTransReflDel, saveBool = saveBool, figName = "_transmissionReflection")
    
    # plot combined plots
    
    
    filePathArr = [filePathsPowerTrans, filePathsPowerRefl,  filePathsPowerTransRefl]
    powersArr = [ powersTrans,powersRefl, powersTransRefl]
    powersDel = [powersTransDel,powersReflDel,  powersTransReflDel]
    
    
    j = 0
    
    for filePaths in filePathArr:
        g20 = np.zeros(len(filePaths))
        
        i = 0
        for filePath in filePaths:
            dictData = parseFile(dataDir + filePath)
            timeData, corrData, corrDataNorm = calcNormData(dictData)

            g20[i] = np.max(calcg2FromNorm(corrDataNorm, timeData))
            # g20[i] = np.sum(corrDataNorm[corrDataNorm > 2]-1)#car
            i += 1

        param, cov = curve_fit(fitInverse, powersArr[j], g20)
        ppowerCont = np.linspace(5, 85, 500)

        print(g20)
        ax.errorbar(powersArr[j], g20, xerr = powersDel[j], yerr = np.sqrt(g20), 
                    marker = '.', linestyle = 'None', color = colorArray[j], zorder = 10, ms = 2, capsize=1, capthick=0.5, elinewidth  = 0.5)

        pfit1, = ax.plot(ppowerCont, fitInverse(ppowerCont, param[0]), color = colorArray[j], zorder = 1, alpha = 0.6, lw = lw)

        # ax.grid()
        j += 1
        
    min_x_value = 5
    max_x_value = 85
    min_y_value = 0
    max_y_value = 100
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.12
    limScaley = 0.2
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
       
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    
    
    ax.set_xticks(filtered_xticks)
    ax.set_xticks([5,85])
    
    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value])
    
    ax.tick_params(axis='x', pad=3)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_SPDCg20PaperInset" + ".svg", dpi=200, bbox_inches="tight",transparent=True
        )
    
def plotFigure2Spectra(saveBool = False, L = 10.145e-6-2*0.788e-6/2.1, envelopeFunc = lambda lamb: 1 ):
    # (0,0) empty figure for schematic
    # (0,1) g(2)(t) for the Si LN device
    # (0,2) CAR and coincidence rate as a function of power
    # (1,1:3) spectra of etalon generation 
    colorArray = [colorBlue, colorRed, colorPurple]
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.35, 0.45*PRQuantumWidthFull, r"wavelength, $\lambda$ (nm)", ["",r"normalized intensity (au)", ""] #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 6, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = True,  heightRatios = [1,1,1.25], widthRatios=[1]
    )
    
    min_x_value = 1400
    max_x_value = 1800
    min_y_value = 0
    max_y_value = 1
    
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.05
    limScaley = 0.04
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
        
    normBool = True
    avgBool = True


    figName = "Fiber spectroscopy comparison: 788 nm pump, 3 km SMF-28"
    logBool = False
    plotDelayBool = False

    fileDir = 'LN-Si/'
    filepaths = [
        fileDir + "fiberSpectroscopy/reflection/20230712_1616_50mW_coincidences_zpol_lp1400_fiber_combined.txt",
        # fileDir + "fiberSpectroscopy/reflection/20230712_1614_50mW_coincidences_zpol_lp1500_fiber_combined.txt",
        fileDir + "fiberSpectroscopy/transmission/20230707_1625_50mW_coincidences_zpol_lp1400_transmission_fiber.txt",
        # fileDir + "fiberSpectroscopy/transmission/20230707_1800_50mW_coincidences_zpol_lp1500_transmission_fiber.txt",
        fileDir + "fiberSpectroscopy/frontBack/20230713_0104_50mW_coincidences_zpol_lp1400_transmission_reflection_fiber_web.txt",
        # fileDir + "fiberSpectroscopy/frontBack/20230712_0833_50mW_coincidences_zpol_lp1500_transmission_reflection_fiber_web.txt",
        
        ]
    
    avgSampleArr = [10,10,3]
    expTime1 = 60*(33 + 60*(8 + 10))
    expTime2 = 60*(33 + 60*(8 + 10))
    exposureTimeArr = np.array([None, None, None, None, expTime1, expTime2])
    laserPowerArr = [50, 50, 50, 50, 50, 50]
    rate1Arr = [None, None, None, None, 0, 0]
    rate2Arr = [None, None, None, None, 0, 0]
    
    exposureTimeArr = np.array([None, None, expTime1])
    laserPowerArr = [50, 50, 50]
    rate1Arr = [None, None, 0]
    rate2Arr = [None, None, 0]
    delayDoubleBool = [True, True, False]
    timeBinWidthDefault = 100e-12
    
    corrDataOff = [None]*(len(filePaths))
    plotHandle = [None]*(len(filePaths))
    timeData = corrDataOff.copy()
    corrData = corrDataOff.copy()
       
    legendHandles = [r"1400 nm", r"1500 nm"]
    
    i = 0    
    for file in filepaths:
        avgSample = avgSampleArr[i]
        
        dictData = parseFile(dataDir + file)
        timeData[i], corrData[i], corrDataOff[i], vOffset = removeBackgroundData(dictData)
        
        if normBool is True:
            try:
                exposureTime = dictData["Correlation time (s)"]
            except:
                exposureTime = exposureTimeArr[i]
            try:
                laserPower = dictData["Laser power (mW)"]
            except:
                laserPower = laserPowerArr[i]
            
            if i != 2:
                time, data = countsToSpectra(timeData[i], corrData[i], offset=0)

            else:
                time = timeData[i]
                data = corrData[i]
            
            if avgBool == True:
                # Calculate the number of blocks
                n = avgSample
                num_blocks = len(data) // n
                # Downsample the data array by summing over every n elements
                data = np.array([np.sum(data[k*n:(k+1)*n]) for k in range(num_blocks)])

                # Downsample the time array by averaging over every n elements
                time = np.array([np.mean(time[k*n:(k+1)*n]) for k in range(num_blocks)])


            


            timeScaled = np.array(time)/1000
            if delayDoubleBool[i] == True:
                x = fcut2wl(timeScaled)
            else:
                x = fcut2wlHalf(timeScaled)
                
            # data = data* (x)**4 /788**4/2**4
                
            # data = corrData[i]
            dataDel = (np.sqrt(data)/exposureTime/laserPower/avgSample + 1e-10)*1e3 # assuming poissonian stats
            
            try:
                rate1 = dictData["Channel 1 count rate:"]
                rate2 = dictData["Channel 2 count rate:"]
                timeBinWidth = (dictData["Correlation bin width (ps)"])*(10**-12)
            except:
                rate1 = rate1Arr[i]
                rate2 = rate2Arr[i]
                timeBinWidth = timeBinWidthDefault
            offset = (rate1*rate2*timeBinWidth)/laserPower
            if offset == 0:
                offset = np.min(data)/exposureTime/laserPower/avgSample

            data = (data/exposureTime/laserPower/avgSample)*1e3 - offset * 1e3
            dataDel = dataDel/np.max(data)
            data = data/np.max(data)
            

            
        
            
        # fiter data
        filterData = np.logical_and(np.array(x < max_x_value), np.array(x > min_x_value))
        data = data[filterData]
        dataDel = dataDel[filterData]
        x = x[filterData]
        
        corr = data
        corrDel = dataDel
        
        def gaussian(w, w0, FWHM):
            sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
            return np.exp(-0.5 * ((w - w0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))



            
        plotHandle[i], = ax[i].plot(x, data, color = colorArray[i], label=legendHandles[i%2], lw = lw)
        # plotHandle[i] = ax[i].errorbar(x, data, xerr=0, yerr=corrDel, color = colorArray[i], label=legendHandles[i%2], fmt=".",ms = 2)

        ax[i].fill_between(x, np.array(corr-corrDel), 
                                            np.array(corr+corrDel), color = colorArray[i], alpha = 0.3)
            
        i += 1
        
    boolPlotComplex = False
    if boolPlotComplex:
        llamb, St, Sr, Str, L = SPDCSpectrumDetection.calcTotalEfficiencyArrays(L = L, numPoints=1000) #8.7e-6

        zorder = 0
        modelColor = 'k'
        ax[0].plot(llamb*1e9, St/np.max(St), zorder = zorder, color = modelColor, lw = lw)
        ax[1].plot(llamb*1e9, Sr/np.max(Sr), zorder = zorder, color = modelColor, lw = lw)
        ax[2].plot(llamb*1e9, Str/np.max(Str), zorder = zorder, color = modelColor, lw = lw)
        # ax.set_ylim([0.5,1.5])
        # ax.set_title("$L = $" + f"{L*1e6:.2f}" + " um", fontsize = 25)
        # ax[2].legend(['trans', 'refl', 'both'])
    else:
        llamb = np.linspace(1.4e-6, 1.8e-6, 1000)
        print(L)
        TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llamb, L = L, interferenceBool = True, collectionBool = True, envelopeFunc = envelopeFunc(llamb*1E9))
        _, _, TRRT = SPDCResonator.calcEtalonEffect(llamb, L = L*1.001 , interferenceBool = True, collectionBool = True, envelopeFunc = envelopeFunc(llamb*1E9))
        # lambP = 0.788e-6           
        # wp = optics.lamb2omega(lambP)
        # ws = optics.lamb2omega(llamb)
        
        # theta = np.linspace(-np.pi/50, np.pi/50,200)
        # ttheta,llambs = np.meshgrid(theta, llamb, indexing='ij');
        # wws = optics.lamb2omega(llambs)
        
        # FAS = np.sum(np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))**2, axis = 0)
        
        # TT *= FAS
        # RR *= FAS
        # TRRT *= FAS
                
        t0 = 1600                       # Center of the Gaussian
        FWHM = 15                    # Full width at half maximum of the Gaussian

        # Generate the Gaussian and function values
        gaussFunc = lambda t: gaussian(t, t0, FWHM)

        # Perform the convolution using FFT for speed
        conv = lambda f, t: fftconvolve(f, gaussFunc(t), mode='same') * (t[1] - t[0])  # Scale by the spacing
        
        zorder = 0
        modelColor = 'k'
        # ax[0].plot(llamb*1e9, gaussFunc(llamb**))
        ax[0].plot(llamb*1e9, conv(TT, llamb*1e9)/max(conv(TT, llamb*1e9)), zorder = zorder, color = modelColor, lw = lw)
        ax[1].plot(llamb*1e9, conv(RR, llamb*1e9)/max(conv(RR, llamb*1e9)), zorder = zorder, color = modelColor, lw = lw)
        ax[2].plot(llamb*1e9, conv(TRRT, llamb*1e9)/max( conv(TRRT, llamb*1e9)), zorder = zorder, color = modelColor, lw = lw)
        
        # ax[0].plot(llamb*1e9, TT/max(TT), zorder = zorder, color = modelColor, lw = lw)
        # ax[1].plot(llamb*1e9, RR/max(RR), zorder = zorder, color = modelColor, lw = lw)
        # ax[2].plot(llamb*1e9, TRRT/max(TRRT), zorder = zorder, color = modelColor, lw = lw)
        
    
    

        
    for i in range(3):
        
        ax[i].set_ylim([0, 1])
        ax[i].set_xlim([1400, 1800])
        # ax[i].set_xlim([1400, (1/788 - 1/1400)**-1])
        # ax[i].grid()
        
    for i in range(3):
        ax[i].set_yticks([0,1])
        
        
    # ax[0].spines["top"].set_visible(False)
    # ax[0].spines["right"].set_visible(False)
    # ax[0].spines["bottom"].set_visible(False)
    # ax[0].yaxis.set_ticks_position('left')
    # ax[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    # ax[0].tick_params(axis='x', which='both', top=False, bottom=False, labelleft=True)
    
    # ax[1].spines["right"].set_visible(False)
    # ax[1].spines["top"].set_visible(False)
    # ax[1].spines["bottom"].set_visible(False)
    # ax[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    # ax[1].tick_params(axis='x', which='both', top=False, bottom=False, labelleft=True)
        
    # ax[2].spines["right"].set_visible(False)
    # ax[2].spines["top"].set_visible(False)
    # ax[2].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["left"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    ax[0].yaxis.set_ticks_position('right')
    ax[0].tick_params(axis='y', which='both', left=False, right=True, labelleft=False)
    ax[0].tick_params(axis='x', which='both', top=False, bottom=False)
    
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].yaxis.set_ticks_position('right')
    ax[1].tick_params(axis='y', which='both', left=False, right=True, labelleft=False)
    ax[1].tick_params(axis='x', which='both', top=False, bottom=False)
        
    ax[2].spines["left"].set_visible(False)
    ax[2].spines["top"].set_visible(False)
    ax[2].yaxis.set_ticks_position('right')
    ax[2].tick_params(axis='y', which='both', left=False, right=True, labelleft=False)
    ax[2].tick_params(axis='x', which='both', top=False, bottom=True)
    
    fig.subplots_adjust(hspace=0.2)

    
    ax[2].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[2].xaxis.set_label_position("bottom")
    ax[2].xaxis.set_ticks_position('bottom')
        
    
    # ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    # ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    # ax[2].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[0].spines['right'].set_bounds(min_y_value,max_y_value)
    ax[1].spines['right'].set_bounds(min_y_value,max_y_value)
    ax[2].spines['right'].set_bounds(min_y_value,max_y_value)
    
    # ax[1].yaxis.set_label_position("left")
    # ax[1].yaxis.set_ticks_position('left')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.set_ticks_position('right')
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    # Set the filtered ticks
    ax[0].set_yticks(filtered_yticks)
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    ax[1].set_yticks(filtered_yticks)
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    ax[2].set_xticks(filtered_xticks)
    ax[2].set_yticks(filtered_yticks)
    
    ax[2].set_ylim([min_y_value - limScaleMin-0.2, max_y_value + limScaleMax])
    ax[2].set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
    
    ax[0].tick_params(axis='y', pad=5)  # Padding for y-axis tick labels
    ax[1].tick_params(axis='y', pad=5)  # Padding for y-axis tick labels
    ax[2].tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax[2].tick_params(axis='y', pad=5)  # Padding for y-axis tick labels
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_SPDCSpectraSiPaper" + ".svg", dpi=200, bbox_inches="tight",transparent=True
        )

def plotSimpleSpectraPaper(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6, normBool = False, 
                             interferenceBool = False, emissionProbBool = True, detEffTrans = 1, detEffRefl = 1,
                             etalonBool = True, interfaceType=2):
    colorArray = [colorBlue, colorRed, colorPurple]
    
    llamb = np.linspace(lambMin, lambMax, numPoints)
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.25, 0.25*PRQuantumWidthFull, r"wavelength, $\lambda$", r"intensity (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 6, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_x_value = lambMin*1e9
    max_x_value = lambMax*1e9
    min_y_value = 0
    max_y_value = 1
    
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.15
    limScaley = 0.1
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llamb, lambP, L, interferenceBool = False,interfaceType=interfaceType)
        
    if emissionProbBool is True:
        wp = optics.lamb2omega(lambP)
        ws = optics.lamb2omega(llamb)
        
        theta = np.linspace(-np.pi/50, np.pi/50,200)
        ttheta,llambs = np.meshgrid(theta, llamb, indexing='ij');
        wws = optics.lamb2omega(llambs)
        
        FAS = np.sum(np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))**2, axis = 0)
        
        if etalonBool:
            TT *= FAS
            RR *= FAS
            TRRT *= FAS
        else:
            TT = FAS
            RR = FAS
            TRRT = FAS
    
    
    ax.plot(llamb*1e9, TT/max(TT), linestyle = '-', color = colorOrange, lw = lw)
    # ax.plot(llamb*1e6, RR/max(TT), linestyle = '--', color = colorArray[1])
    # ax.plot(llamb*1e6, TRRT/max(TT), linestyle = ':', color = colorArray[2])
    

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    
    
    ax.set_xticks(filtered_xticks)
    ax.set_xticks([1600])
    ax.set_xticklabels([r'$2\lambda_{\text{pump}}$'])
    ax.set_yticks([0,1])
    
    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value])
    
    ax.tick_params(axis='x', pad=3)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
        
    
def plotFASForPaper(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10, chi = 0.00001, s = 1, saveBool=False, etalonBool = False):
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,M)
        
        llambs,ttheta = np.meshgrid(lambs, theta, indexing='ij');
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)

        # fig, ax = plot_custom(
        #    textwidth/1.5, textwidth/2.5, r" $\theta$ ($^\circ$)", 
        #    r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
        #    numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
        #    labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
        
        fig, ax = plot_custom(
           PRQuantumWidthFull*0.4, 0.15*PRQuantumWidthFull, r"$\omega_s/2\pi$ (THz)",r" $\theta$ ($^\circ$)" #"coincidence rate (mHz/mW)"
           , labelPadX = -3, labelPadY = 4, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
           spineColor = 'black', tickColor = 'black', textColor = 'black',
           nrow = 1, fontType = 'sanSerif', axefont = labelfontsize-1, numsize=numfontsize,
           widthBool =False
        )
        
        axg0 = ax.twiny()
        
        axg0.tick_params(
            axis="x",
            which="major",
            width=1,
            length=2,
            labelsize=numfontsize,
            zorder=1,
            direction="in",
            pad=2,
            top="off",
            colors='black'
        )
        
        
        FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L, chi = chi, s = s))
        
        
        if etalonBool is True:
            TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llambs, lambp, L, theta = np.abs(ttheta), interferenceBool=True)
            TT = np.nan_to_num(TT)
            FAS = FAS*TT
        

        wBack = np.min( wws*1e-12/(2*np.pi))
        backgroundw = np.array([[2*wBack,  2*wBack],[wBack, wBack]])
        backgroundthe = np.array([[-10, 10],[-10, 10]])
        background = np.array([[0.788,  2*0.788],[0.788,  2*0.788]])*0
        
        p3 = ax.pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
        
        p1 = ax.pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), FAS/np.max(FAS), cmap='plasma')
        # axg0.pcolormesh(llambs*1e6, np.degrees(ttheta),FAS/np.max(FAS), cmap='plasma', alpha = 0)
        
        axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))

        # axg0.xaxis.set_label_position("bottom")
        # axg0.xaxis.set_ticks_position('bottom')
        
        axg0.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 0)
        
        cbar = fig.colorbar(p1, ax = axg0, pad=0.05, aspect = 4)
        cbar.ax.tick_params(labelsize=numfontsize)
        cbar.set_label(r'', rotation = 90, fontsize = labelfontsize, labelpad = 5)
        cbar.set_ticks([0,1])
        
        #print lines of TIR in crystal
        axg0.set_xticks([0.788,  2*0.788])
        axg0.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
        ax.set_xticks([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
        ax.set_xticklabels([r'$\omega_p/4\pi$',  r'$\omega_p/2\pi$'])
        ax.set_xlim([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
        ax.set_ylim([-10,10])
    
        if saveBool is True:        
            dateTimeObj = datetime.now()
            preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
    
            plt.savefig(
                "figures/" + preamble + "_SPDCSpectra" + ".png", dpi=200, bbox_inches="tight",transparent=True
            )
            
def plotNLFASForPaper(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10,s = 1, chi = 1e-10,saveBool=False, etalonBool = False):
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,M)
        
        llambs,ttheta = np.meshgrid(lambs, theta, indexing='ij');
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)
        
        St = llambs.copy()*0
        Sr = llambs.copy()*0
        Str = llambs.copy()*0
        
        for i in range(M):
            # Sr[:,i] = SPDCResonator.calcSpectrumSPDCRefl(lambs, lambp, theta = theta[i], L = L, s= s, chi = chi)
            # Str[:,i] = SPDCResonator.calcSpectrumSPDCTransRefl(lambs, lambp, theta = theta[i], L = L, s=s, chi = chi)
            St[:,i] = np.nan_to_num(SPDCResonator.calcSpectrumSPDCTrans(lambs, lambp, theta = theta[i], L = L, s= s, chi = chi))

        # fig, ax = plot_custom(
        #    textwidth/1.5, textwidth/2.5, r" $\theta$ ($^\circ$)", 
        #    r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
        #    numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
        #    labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
        
        fig, ax = plot_custom(
           PRQuantumWidthFull*0.4, 0.15*PRQuantumWidthFull, r"$\omega_s/2\pi$ (THz)",r" $\theta$ ($^\circ$)" #"coincidence rate (mHz/mW)"
           ,labelPadX = -3, labelPadY = 4, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
           spineColor = 'black', tickColor = 'black', textColor = 'black',
           nrow = 1, fontType = 'sanSerif', axefont = labelfontsize-1, numsize=numfontsize,
           widthBool =False
        )
        
        axg0 = ax.twiny()
        
        axg0.tick_params(
            axis="x",
            which="major",
            width=1,
            length=2,
            labelsize=numfontsize,
            zorder=1,
            direction="in",
            pad=2,
            top="off",
            colors='black'
        )
        


        wBack = np.min( wws*1e-12/(2*np.pi))
        backgroundw = np.array([[2*wBack,  2*wBack],[wBack, wBack]])
        backgroundthe = np.array([[-10, 10],[-10, 10]])
        background = np.array([[0.788,  2*0.788],[0.788,  2*0.788]])*0
        
        p3 = ax.pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
        
        p1 = ax.pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), St/np.max(St), cmap='plasma')
        axg0.pcolormesh(llambs*1e6, np.degrees(ttheta),St/np.max(St), cmap='plasma', alpha = 0)
        
        axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))

        # axg0.xaxis.set_label_position("bottom")
        # axg0.xaxis.set_ticks_position('bottom')
        
        axg0.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 0)
        
        cbar = fig.colorbar(p1, ax = axg0, pad=0.05, aspect = 4)
        cbar.ax.tick_params(labelsize=numfontsize)
        cbar.set_label(r'', rotation = 90, fontsize = labelfontsize, labelpad = 5)
        
        #print lines of TIR in crystal
        axg0.set_xticks([0.788,  2*0.788])
        axg0.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
        ax.set_xticks([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
        ax.set_xticklabels([r'$\omega_p/4\pi$',  r'$\omega_p/2\pi$'])
        ax.set_xlim([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
        ax.set_ylim([-10,10])
    
    
        if saveBool is True:        
            dateTimeObj = datetime.now()
            preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
    
            plt.savefig(
                "figures/" + preamble + "_SPDCSpectra" + ".png", dpi=200, bbox_inches="tight",transparent=True
            )
            
def plotFASGridForPaper(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10,s = 1, chi = 1e-10,saveBool=False, etalonBool = False):
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,M)
        
        llambs,ttheta = np.meshgrid(lambs, theta, indexing='ij');
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)

        fig, axArr = plot_custom(PRQuantumWidthFull*0.5, 0.3*PRQuantumWidthFull, ["","", ""], ["", "", "", ""]#[r"$\omega_s/2\pi$ (THz)",r"$\omega_s/2\pi$ (THz)", ''],[r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)"] #"coincidence rate (mHz/mW)"
           , labelPadX = -3, labelPadY = 2, commonY = True, commonX = True, wSpaceSetting=0.0, hSpaceSetting = 0.0,
           spineColor = 'black', tickColor = 'black', textColor = 'black',
           nrow = 3, ncol = 3, fontType = 'sanSerif', axefont = labelfontsize-1, numsize=numfontsize-2,
           widthBool =True, widthRatios=[10, 10, 1], heightRatios = [1,1,1],
        )
                        
        axg0 = axArr[0,0].twiny()
        axg1 = axArr[0,1].twiny()
        
        axg0.tick_params(
            axis="x",
            which="major",
            width=1,
            length=2,
            labelsize=numfontsize-2,
            zorder=1,
            direction="in",
            pad=2,
            top="off",
            colors='black'
        )
        axg1.tick_params(
            axis="x",
            which="major",
            width=1,
            length=2,
            labelsize=numfontsize-2,
            zorder=1,
            direction="in",
            pad=2,
            top="off",
            colors='black'
        )
        fig.subplots_adjust(hspace=0.25, wspace = 0.18)
        
        St = llambs.copy()*0
        Sr = llambs.copy()*0
        Str = llambs.copy()*0
        etalonBool = [False, True, True]
        
        backgroundw = [[0.788,  2*0.788],[0.788,  2*0.788]]
        background = [[0.788,  2*0.788],[0.788,  2*0.788]]*0
        
        
        wBack = np.min( wws*1e-12/(2*np.pi))
        backgroundw = np.array([[2*wBack,  2*wBack],[wBack, wBack]])
        backgroundthe = np.array([[-10, 10],[-10, 10]])
        background = np.array([[0.788,  2*0.788],[0.788,  2*0.788]])*0
        
        
        for i in range(3):
            for j in range(M):
                St[:,j] = np.nan_to_num(SPDCResonator.calcSpectrumSPDCTrans(lambs, lambp, theta = theta[j], L = L, s= s[i], chi = chi, etalonBool = etalonBool[i]))
                # p1 = axArr[i,0].pcolormesh(backgroundw, background, St/np.max(St), cmap='plasma')
        
                axArr[i,0].pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
                p1 = axArr[i,0].pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), St/np.max(St), cmap='plasma')
                
                FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))
                if etalonBool[i] is True:
                    TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llambs, lambp, L, theta = np.abs(ttheta), interferenceBool=True)
                    TT = np.nan_to_num(TT)
                    FAS = FAS*TT
    
                axArr[i,1].pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
                p1 = axArr[i,1].pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), FAS/np.max(FAS), cmap='plasma')
            if i == 1:
                axg0.pcolormesh(llambs*1e6, np.degrees(ttheta),St/np.max(St), cmap='plasma', alpha = 0)
                axg1.pcolormesh(llambs*1e6, np.degrees(ttheta),FAS/np.max(FAS), cmap='plasma', alpha = 0)

        for axRow in axArr:
            for ax in axRow:
                ax.set_xticks([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
                ax.set_xticklabels([r'$\frac{\omega_p}{4\pi}$',  r'$\frac{\omega_p}{2\pi}$'])
                ax.set_yticks([-10,10])
                ax.set_xlim([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
                ax.set_ylim([-10,10])
            
                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                # ax.spines['left'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)
                ax.tick_params(left=False, right=False, top=False, bottom=False)
        
        # axg0.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 0)
        # axg1.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 0)

        axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        axg1.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        axg1.tick_params(left=False, right=False, top=False, bottom=False)
        axg0.tick_params(left=False, right=False, top=False, bottom=False)
        
        gs = gridspec.GridSpec(3,4,width_ratios = [1,1,1,0.15])
        
        cbar_ax = fig.add_subplot(gs[:,-1])
        
        cbar = fig.colorbar(p1, cax = cbar_ax, pad=0.05, aspect = 4)
        cbar.ax.tick_params(labelsize=numfontsize-2)
        cbar.ax.set_yticks([0,1])
        cbar.set_label(r'', rotation = 90, fontsize = labelfontsize, labelpad = -3)
        axArr[2,2].set_xticklabels([])
        
        axg0.set_xticks([0.788,  2*0.788])
        axg0.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
        axg1.set_xticks([0.788,  2*0.788])
        axg1.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
    
        if saveBool is True:        
            dateTimeObj = datetime.now()
            preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
    
            plt.savefig(
                "figures/" + preamble + "_SPDCSpectra" + ".png", dpi=500, bbox_inches="tight",transparent=True
            )
            
def plotFASRSquared(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10,s = 1, chi = 1e-10,saveBool=False, etalonBool = False):
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        # theta = np.linspace(-thetaRange/2, thetaRange/2,M)
        
        # llambs,ttheta = np.meshgrid(lambs, theta, indexing='ij');
        
        wp = optics.lamb2omega(lambp)
        ws = optics.lamb2omega(lambs)
        
        sMin = 1
        sMax = 0.01/chi
        
        s = np.linspace(sMin, sMax, M)

        fig, ax = plot_custom(PRQuantumWidthFull*0.3, 0.2*PRQuantumWidthFull, r"gain, $\gamma$", r"$R^2$",#$\chi^{(2)}E_0$[r"$\omega_s/2\pi$ (THz)",r"$\omega_s/2\pi$ (THz)", ''],[r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)"] #"coincidence rate (mHz/mW)"
           labelPadX = 3, labelPadY = 4, commonY = True, commonX = True, wSpaceSetting=0.0, hSpaceSetting = 0.0,
           spineColor = 'black', tickColor = 'black', textColor = 'black',
           nrow = 1, ncol = 1, fontType = 'sanSerif', axefont = labelfontsize-1, numsize=numfontsize-2,
        )
        
        # St = lambs.copy()*0
        
        rSquared = s*0
        gain = s*0
        e0 = 8.84e-12
        c = 2.99e8
        for j in range(M):
            
            St = np.nan_to_num(SPDCResonator.calcSpectrumSPDCTrans(lambs, lambp, theta = 0, L = L, s= s[j], chi = chi, etalonBool=True))
            gain[j] = max(St)/(1/2*e0*c*(s[j])**2)
            St = St/np.max(St)
            
            FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, ws, 0, L))

            TT, RR, TRRT = SPDCResonator.calcEtalonEffect(lambs, lambp, L, theta = 0, interferenceBool=True)
            TT = np.nan_to_num(TT)
            FAS = FAS*TT
            FAS = FAS/np.max(FAS)
            
            # ax.plot(lambs, St, 'r')
            # ax.plot(lambs, FAS, '--k')
            
            rSquared[j] = r2_score(St, FAS)
            
            # rSquared[j] = np.sum(np.abs(St - FAS)/St)
            
        ax.plot((gain)**1, rSquared, color = colorPurple, lw=lw)
        plt.xscale('log')
        
        min_x_value = 1e-14
        max_x_value = 5e-13
        min_y_value = -1.5
        max_y_value = 1.04
        
        rangex = max_x_value - min_x_value
        rangey = max_y_value - min_y_value
        
        limScale = 10
        limScaley = 0.15
        limScaleMin = rangey*limScaley
        limScaleMax = rangey*limScaley
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
        ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')
        
        ax.spines['left'].set_bounds(min_y_value,max_y_value)
        
        
        # ax.set_xticks(filtered_xticks)
        ax.set_xticks([1e-14, 1e-13])
        # ax.set_xticklabels([r'$2\lambda_{\text{pump}}$'])
        ax.set_yticks([-1, 0, 1])
        
        ax.set_ylim([min_y_value - limScaleMin, max_y_value])
        ax.set_xlim([0.91e-14, max_x_value])
        
        ax.tick_params(axis='x', pad=8)  # Padding for x-axis tick labels
        ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
        
        if saveBool is True:        
            dateTimeObj = datetime.now()
            preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
    
            plt.savefig(
                "figures/" + preamble + "_RSquared" + ".svg", dpi=400, bbox_inches="tight",transparent=True
            )
            
def plotGainPumpPower(saveBool = False,lambP = 0.788e-6, L = 10.145e-6,chi =34e-12, smin = 0, smax = 6e8):
    smin = 0.001/chi
    smax = 0.0525/chi
    fig, ax = plot_custom(
       0.24*PRQuantumWidthFull, 0.18*PRQuantumWidthFull, r"$4\beta/\Delta$",
       r"gain, $\gamma$", labelPadX = 4, labelPadY = 8, commonY = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=labelfontsize,
       widthBool = True,
       widthRatios = [1], heightRatios=[1]
    )
    
    numPoints = 100
    s = np.linspace(smin, smax, numPoints)
    s= np.logspace(np.log10(smin), np.log10(smax), numPoints)
    
    gains = np.zeros(numPoints)
    betas = np.zeros(numPoints)
    deltas = np.zeros(numPoints)
    
    llamb = lambP*2#np.linspace(lambMin, lambMax, numPoints)
    
    for i in range(numPoints):
        Ufunc, gainfuncp, betafuncP, deltafunc= SPDCResonator.calcInteractionMatrix(lambP, L = L, chi = chi, s =s[i] , gainBool = True)
        U = lambda lamb: Ufunc(lamb)
        betaP = lambda lamb: betafuncP(lamb)
        gainP = lambda lamb: gainfuncp(lamb)
        gain = lambda lamb: np.real(gainP(lamb))
        delta = lambda lamb: deltafunc(lamb)
        
        gains[i] = np.max(gain(llamb))
        ind = gain(llamb).argmax()
        betas[i] = np.real(betaP(llamb))#[ind]
        deltas[i] = np.real(delta(llamb))
    
    
    ax.plot(betas/deltas*2, gains, colorPurple, lw = lw)
    
    ax.set_xscale('log')
    # min_x_value = 0
    # max_x_value = 2
    # min_y_value = 0
    # max_y_value = 3
    # rangex = max_x_value - min_x_value
    # rangey = max_y_value - min_y_value
    
    # limScale = 0.12
    # limScaley = 0.2
    # limScaleMin = rangey*limScaley
    # limScaleMax = rangey*limScaley
       
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    # ax.tick_params(axis='x', which='both', top=False, bottom=True)


    
    # ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.xaxis.set_label_position("bottom")
    # ax.xaxis.set_ticks_position('bottom')
    
    # ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # # Get the automatically set ticks
    # xticks = plt.gca().get_xticks()
    # yticks = plt.gca().get_yticks()
    
    
    # # Filter out ticks outside the spine bounds
    # tickFilter = 1e-2
    # filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    # filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    
    
    # ax.set_xticks(filtered_xticks)
    # ax.set_xticks([0, 1, 2, 3])
    # ax.set_yticks([0, 3])
    
    # ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    # ax.set_xlim([min_x_value - rangex*limScale, max_x_value])
    
    # ax.tick_params(axis='x', pad=3)  # Padding for x-axis tick labels
    # ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_SPDCGainInset" + ".svg", dpi=200, bbox_inches="tight",transparent=True
        )
        
def plotGainPumpPowerRSquared(lambsLow, lambsHigh, saveBool = False,lambP = 0.788e-6, L = 10.145e-6,chi =34e-12, smin = 1, smax = 6e8):
    smin = 0.0006/chi
    smax = 0.0133/chi
    
    numPointsS = 40
    numPointsLamb = 1000
    lambs = np.linspace(lambsLow, lambsHigh, numPointsLamb)
    wp = optics.lamb2omega(lambP)
    ws = optics.lamb2omega(lambs)
    s = np.linspace(smin, smax, numPointsS)
    s= np.logspace(np.log10(smin), np.log10(smax), numPointsS)
    
    fig, ax = plot_custom(PRQuantumWidthFull*0.3, 0.3*PRQuantumWidthFull, r"$\beta$", [r"gain, $\operatorname{Re}\{\gamma\}$", r"$R^2$"],#$\chi^{(2)}E_0$[r"$\omega_s/2\pi$ (THz)",r"$\omega_s/2\pi$ (THz)", ''],[r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)", r" $\theta$ ($^\circ$)"] #"coincidence rate (mHz/mW)"
       labelPadX = 3, labelPadY = 8, commonY = True, commonX = True, wSpaceSetting=0.0, hSpaceSetting = 0.0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'sanSerif', axefont = labelfontsize-1, numsize=numfontsize-2,
    )
    
    rSquared = s*0
    gain = s*0
    for j in range(numPointsS):
        
        St = np.nan_to_num(SPDCResonator.calcSpectrumSPDCTrans(lambs, lambp, theta = 0, L = L, s= s[j], chi = chi, etalonBool=True))
        # gain[j] = max(St)/(1/2*e0*c*(s[j])**2)
        St = St/np.max(St)
        
        FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, ws, 0, L))

        TT, RR, TRRT = SPDCResonator.calcEtalonEffect(lambs, lambp, L, theta = 0, interferenceBool=True)
        TT = np.nan_to_num(TT)
        FAS = FAS*TT
        FAS = FAS/np.max(FAS)
        
        rSquared[j] = r2_score(St, FAS)
        
    gains = np.zeros(numPointsS)
    betas = np.zeros(numPointsS)
    deltas = np.zeros(numPointsS)
    
    llamb = lambP*2#np.linspace(lambMin, lambMax, numPoints)
    
    for i in range(numPointsS):
        Ufunc, gainfuncp, betafuncP, deltafunc= SPDCResonator.calcInteractionMatrix(lambP, L = L, chi = chi, s =s[i] , gainBool = True)
        U = lambda lamb: Ufunc(lamb)
        betaP = lambda lamb: betafuncP(lamb)
        gainP = lambda lamb: gainfuncp(lamb)
        gain = lambda lamb: np.real(gainP(lamb))
        delta = lambda lamb: deltafunc(lamb)
        
        gains[i] = np.max(gain(llamb))
        ind = gain(llamb).argmax()
        betas[i] = np.real(betaP(llamb))#[ind]
        deltas[i] = np.real(delta(llamb))
        
    
    
    ax[0].plot(betas, gains, 'k', lw = lw)
    ax[1].plot(betas, rSquared, color = 'k', lw=lw)
    ax[1].set_xscale('log')
    
    min_x_value = min(betas)
    max_x_value = max(betas)
    min_y_value = 0
    max_y_value = 1.5
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.12
    limScaley = 0.2
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    xlabelPad = 5
       
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[0].yaxis.set_ticks_position('left')
    ax[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[0].tick_params(axis='x', which='both', top=False, bottom=True)
    ax[0].xaxis.set_label_position("bottom")
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value])
    ax[0].set_xlim([min_x_value/2, max_x_value*2])
    ax[0].tick_params(axis='x', pad=xlabelPad)  # Padding for x-axis tick labels
    ax[0].tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
        
    # # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    # # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    ax[1].set_xticks(filtered_xticks)
    
    # # Get the automatically set ticks
    xticks = plt.gca().get_xticks(minor = True)
    yticks = plt.gca().get_yticks(minor = True)
    
    # # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value]
    ax[1].set_xticks(filtered_xticks, minor = True)
    
    min_y_value = min(rSquared)
    max_y_value = 1.1
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.05
    limScaley = 0.2
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
       
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[1].yaxis.set_ticks_position('left')
    ax[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[1].tick_params(axis='x', which='both', top=False, bottom=True)
    ax[1].xaxis.set_label_position("bottom")
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value])
    ax[1].set_xlim([min_x_value/1.5, max_x_value*1.5])
    ax[1].tick_params(axis='x', pad=xlabelPad)  # Padding for x-axis tick labels
    ax[1].tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
        
    # # Get the automatically set ticks
    ax[1].set_xticks([0.1, 1])
    ax[1].set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],minor = True)
    
    ax[0].set_yticks([0,0.5,1,1.5])
    ax[1].set_yticks([1, 0,-2,-4, -6])
    
    ax[0].yaxis.labelpad = 10
    ax[1].yaxis.labelpad = 5
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_SPDCGain" + ".svg", dpi=200, bbox_inches="tight",transparent=True
        )
            
def plotChangjinDataCounts(saveBool = False):
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/etalon_GaP_changjin/"
    
    fig, ax = plot_custom(
        0.5*PRQuantumWidthFull,  0.55*PRQuantumWidthFull, r"wavelength, $\lambda$ (nm)",
        [r"transmission, $T$ (au)", r'coincidence rate (mcps/mW)'], labelPadX = 4, labelPadY = 10, commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        commonX = True,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 1, nrow = 2, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize+1,
        widthBool=True, widthRatios = [1], heightRatios = [1, 1.5]
    )
    
    dataOSA = pd.read_table(dataDir + 'data_OSA.txt', delimiter=' ')
    dataOSAWL = dataOSA['wl'].to_numpy()
    dataOSATrans = dataOSA['Transmission'].to_numpy()
    
    #% plot the first plot
    min_x_value = 1160
    max_x_value = 1210
    resoX = 1184.4
    backX = resoX
    degX = 1187
    forwX = 2*degX - backX
    min_y_value = 0.2
    max_y_value = 1
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    filterArr = (dataOSAWL > min_x_value) & (dataOSAWL < max_x_value)
    
    dataOSAWL = dataOSAWL[filterArr]
    dataOSATrans = dataOSATrans[filterArr]
    
    limScale = 0.08
    limScaley = 0.08
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax[0].set_xticks([min_x_value, resoX, max_x_value])
       
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].yaxis.set_ticks_position('left')
    ax[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[0].tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax[0].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[0].xaxis.set_label_position("bottom")
    ax[0].xaxis.set_ticks_position('bottom')
    
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    
    ax[0].set_yticks([1, 0.6, 0.2])
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax[0].set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
           
    #% second plot
    dataBulk = pd.read_table(dataDir + 'SPDC spectrum_FB/GaP_layer_FB_spectrum.txt', delimiter=' ')
    dataBulkWL = dataBulk['#'].to_numpy()
    dataBulkTrans = dataBulk['[wl]'].to_numpy()
    
    dataMS = pd.read_table(dataDir + 'SPDC spectrum_FB/GaP_MS_FB_spectrum.txt', delimiter=' ')
    dataMSTransWL = dataMS['[R_error]'].to_numpy()
    dataMSTrans = dataMS['[T_wl]'].to_numpy()
    dataMSTransError = dataMS['[T_spectrum]'].to_numpy()
    dataMSReflWL = dataMS['#'].to_numpy()
    dataMSRefl = dataMS['[R_wl]'].to_numpy()
    dataMSReflError = dataMS['[R_spectrum]'].to_numpy()
    
    min_y_value = 0
    max_y_value = 0.4
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    scale = 1e3
    
    ax[0].plot(dataOSAWL, dataOSATrans, color = colorPurple, lw = lw)
    ax[0].plot([forwX, forwX], [0, 0.84], '--',lw = lw, color = colorBlue, zorder = -1)
    ax[1].plot([forwX, forwX], [max(dataMSTrans*scale), 1], '--',lw = lw, color = colorBlue, zorder = -1)
    # ax[0].plot([degX, degX], [0, 1], '--',lw = lw, color = 'k', zorder = -1)
    # ax[1].plot([degX, degX], [-1, 1], '--',lw = lw, color = 'k', zorder = -1)
    ax[0].plot([backX, backX], [0, 1], '--',lw = lw, color = colorRed, zorder = -1)
    ax[1].plot([backX, backX], [max(dataMSRefl*scale), 1], '--',lw = lw, color = colorRed, zorder = -1)
    
    
    filterArr = (dataBulkWL > min_x_value) & (dataBulkWL < max_x_value)
    
    dataBulkWL = dataBulkWL[filterArr]
    dataBulkTrans = dataBulkTrans[filterArr]
    
    filterArr = (dataMSTransWL > min_x_value) & (dataMSTransWL < max_x_value)
    
    dataMSTransWL = dataMSTransWL[filterArr]
    dataMSTransError = dataMSTransError[filterArr]
    dataMSTrans = dataMSTrans[filterArr]
    
    filterArr = (dataMSReflWL > min_x_value) & (dataMSReflWL < max_x_value)
    
    dataMSReflWL = dataMSReflWL[filterArr]
    dataMSRefl = dataMSRefl[filterArr]
    dataMSReflError = dataMSReflError[filterArr]
    
    ax[1].plot(dataBulkWL, dataBulkTrans*66*scale, color = 'grey', lw = lw)
    ax[1].plot(dataMSTransWL, dataMSTrans*scale, color = colorBlue, lw = lw)
    ax[1].plot(dataMSReflWL, dataMSRefl*scale, color = colorRed, lw = lw)
    ax[1].scatter(dataBulkWL, dataBulkTrans*66*scale, color = 'grey', s = 2)
    ax[1].scatter(dataMSTransWL, dataMSTrans*scale, color = colorBlue, s = 2)
    ax[1].scatter(dataMSReflWL, dataMSRefl*scale, color = colorRed, s = 2)
    
    
    ax[1].fill_between(dataBulkWL, dataBulkTrans*0, dataBulkTrans*66*scale, color = 'grey', alpha = 0.2)    
    ax[1].fill_between(dataMSTransWL, (dataMSTrans - dataMSTransError)*scale, 
                       (dataMSTrans + dataMSTransError)*scale, color = colorBlue, alpha = 0.3)
    ax[1].fill_between(dataMSReflWL, (dataMSRefl - dataMSTransError)*scale, 
                       (dataMSRefl + dataMSTransError)*scale, color = colorRed, alpha = 0.3)

    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax[1].errorbar(dataMSWL, dataMSTrans*scale, dataMSTransError*scale, marker = '.', 
    #                linestyle = 'None', color = colorBlue, zorder = 10, ms = 5, capsize=1, capthick=1, elinewidth  = 1)
    # ax[1].errorbar(dataMSWL, dataMSRefl*scale, dataMSReflError*scale, marker = '.', 
    #                linestyle = 'None', color = colorRed, zorder = 10, ms = 5, capsize=1, capthick=1, elinewidth  = 1)
    # ax[0].plot(dataOSAWL, dataOSATrans, color = colorPurple, lw = lw)
    
    limScale = 0.05
    limScaley = 0.2
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax[1].set_xticks([min_x_value, resoX, max_x_value])
       
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].yaxis.set_ticks_position('left')
    ax[1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax[1].tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax[1].spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax[1].xaxis.set_label_position("bottom")
    ax[1].xaxis.set_ticks_position('bottom')
    
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    
    # ax[0].set_yticks([0, 0.05, 0.1])
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax[1].set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_metasurface_counts" + ".svg", dpi=400, bbox_inches="tight",transparent=True
        )
    
def plotMSCountrate(saveBool = False):
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/etalon_GaP_changjin/zero delay/"
    
    fig, ax = plot_custom(
        0.75*PRQuantumWidthFull,  0.28*PRQuantumWidthFull, [r"time delay, $\tau$ (ns)", r"time delay, $\tau$ (ns)", r"time delay, $\tau$ (ns)"],
        [r'coincidence rate (cps)', '', ''], labelPadX = 4, labelPadY = 4, 
        commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 3, nrow = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize+1,

    )
    
    FZero = 4
    BZero = 4
    FBZero = 7.5
    
    dataF = pd.read_table(dataDir + 'GaP_MS/forward/GaP20deg_nodelay_t1-t6_old-new_LP850nm_LP1000nm_BP1200nm-atinput_IR-IR_Bias15p5both_70mW_Coincidences.txt', delimiter='	')
    dataFTime = dataF.index.to_numpy()/1e3 - FZero
    dataFCounts = dataF['Time(ps)'].to_numpy()
    
    dataB = pd.read_table(dataDir + 'GaP_MS/backward/GaP20deg_nodelay_t1-t6_old-new_LP850nm_BP1200nm-atinput_IR-IR_Bias15p5both_Coincidences.txt', delimiter='	')
    dataBTime = dataB.index.to_numpy()/1e3 - BZero
    dataBCounts = dataB['Time(ps)'].to_numpy()
    
    dataFB = pd.read_table(dataDir + 'GaP_MS/forward backward/GaP20deg_both direction_zero delay_BP1200nm_LP850nm at R_LP1000nm_LP850nm_BP1200nm at T_Coincidences.txt', delimiter='	')
    dataFBTime = dataFB.index.to_numpy()/1e3 - FBZero
    dataFBCounts = dataFB['Time(ps)'].to_numpy()
    
    
    min_x_value = -4
    max_x_value = 4
    rangex = max_x_value - min_x_value
    
        
    filterArr = (dataFTime > min_x_value) & (dataFTime < max_x_value)
    dataFTime = dataFTime[filterArr]
    dataFCounts = dataFCounts[filterArr]
    dataFErr = np.sqrt(dataFCounts)

    filterArr = (dataBTime > min_x_value) & (dataBTime < max_x_value)
    dataBTime = dataBTime[filterArr]
    dataBCounts = dataBCounts[filterArr]
    dataBErr = np.sqrt(dataBCounts)
    
    filterArr = (dataFBTime > min_x_value) & (dataFBTime < max_x_value)
    dataFBTime = dataFBTime[filterArr]
    dataFBCounts = dataFBCounts[filterArr]
    dataFBErr = np.sqrt(dataFBCounts)
    
    scaleF = 3614
    scaleB = 1633
    scaleFB = 1194
    
    ax[0].plot(dataFTime, dataFCounts/scaleF, color = colorBlue, lw = lw)
    ax[1].plot(dataBTime, dataBCounts/scaleB, color = colorRed, lw = lw)
    ax[2].plot(dataFBTime, dataFBCounts/scaleFB, color = colorPurple, lw = lw)
    ax[0].scatter(dataFTime, dataFCounts/scaleF, color = colorBlue, s = 2)
    ax[1].scatter(dataBTime, dataBCounts/scaleB, color = colorRed, s = 2)
    ax[2].scatter(dataFBTime, dataFBCounts/scaleFB, color = colorPurple, s = 2)
    
    ax[0].fill_between(dataFTime, (dataFCounts-dataFErr)/scaleF, (dataFCounts+dataFErr)/scaleF, color = colorBlue, alpha = 0.3)    
    ax[1].fill_between(dataBTime, (dataBCounts-dataBErr)/scaleB, (dataBCounts+dataBErr)/scaleB, color = colorRed, alpha = 0.3)    
    ax[2].fill_between(dataFBTime, (dataFBCounts-dataFBErr)/scaleFB, (dataFBCounts+dataFBErr)/scaleFB, color = colorPurple, alpha = 0.3)    
    
    
    limScale = 0.11
    
    for axx in ax:
        axx.set_ylim([0, None])
        axx.set_xticks([min_x_value, 0, max_x_value])
        
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)    
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=True)    
        axx.spines['bottom'].set_bounds(min_x_value,max_x_value)
        axx.xaxis.set_label_position("bottom")
        axx.xaxis.set_ticks_position('bottom')
        axx.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
        
        fig.subplots_adjust(wspace=0.2)
    
    

    min_y_value = 0
    max_y_value = 0.03
    limScaley = 0.11
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley    
    ax[0].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[0].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 0.12
    ax[1].set_yticks([0, 0.04, 0.08, 0.12])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[1].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[1].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    
    min_y_value = 0
    max_y_value = 0.24
    ax[2].set_yticks([0, 0.08, 0.16, 0.24])
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    ax[2].spines['left'].set_bounds(min_y_value,max_y_value)
    ax[2].set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
   
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_metasurface_Countrate" + ".svg", dpi=400, bbox_inches="tight",transparent=True
        )
    
def plotProbEmissionsMS(saveBool = False):
    fig, ax = plot_custom(
        0.25*PRQuantumWidthFull,  0.28*PRQuantumWidthFull, r'emission schemes',
        r'emission probability (\%)', labelPadX = 4, labelPadY = 7, 
        commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 1, nrow = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize+1,

    )
    
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    # plot bar charts
    labels = ['F', 'B', 'FB']
    theProb = [12, 32, 56]
    expProb = [9, 35, 56]
    expProbErr = [2, 5, 6]
    x = np.arange(len(labels))+1  # the label locations
    width = 0.8  # the width of the bars
    rectsmaxs = ax.bar(1, expProb[0], width, label='forward', color = colorBlue)
    rectsmeans = ax.bar(2, expProb[1], width, label='backward', color = colorRed)
    rectsmins = ax.bar(3, expProb[2], width, label='forward/backward', color = colorPurple)
    capsize = 6
    offset = 0.0
    ax.errorbar(x[0]-offset,expProb[0],expProbErr[0],marker = '.', 
                    linestyle = 'None', color = 'k', zorder = 10, ms = 0, capsize=capsize, capthick=1, elinewidth  = 1)
    ax.errorbar(x[1]-offset,expProb[1],expProbErr[1],marker = '.', 
                    linestyle = 'None', color = 'k', zorder = 10, ms = 0, capsize=capsize, capthick=1, elinewidth  = 1)
    ax.errorbar(x[2]-offset,expProb[2],expProbErr[2],marker = '.', 
                    linestyle = 'None', color = 'k', zorder = 10, ms = 0, capsize=capsize, capthick=1, elinewidth  = 1)
    xFill = [-10, 10]
    yFillMin = np.array([0, 0])+0.5
    ax.fill_between(xFill, yFillMin, np.ones(2)*theProb[0], color = colorBlueLight, alpha = 1, zorder = -1)  
    ax.fill_between(xFill, yFillMin, np.ones(2)*theProb[1], color = colorRedLight, alpha = 1, zorder = -2)  
    ax.fill_between(xFill, yFillMin, np.ones(2)*theProb[2], color = colorPurpleLight, alpha = 1, zorder = -3)   
    
    
    limScale = 0.11    
    min_x_value = 0.5
    max_x_value = 3.5
    rangex = max_x_value - min_x_value
    ax.set_xticks([min_x_value, 0, max_x_value])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
    min_y_value = 0
    max_y_value = 62
    limScaley = 0.01
    rangey = max_y_value - min_y_value
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    ax.set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    

    ax.set_yticks(np.append([0], theProb))
    ax.set_xticks(x)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', bottom = False,top=False, labeltop=False)
    ax.set_xticklabels(labels)
    
    
    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_metasurface_Probs" + ".svg", dpi=400, bbox_inches="tight",transparent=True
        )
        
def plotChangjinDataCountsFFBB(saveBool = False):
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/etalon_GaP_changjin_otherData/"
    
    fig, ax = plot_custom(
        0.5*PRQuantumWidthFull,  0.3*PRQuantumWidthFull, r"wavelength, $\lambda$ (nm)",
        [r"transmission, $T$ (au)", r'coincidence rate (mcps/mW)'], labelPadX = 4, labelPadY = 10, commonY = False, wSpaceSetting=0, hSpaceSetting=0,
        commonX = True,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        ncol = 1, nrow = 1, fontType = 'sanSerif', axefont = labelfontsize, numsize=numfontsize+1,
    )
    
    
    #% plot the first plot
    min_x_value = 1160
    max_x_value = 1210
    resoX = 1184.4
    backX = resoX
    degX = 1187
    forwX = 2*degX - backX
    min_y_value = 0
    max_y_value = 0.02
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    # filterArr = (dataOSAWL > min_x_value) & (dataOSAWL < max_x_value)
    
    # dataOSAWL = dataOSAWL[filterArr]
    # dataOSATrans = dataOSATrans[filterArr]
    
    limScale = 0.08
    limScaley = 0.08
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax.set_xticks([min_x_value, resoX, max_x_value])
       
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    
    # Get the automatically set ticks
    xticks = plt.gca().get_xticks()
    yticks = plt.gca().get_yticks()
    # 
    
    # Filter out ticks outside the spine bounds
    tickFilter = 1e-2
    # filtered_xticks = [tick for tick in xticks if min_x_value- rangex*tickFilter <= tick <= max_x_value+ rangey*tickFilter]
    # filtered_yticks = [tick for tick in yticks if min_y_value - rangey*tickFilter  <= tick <= max_y_value+ rangey*tickFilter]
    
    
    # ax.set_yticks([1, 0.6, 0.2])
    # ax.set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMax])
    ax.set_xlim([min_x_value - rangex*limScale, max_x_value+ rangex*limScale])
           
    dataMSBB = pd.read_table(dataDir + 'GaP_MS_BB_spectrum.txt', delimiter=' ')
    dataMSBBWL = np.array(dataMSBB.index)
    dataMSBBTrans = dataMSBB['[wl]'].to_numpy()
    dataMSBBTransError = dataMSBB['[error]'].to_numpy()

    dataMSFF = pd.read_table(dataDir + 'GaP_MS_FF_spectrum.txt', delimiter=' ')
    dataMSFFWL = np.array(dataMSFF.index)
    dataMSFFTrans = dataMSFF['[wl]'].to_numpy()
    dataMSFFTransError = dataMSFF['[error]'].to_numpy()
    
    # min_y_value = 0
    # max_y_value = 0.4
    # rangex = max_x_value - min_x_value
    # rangey = max_y_value - min_y_value
    
    # scale = 1e3
    
    ax.plot(dataMSFFWL, dataMSFFTrans/max(dataMSFFTrans), color = colorBlue, lw = lw)
    ax.plot(dataMSBBWL, dataMSBBTrans/max(dataMSBBTrans), color = colorRed, lw = lw)
    # ax.plot([forwX, forwX], [0, 0.84], '--',lw = lw, color = colorBlue, zorder = -1)

    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/etalon_GaP_changjin/"
    

    dataBulk = pd.read_table(dataDir + 'SPDC spectrum_FB/GaP_layer_FB_spectrum.txt', delimiter=' ')
    dataBulkWL = dataBulk['#'].to_numpy()
    dataBulkTrans = dataBulk['[wl]'].to_numpy()
    
    dataMS = pd.read_table(dataDir + 'SPDC spectrum_FB/GaP_MS_FB_spectrum.txt', delimiter=' ')
    dataMSTransWL = dataMS['[R_error]'].to_numpy()
    dataMSTrans = dataMS['[T_wl]'].to_numpy()
    dataMSTransError = dataMS['[T_spectrum]'].to_numpy()
    dataMSReflWL = dataMS['#'].to_numpy()
    dataMSRefl = dataMS['[R_wl]'].to_numpy()
    dataMSReflError = dataMS['[R_spectrum]'].to_numpy()
    
    min_y_value = 0
    max_y_value = 0.4
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    scale = 1e3

    filterArr = (dataBulkWL > min_x_value) & (dataBulkWL < max_x_value)
    
    dataBulkWL = dataBulkWL[filterArr]
    dataBulkTrans = dataBulkTrans[filterArr]
    
    filterArr = (dataMSTransWL > min_x_value) & (dataMSTransWL < max_x_value)
    
    dataMSTransWL = dataMSTransWL[filterArr]
    dataMSTransError = dataMSTransError[filterArr]
    dataMSTrans = dataMSTrans[filterArr]
    
    filterArr = (dataMSReflWL > min_x_value) & (dataMSReflWL < max_x_value)
    
    dataMSReflWL = dataMSReflWL[filterArr]
    dataMSRefl = dataMSRefl[filterArr]
    dataMSReflError = dataMSReflError[filterArr]
    
    ax.plot(dataMSTransWL, dataMSTrans/max(dataMSTrans), color = colorBlueLight, lw = lw)
    ax.plot(dataMSReflWL, dataMSRefl/max(dataMSRefl), color = colorRedLight, lw = lw)
    
    
    dataOSA = pd.read_table(dataDir + 'data_OSA.txt', delimiter=' ')
    dataOSAWL = dataOSA['wl'].to_numpy()
    dataOSATrans = dataOSA['Transmission'].to_numpy()
    
    ax.plot(dataOSAWL, dataOSATrans/max(dataOSATrans), color = 'grey', lw = lw)
    forwX = 1187
    ax.plot([forwX, forwX], [0, 0.84], '--',lw = lw, color = colorBlue, zorder = -1)


    if saveBool is True:        
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")

        plt.savefig(
            "figures/" + preamble + "_metasurface_counts" + ".svg", dpi=400, bbox_inches="tight",transparent=True
        )

if __name__ == '__main__':
    # dataDir = "C:/Users/nicho/Documents/gitProjects/dataMPL/"
    dataDir = "C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataMPL/"
        
    # combine datasets from web and py
    combine = False
    if combine is True:
        dateTimeObj = datetime.now()
        preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        
        fileNameWeb = "20230704_0737_50mW_coincidences_zpol_lp1400_fiber_web_overnight.txt"
        fileNamePy = "20230703_1531_50mW_coincidences_zpol_lp1400_fiber.txt"
        fileNameNew = preamble + "_50mW_coincidences_zpol_lp1400_fiber_combined.txt"
        
        combineWebAndPyData(fileNameWeb, fileNamePy, dataDir, fileNameNew)
    
    
    #%% Plot the data from the GaAs, and other LN sources other than silicon    
    # plot coincidences and g2
    fileDirGaAs = 'GaAs/'
    fileDirLNFS = "LN-FS/"
    fileDirLNFe = "LN-Fe/pumpPolTomography/"
    fileGaAs = "20230727_1046_GaAs111_30mW_hPolPump_788nm_transmission_10min_newSetup_web.txt"
    fileCRGaAs = "20230727_1046_GaAs111_30mW_hPolPump_788nm_transmission_10min_newSetup_webCR.txt"
    fileLNFS = "20230801_1209_LN-FS_70mW_vPolPump_788nm_transmission_100s_highCorrRate.txt"
    fileLNFe = "20230731_1608_LN-Fe_30mW_vPolPump_788nm_transmission_56min.txt"
    filePaths = [fileDirLNFS + fileLNFS, fileDirLNFe + fileLNFe, fileDirGaAs + fileGaAs ]
    saveBool = False
    timeOffset = 0
    legendLabels = [r"LiNbO$_3$:FS", r"LiNbO$_3$:Fe", r"GaAs (111)"]

    
    # plotCorrelation(dataDir + fileDirGaAs + fileGaAs, saveBool = saveBool, log = True, 
    #                 CRFilePath = dataDir + fileDirGaAs + fileCRGaAs, CRFileBool = True,  
    #                 title = "", fileName = fileLNFS, fitg2 = False,
    #                 exposureTime = 10*60, timeOffset = timeOffset, PpumpmW = 30)
    # plotCorrelation(dataDir + fileDirLNFe + fileLNFe, saveBool = saveBool, log = True, 
    #                 CRFilePath = dataDir + fileDirGaAs + fileCRGaAs, CRFileBool = False,  
    #                 title = "", fileName = fileLNFS, fitg2 = False,
    #                 exposureTime = 56*60, timeOffset = timeOffset, PpumpmW = 30)
    # plotCorrelation(dataDir + fileDirLNFS + fileLNFS, saveBool = saveBool, log = True, 
    #                 CRFilePath = dataDir + fileDirGaAs + fileCRGaAs, CRFileBool = False,  
    #                 title = "", fileName = fileLNFS, fitg2 = False,
    #                 exposureTime = 100, timeOffset = timeOffset, PpumpmW = 70)
    # plotCorrelations(filePaths = filePaths, saveBool = saveBool, legBool = True, logBool = True,
    #                   title = "",#r"Comparison of pair detection rates between" + "\n" + r"10 um LiNbO$_3$ and 300 um iron-doped LiNbO$_3$", 
    #                   exposureTimeArr = [100,60*56, 10*60], laserPowerArr=[70, 30, 30], 
    #                   timeOffsetArr = [0,0,-0.55], legendLabels = legendLabels, figName = "LNDopedVSUndoped")
    # plotNormalizedCorrelation(dataDir + file, saveBool = saveBool)
    # plotg2(dataDir + file, saveBool= saveBool,fitg2 = False, title = "Coincidences in transmission with a 9 mW pump")
    
    
    #%% Plot power dispersion of g2(0) for Fe samples
    #
    fileDir = 'LN-Fe/'
    filePathsPowerRefl = [
                fileDir + "pumpPowerVar/20230801_1001_LN-Fe_70mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1004_LN-Fe_50mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1006_LN-Fe_30mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1014_LN-Fe_20mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1016_LN-Fe_15mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1134_LN-Fe_10mW_vPolPump_788nm_transmission_100s.txt",
                fileDir + "pumpPowerVar/20230801_1132_LN-Fe_5mW_vPolPump_788nm_transmission_540s.txt",
                ]
    powersRefl = np.array([70, 50,30,20,15,10, 5]) # uncertainty of 5%
    powersReflDel = 0.05*powersRefl
    

    
    # plotg2Powers(dataDir, filePathsPowerRefl, powersRefl, powersReflDel, saveBool = saveBool, 
    #               fileName = "_LN-Fe", fitg2 = True)
    
    
    # %% Plot fiber spectroscopy coincidences for nonSi-mounted LN together
    # plot fiber dispersion
    saveBool = False
    fileDir = 'GaAs/'
    filepaths = [
        # fileDir + "photolumDecayBurningFiberSpec/20230802_0006_GaAs111_30mW_vPolPump_788nm_transmission_fiber_1400nm_burning.txt",
        # "LN-FS/" + "fiberSpectroscopy/20230801_1326_LN-FS_70mW_vPolPump_788nm_transmission_fiber_1400nm.txt",
        # "LN-Fe/" + "fiberSpectroscopy/20230801_2210_LN-Fe_70mW_vPolPump_788nm_transmission_fiber_1400nm.txt",
        'GaAs/' + "photolumDecayBurningFiberSpec/20230802_1003_GaAs111_10mW_vPolPump_788nm_transmission_fiber_1400nm_expdecay.txt",
        # "LN-Si/" + "fiberSpectroscopy/reflection/20230712_1614_50mW_coincidences_zpol_lp1500_fiber_combined.txt",
        # "LN-Si/" + "fiberSpectroscopy/reflection/20230712_1616_50mW_coincidences_zpol_lp1400_fiber_combined.txt",
        # "LN-Si/" + "fiberSpectroscopy/frontback/20230713_0104_50mW_coincidences_zpol_lp1400_transmission_reflection_fiber_web.txt",
        "LN-Si/" + "fiberSpectroscopy/reflection/20230712_1616_50mW_coincidences_zpol_lp1400_fiber_combined.txt",
        "LN-Si/" + "fiberSpectroscopy/transmission/20230707_1625_50mW_coincidences_zpol_lp1400_transmission_fiber.txt",
        "LN-Si/" + "fiberSpectroscopy/frontBack/20230713_0104_50mW_coincidences_zpol_lp1400_transmission_reflection_fiber_web.txt",
        
        ]
    legendLabels = [
        "LN-FS",
        r"LN-Fe ($\times 10$)",
        r"GaAs (111) ($\times 10$)" ,
        r"GaAs (111) ($\times 10$)"                         
                    ]
    
    #
    # plotFiberCoincidences(dataDir, filepaths, saveBool = saveBool, 
    #                       avgBool = True, avgSample = 4, normBool = True, offsetBool = False, 
    #                       figName = "Fiber spectroscopy comparison: 788 nm pump, 3 km SMF-28", 
    #                       legendBool = True, logBool = False,
    #                       legendLabels = legendLabels, cutoffBool = False, plotDelayBool=True)

    # envelopeFunctionDetectionEff()
    # plotEnvelopeFunction(saveBool = False)

    # %% Plot fiber spectroscopy coincidences for Si-mounted LN separately
    # plot fiber dispersion separately
    saveBool = False
    fileDir = 'LN-Si/'
    filepaths = [
        fileDir + "fiberSpectroscopy/reflection/20230712_1616_50mW_coincidences_zpol_lp1400_fiber_combined.txt",
        # fileDir + "fiberSpectroscopy/reflection/20230712_1614_50mW_coincidences_zpol_lp1500_fiber_combined.txt",
        fileDir + "fiberSpectroscopy/transmission/20230707_1625_50mW_coincidences_zpol_lp1400_transmission_fiber.txt",
        # fileDir + "fiberSpectroscopy/transmission/20230707_1800_50mW_coincidences_zpol_lp1500_transmission_fiber.txt",
        fileDir + "fiberSpectroscopy/frontBack/20230713_0104_50mW_coincidences_zpol_lp1400_transmission_reflection_fiber_web.txt",
        # fileDir + "fiberSpectroscopy/frontBack/20230712_0833_50mW_coincidences_zpol_lp1500_transmission_reflection_fiber_web.txt",
        
        ]

    expTime1 = 60*(33 + 60*(8 + 10))
    expTime2 = 60*(33 + 60*(8 + 10))
    exposureTimeArr = np.array([None, None, None, None, expTime1, expTime2])
    laserPowerArr = [50, 50, 50, 50, 50, 50]
    rate1Arr = [None, None, None, None, 0, 0]
    rate2Arr = [None, None, None, None, 0, 0]

    exposureTimeArr = np.array([None, None, expTime1])
    laserPowerArr = [50, 50, 50]
    rate1Arr = [None, None, 0]
    rate2Arr = [None, None, 0]
    delayDoubleBool = [True, True, False]
    
    # plotFiberCoincidencesSeparate(dataDir, filepaths, saveBool = saveBool, 
    #                       avgBool = True, avgSample = 1, normBool = True, 
    #                       figName = "Fiber spectroscopy comparison: 788 nm pump, 3 km SMF-28", 
    #                       logBool = False, plotDelayBool = True, delayDoubleBool = delayDoubleBool,
    #                       exposureTimeArr=exposureTimeArr, laserPowerArr=laserPowerArr,
    #                       rate1Arr = rate1Arr, rate2Arr = rate2Arr, )
    
    # plotFiberCoincidencesSeparateWithModel(dataDir, filepaths, saveBool = saveBool, 
    #                       avgBool = True, avgSample = 1, normBool = True, 
    #                       figName = "Fiber spectroscopy comparison: 788 nm pump, 3 km SMF-28", 
    #                       logBool = False, plotDelayBool = False, delayDoubleBool = delayDoubleBool,
    #                       exposureTimeArr=exposureTimeArr, laserPowerArr=laserPowerArr,
    #                       rate1Arr = rate1Arr, rate2Arr = rate2Arr, )

    

    
    # %% plot the polarization tomography for the Si cells
    
    del(filepaths)
    filePathExtra = 'LN-Si/pumpPolTomography/reflection/'
    filepaths = [
        filePathExtra+ '20230621_1700_30mW_coincidences_zpol_HWP0_hpolGlan.txt', # type 0 z pol
        filePathExtra+ '20230621_1755_30mW_coincidences_zpol_HWP45_hpolGlan.txt',# type 1 z pol
        filePathExtra+ '20230622_1000_30mW_coincidences_ypol_HWP0_hpolGlan.txt', # type 1 y pol
        filePathExtra+ '20230622_1159_30mW_coincidences_ypol_HWP45_hpolGlan.txt'  # type 0 y pol
        ]
    
    # for file in filepaths:
    #     plotg2(dataDir + file, saveBool=saveBool)
    
    # plotPolTomography(filepaths, saveBool = saveBool)
    
    
    # %% plot transmission measurements for Si
    
    # plot coincidences and g2
    filePathExtra = 'LN-Si/pumpPowerVar/transmission/'
    file = filePathExtra+"20230705_1053_9mW_coincidences_zpol_transmission.txt"
    
    saveBool = False
    # plotCorrelation(dataDir + file, saveBool = saveBool)
    # plotNormalizedCorrelation(dataDir + file, saveBool = saveBool)
    # plotg2(dataDir + file, saveBool= saveBool, figName = "_transmission")
    
    # plot power dispersion of g2(0)
    
    filePathsPowerTrans = [
        filePathExtra+'20230705_1053_9mW_coincidences_zpol_transmission.txt',
        filePathExtra+'20230705_1102_18mW_coincidences_zpol_transmission.txt',
        filePathExtra+'20230705_1108_37mW_coincidences_zpol_transmission.txt',
        filePathExtra+'20230705_1130_55mW_coincidences_zpol_transmission_web.txt',
        filePathExtra+'20230705_1135_82mW_coincidences_zpol_transmission_web.txt',
        ]
    
    powersTrans = np.array([9.47, 18.16, 37.44, 55.37, 81.55]) # uncertainty of 5%
    powersTransDel = 0.05*powersTrans
    
    # plotg2Power(dataDir, filePathsPowerTrans, powersTrans, powersTransDel, saveBool = saveBool, figName = "_transmission")
    
    # %% plot the polarization tomography for Si
    
    del(filepaths)
    filepaths = [
        'LN-Si/pumpPolTomography/transmission/' + '20230705_1420_55mW_coincidences_zpol_hpolGlan_transmission_web.txt', # type 0 z pol
        'LN-Si/pumpPolTomography/transmission/' + '20230705_1431_55mW_coincidences_zpol_vpolGlan_transmission_web.txt',# type 1 z pol
        'LN-Si/pumpPolTomography/transmission/' + '20230705_1455_55mW_coincidences_ypol_hpolGlan_transmission_web.txt', # type 1 y pol
        'LN-Si/pumpPolTomography/transmission/' + '20230705_1532_55mW_coincidences_ypol_vpolGlan_transmission_web.txt'  # type 0 y pol
        ]
    
    # for file in filepaths:
    #     plotg2(dataDir + file, saveBool=saveBool)
    
    # plotPolTomography(filepaths, saveBool = saveBool, figName = "_transmission")
    
    
    
    
    # %% plot transmission/reflection measurements for Si
    
    # plot coincidences and g2
    
    file = "20230705_1640_9mW_coincidences_zpol_transmission_reflection_web.txt"
    
    saveBool = False
    # plotCorrelation(dataDir + file, saveBool = saveBool)
    # plotNormalizedCorrelation(dataDir + file, saveBool = saveBool)
    # plotg2(dataDir + file, saveBool= saveBool, figName = "_transmissionReflection")
    
    # %% plot power dispersion of g2(0) 
    filePathExtraFB = 'LN-Si/pumpPowerVar/frontBack/'
    filePathsPowerTransRefl = [
        filePathExtraFB+'20230705_1640_9mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1635_18mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1628_37mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1613_55mW_coincidences_zpol_transmission_reflection_web.txt',
        filePathExtraFB+'20230705_1618_82mW_coincidences_zpol_transmission_reflection_web.txt',
        ]
    
    filePathExtraR = 'LN-Si/pumpPowerVar/reflection/'    
    filePathsPowerRefl = [
        filePathExtraR+'20230622_1255_5mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1305_10mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1312_20mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1319_30mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1323_50mW_coincidences_zpol.txt',
        filePathExtraR+'20230622_1328_70mW_coincidences_zpol.txt',
        ]
    
    filePathExtraT = 'LN-Si/pumpPowerVar/transmission/'
    filePathsPowerTrans = [
        filePathExtraT+'20230705_1053_9mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1102_18mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1108_37mW_coincidences_zpol_transmission.txt',
        filePathExtraT+'20230705_1130_55mW_coincidences_zpol_transmission_web.txt',
        filePathExtraT+'20230705_1135_82mW_coincidences_zpol_transmission_web.txt',
        ]
    
    powersRefl = np.array([5, 10, 20, 30, 50, 70])
    powersReflDel = 0.05*powersRefl # uncertainty of 5%
    powersTransRefl = np.array([9.47, 18.16, 37.44, 55.37, 81.55]) # uncertainty of 5%
    powersTransReflDel = 0.05*powersTransRefl
    powersTrans = powersTransRefl
    powersTransDel = powersTransReflDel
    
    # plotg2Power(dataDir, filePathsPowerTransRefl, powersTransRefl, powersTransReflDel, saveBool = saveBool, figName = "_transmissionReflection")
    
    # plot combined plots
    
    saveBool = False
    
    filePathsArr = [filePathsPowerTrans, filePathsPowerRefl,  filePathsPowerTransRefl]
    powersArr = [ powersTrans,powersRefl, powersTransRefl]
    powersDel = [powersTransDel,powersReflDel,  powersTransReflDel]
    
    # ax, fig = plotg2Powers(dataDir, filePathsArr, powersArr, powersDel, saveBool = saveBool, fitg2 = False,
    #                         figName = "_combined", legendBool=True, legendLabels=["transmission", "reflection", "trans./refl."])

    # %% polarization tomography measurements
    # new analysis
    saveBool = False
    path = 'GaAs/'
    fileName = "pairPolTomography/20230724_2259_GaAs111_50mW_horGlan_vPolPump_788nm_transmission.csv"
    title = (r"Measured $g^{(2)}(0)$ while rotating a polarizer" + "\n" + r" in transmission through 250 nm thick GaAs 111")
    offset = 5
    polType = 2
    
    path = 'LN-FS/'
    fileName = "pairPolTomography/20230725_1751_LN_50mW_horGlan_hPolPump_788nm_transmission.csv"
    title = (r"Measured $g^{(2)}(0)$ while rotating a polarizer" + "\n" + r" in transmission through 250 nm thick GaAs 111")
    offset = 0
    polType = 0
    
    # analyzePolarizationData(dataDir + path + fileName, fitg2 = True, offset = offset, title = title, saveBool = saveBool, HWPAngle=False, polType = polType)
    
    # fiber regression
    saveBool = True
    # plotCut2Wl(saveBool = saveBool)
    
    #%% Plot Figure 2 of the SPDC paper. 2x3 
    # (0,0) empty figure for schematic
    # (0,1) g(2)(t) for the Si LN device
    # (0,2) CAR and coincidence rate as a function of power
    # (1,1:3) spectra of etalon generation
    saveBool = False
    # plotFigure2Top(saveBool)
    # plotFigure2g20(saveBool)
    # plotFigure2Spectra(saveBool)
    
    #%% Plot figure 1 simple spectra
    lambMin = 0.9e-6
    lambMax = 1.5e-6
    L = 10.145e-6
    # plotSimpleSpectraPaper(lambMin, lambMax, 1000, L = L, normBool = False, interfaceType=2,  
    #                                         interferenceBool=True, emissionProbBool=True)
    # plotSimpleSpectraPaper(lambMin, lambMax, 1000, L = L, normBool = False, interfaceType=2, 
    #                                         interferenceBool=True, emissionProbBool=True, etalonBool = False)
    
    # %% plot the polarization tomography for the Si cells

    # new analysis
    saveBool = False
    
    path = 'LN-FS/'
    fileName = "pairPolTomography/20230725_1751_LN_50mW_horGlan_hPolPump_788nm_transmission.csv"
    offset = 0
    polType = 0
    
    # analyzePolarizationDataPaper(dataDir + path + fileName, fitg2 = True, offset = offset, title = title, saveBool = saveBool, HWPAngle=False, polType = polType)
    
    # %% fiber regression
    # plotCut2Wl(saveBool = True)
    
    # %% Plot FAS 
    saveBool = True
    lambp = 0.787e-6
    lambMin =0.825e-6
    lambMax = 2*lambMin 
    L = 10.145e-6
    # L = 1e-6
    thetaRange = 20*np.pi/180
    s = 2e5
    chi = 34e-12
    # plotFASForPaper(lambMin, lambMax, lambp, thetaRange, L, N=300, saveBool=saveBool, etalonBool = True)
    # plotFASForPaper(lambMin, lambMax, lambp, thetaRange, L, N=300, M = 101, chi = chi, s = s,saveBool=saveBool, etalonBool = True)
    # plotNLFASForPaper(lambMin, lambMax, lambp, thetaRange, L, N=300, M = 11, chi = chi, s = s, saveBool=saveBool)
    s = [1, 1, 3.1e8]
    plotFASGridForPaper(lambMin, lambMax, lambp, thetaRange, L, N=3000, M = 101, chi = chi, s = s, saveBool=saveBool)
    
    lambMin = 1e-6
    # plotFASRSquared(lambMin, lambMax, lambp, thetaRange, L, N=100, M = 30, chi = chi, saveBool=saveBool)
    # plotGainPumpPower(saveBool = saveBool, lambP = lambp, L = L, chi = chi)
    
    lambMin = 1e-6
    # plotGainPumpPowerRSquared(lambMin, lambMax, saveBool = saveBool, lambP = lambp, L = L, chi = chi)
    
    #%% MS figures
    saveBool = False
    # plotChangjinDataCounts(saveBool = saveBool)
    # plotChangjinDataCountsFFBB(saveBool = saveBool)
    # plotMSCountrate(saveBool = saveBool)
    # plotProbEmissionsMS(saveBool = saveBool)
    # plotProbEmissionsMS(saveBool = saveBool)
    
    plt.show()
    
    
    
    
    