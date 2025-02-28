# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:57:00 2023

@author: Nicholas Sorensen

@purpose: written to better understand the phasematching and angular matching
        for SPDC in crystals. Particularly thin crystals of LN. 
        
        Emulates the functionality of SPDCalc.org
        
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_custom import plot_custom, thesisColorPalette, plotPalette
from datetime import datetime
from numpy import genfromtxt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
from tqdm import tqdm
import cProfile
import pstats

from pathlib import Path

dirCurr = Path.cwd()


labelfontsize = 12
numfontsize = 10 
textwidth = 8.5 - 2 # in inches
lw = 1

class lithiumNiobate:
    indexE = str(dirCurr) + '\\indexData\\lithiumNiobate/Zelmon-e.csv'
    indexO = str(dirCurr) + '\\indexData\\lithiumNiobate/Zelmon-o.csv'
    
    # dispersion formulae for ordinary and extraordinary light
    def EIndexDisp(lamb):
        lambLocal = lamb*1e6
        return np.sqrt(1 
                        + 2.9804*lambLocal**2/(lambLocal**2 - 0.02047) 
                        + 0.5981*lambLocal**2/(lambLocal**2 - 0.0666)
                        + 8.9543*lambLocal**2/(lambLocal**2 - 416.08))
    
    def OIndexDisp(lamb):
        lambLocal = lamb*1e6
        return np.sqrt(1 
                        + 2.6734*lambLocal**2/(lambLocal**2 - 0.01764) 
                        + 1.2290*lambLocal**2/(lambLocal**2 - 0.05914)
                        + 12.614*lambLocal**2/(lambLocal**2 - 474.60))

class galliumPhosphide:
    
    dataDir = str(dirCurr) + "\\galliumPhosphide\\"

    def loadData():
        myData = genfromtxt(galliumPhosphide.dataDir + 'Adachi.csv', delimiter=',')
        return myData
    
    def sellmaier(lamb, B1, B2, B3, C1, C2, C3, A):
        return 1 + A + B1*lamb**2/(lamb**2-C1) + B2*lamb**2/(lamb**2-C2) + B3*lamb**2/(lamb**2-C3) 
    
    def sell2Index(lamb, B1, B2, B3, C1, C2, C3, A):
        return np.sqrt(np.abs(galliumPhosphide.sellmaier(lamb, B1, B2, B3, C1, C2, C3, A)))**2
    
    def fitIndexData(wl, index, poptGuess = [4, 10, 0, 0.05, 400, 0, -1]):
        popt, pcov = curve_fit(galliumPhosphide.sellmaier, wl, index, p0 = poptGuess)
        return popt
    
    def loadIndexData():
        myData = galliumPhosphide.loadData()
        myData = myData[~np.isnan(myData[:,0]),:]
        wl = myData[:,0]
        n = myData[:,1]
        k = myData[:,2]
        return wl, n, k
    
    def findRegression():
        # only valid from 0.38 to 1.5 um
        
        wl, n, k = galliumPhosphide.loadIndexData()
        
        poptn = galliumPhosphide.fitIndexData(wl, n)
        # poptk = galliumPhosphide.fitIndexData(wl, k, poptGuess = [0.012, 3000, 0, 0.1424, 4.65e6, 0, -1.02])
        # print(poptn, poptk)
        return poptn
    
    
    def savePoptAsCSV():
        poptn = galliumPhosphide.findRegression()
        np.savetxt(galliumPhosphide.dataDir + "poptnk.csv", np.vstack([poptn]), delimiter=",")
                    
        
    def sellmaierReal(wl):
        # wl in um
        popt = genfromtxt(galliumPhosphide.dataDir + 'poptnk.csv', delimiter=',')
        
        return galliumPhosphide.sell2Index(wl, *popt)
    
    def sellmaierImag(wl):
        # wl in um
        popt = genfromtxt(galliumPhosphide.dataDir + 'poptnk.csv', delimiter=',')
        
        return galliumPhosphide.sell2Index(wl, *popt)
    
    def interpIndex(wl, wlData, nData):
        return np.interp(wl, wlData, nData)
        
    
    def plotData():
        wl, n, k = galliumPhosphide.loadIndexData()
        poptn = galliumPhosphide.findRegression()
        
        fig, ax = plot_custom(
            3, 3, r"wavelength, $\lambda$ ($\mu$m)", r"refractive index, $n, k$", equal=False
        )
        
        ax.plot(wl, n)
        # ax.plot(wl, k)
        ax.plot(wl, galliumPhosphide.sell2Index(wl, *poptn))
        # ax.plot(wl, galliumPhosphide.sell2Index(wl, *poptk))
        
        return fig, ax
    
class BBO:
    # dispersion formulae for ordinary and extraordinary light
    def EIndexDisp(x):
        x = x*1e6 # x is lamb
        return (1+1.151075/(1-0.007142/x**2)+0.21803/(1-0.02259/x**2)+0.656/(1-263/x**2))**.5
    def OIndexDisp(x):
        x = x*1e6 # x is lamb
        return (1+0.90291/(1-0.003926/x**2)+0.83155/(1-0.018786/x**2)+0.76536/(1-60.01/x**2))**.5
    
class silicon:
    
    dataDir =  str(dirCurr) + "\\refractiveIndex\\silicon\\"

    def loadData():
        myData = genfromtxt(silicon.dataDir + 'green-2008.csv', delimiter=',')
        return myData
    
    def sellmaier(lamb, B1, B2, B3, C1, C2, C3, A):
        return 1 + A + B1*lamb**2/(lamb**2-C1) + B2*lamb**2/(lamb**2-C2) + B3*lamb**2/(lamb**2-C3) 
    
    def sell2Index(lamb, B1, B2, B3, C1, C2, C3, A):
        return np.sqrt(np.abs(silicon.sellmaier(lamb, B1, B2, B3, C1, C2, C3, A)))**2
    
    def fitIndexData(wl, index, poptGuess = [4, 10, 0, 0.05, 400, 0, -1]):
        popt, pcov = curve_fit(silicon.sellmaier, wl, index, p0 = poptGuess)
        return popt
    
    def loadIndexData():
        myData = silicon.loadData()
        myData = myData[myData[:,0]>=0.38]
        wl = myData[:,0]
        n = myData[:,1]
        k = myData[:,2]
        return wl, n, k
    
    def findSiliconRegression():
        # only valid from 0.38 to 1.5 um
        
        wl, n, k = silicon.loadIndexData()
        
        poptn = silicon.fitIndexData(wl, n)
        poptk = silicon.fitIndexData(wl, k, poptGuess = [0.012, 3000, 0, 0.1424, 4.65e6, 0, -1.02])
        # print(poptn, poptk)
        return poptn, poptk
    
    
    def savePoptAsCSV():
        poptn, poptk = silicon.findSiliconRegression()
        np.savetxt(silicon.dataDir + "poptnk.csv", np.vstack([poptn, poptk]), delimiter=",")
                    
        
    def sellmaierReal(wl):
        # wl in um
        popt = genfromtxt(silicon.dataDir + 'poptnk.csv', delimiter=',')
        
        return silicon.sell2Index(wl, *popt[0,:])
    
    def sellmaierImag(wl):
        # wl in um
        popt = genfromtxt(silicon.dataDir + 'poptnk.csv', delimiter=',')
        
        return silicon.sell2Index(wl, *popt[1,:])
    
    def interpIndex(wl, wlData, nData):
        return np.interp(wl, wlData, nData)
        
    
    def plotData():
        wl, n, k = silicon.loadIndexData()
        poptn, poptk = silicon.findSiliconRegression()
        
        fig, ax = plot_custom(
            9, 6, r"wavelength, $\lambda$ ($\mu$m)", r"refractive index, $n, k$", equal=False
        )
        
        ax.plot(wl, n)
        ax.plot(wl, k)
        ax.plot(wl, silicon.sell2Index(wl, *poptn))
        ax.plot(wl, silicon.sell2Index(wl, *poptk))
        
        return fig, ax
    
    def indexComplex(wl):
        # wl in m
        return silicon.sellmaierReal(wl*1e6) + 1j * silicon.sellmaierImag(wl*1e6)
    
    def indexComplexInterp(wl):
        wlData, nData, kData = silicon.loadIndexData()
        return silicon.interpIndex(wl*1e6, wlData, nData) + 1j * silicon.interpIndex(wl*1e6, wlData, kData)
    
class waferLNSi:
    
    sPol = 'sPol'
    pPol = 'pPol'
    
    def cosThetat(n1, n2, thetai):
        return np.sqrt(1-(n1/n2*np.sin(thetai))**2)
    
    def rpCalc(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = n2*np.cos(thetai) - n1*cosThetat
        den = n2*np.cos(thetai) + n1*cosThetat
        
        return num/den
    
    def tpCalc(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = 2*n1*np.cos(thetai)
        den = n2*np.cos(thetai) + n1*cosThetat
        
        return num/den
    
    def rsCalc(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = n1*np.cos(thetai) - n2*cosThetat
        den = n1*np.cos(thetai) + n2*cosThetat
        
        return num/den
    
    def tsCalc(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = 2*n1*np.cos(thetai)
        den = n1*np.cos(thetai) + n2*cosThetat
        
        return num/den
    
    def rpCalcAir(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = n2*np.cos(thetai) - n1*cosThetat
        den = n2*np.cos(thetai) + n1*cosThetat
        
        return num/den
    
    def tpCalcAir(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = 2*n1*np.cos(thetai)
        den = n2*np.cos(thetai) + n1*cosThetat
        
        return num/den
    
    def rsCalcAir(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = n1*np.cos(thetai) - n2*cosThetat
        den = n1*np.cos(thetai) + n2*cosThetat
        
        return num/den
    
    def tsCalcAir(thetai, wl):
        # assume wl is in m and thetai is in rad
        
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        num = 2*n1*np.cos(thetai)
        den = n1*np.cos(thetai) + n2*cosThetat
        
        return num/den
    
    def reflectance(thetai, wl, pol):
        
        if pol == waferLNSi.sPol:
            r = waferLNSi.rsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            r = waferLNSi.rpCalc(thetai, wl)
        
        return np.abs(r)**2
    
    def angleReflectivity(thetai, wl, pol):
        if pol == waferLNSi.sPol:
            r = waferLNSi.rsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            r = waferLNSi.rpCalc(thetai, wl)
        return np.angle(r)

        
    def transmittance(thetai, wl, pol, dist):
        if pol == waferLNSi.pPol:
            t = waferLNSi.tpCalc(thetai, wl)
        elif pol == waferLNSi.sPol:
            t = waferLNSi.tsCalc(thetai, wl)
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        T = np.real(n2)*cosThetat/(n1*np.cos(thetai))*np.abs(t)**2
        
        return T
    
    def angleTrans(thetai, wl, pol):
        if pol == waferLNSi.sPol:
            t = waferLNSi.tsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            t = waferLNSi.tpCalc(thetai, wl)
        return np.angle(t)
    
    def absorption(thetai, wl, dist):
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        omega = optics.lamb2omega(wl)
        
        kappa = np.imag(n2)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        
        a = np.exp(-2*omega*kappa/optics.c*dist/cosThetat)
        return a
        
    
    def plotReflectanceAndPhase(lambLow, lambHigh, thetaRange, N = 100):
        
        lamb = np.linspace(lambLow, lambHigh, N)
        thetai = np.linspace(-thetaRange/2, thetaRange/2,N)
        
        tthetai,llamb = np.meshgrid(thetai, lamb, indexing='ij');        
        
        Rp = waferLNSi.reflectance(tthetai, llamb, waferLNSi.pPol)
        Rs = waferLNSi.reflectance(tthetai, llamb, waferLNSi.sPol)
        
        angRp = waferLNSi.angleReflectivity(tthetai, llamb, waferLNSi.pPol)
        angRs = waferLNSi.angleReflectivity(tthetai, llamb, waferLNSi.sPol)
        
        fig, ax = plot_custom(
            12,9,  [r"incident angle, $\theta_i$ ($^\circ$)", r"incident angle, $\theta_i$ ($^\circ$)"], 
            [r"wavelength, $\lambda$ (um)",r"wavelength, $\lambda$ (um)"],ncol = 2, nrow = 2, equal=False, 
            numsize = 20, commonY = True, commonX = True, axefont = 25
        )
        
        p1 = ax[0,0].pcolormesh(np.degrees(tthetai), llamb*1e6, Rp, cmap='hot', vmin = 0, vmax = 1)
        p2 = ax[0,1].pcolormesh(np.degrees(tthetai), llamb*1e6, Rs, cmap='hot', vmin = 0, vmax = 1)
        p3 = ax[1,0].pcolormesh(np.degrees(tthetai), llamb*1e6, np.degrees(angRp), cmap='hot')
        p4 = ax[1,1].pcolormesh(np.degrees(tthetai), llamb*1e6, np.degrees(angRs), cmap='hot')
        
        ax[0,0].set_title(r"p-polarized", fontsize = 25)
        ax[0,1].set_title(r"s-polarized", fontsize = 25)
        
        cbar1 = fig.colorbar(p1, ax = ax[0,0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p2, ax = ax[0,1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'reflectance, $R$ (au)', rotation = 90, fontsize = 25, labelpad = 10)
        
        cbar1 = fig.colorbar(p3, ax = ax[1,0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p4, ax = ax[1,1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'reflection phase ($^\circ$)', rotation = 90, fontsize = 25, labelpad = 10)
        
        
    def plotTransmittanceAndPhase(lambLow, lambHigh, thetaRange, dist, N = 100, absorbBool = False):
        
        lamb = np.linspace(lambLow, lambHigh, N)
        thetai = np.linspace(-thetaRange/2, thetaRange/2,N)
        
        tthetai,llamb = np.meshgrid(thetai, lamb, indexing='ij');        
        
        Tp = waferLNSi.transmittance(tthetai, llamb, waferLNSi.pPol, dist)
        Ts = waferLNSi.transmittance(tthetai, llamb, waferLNSi.sPol, dist)
        absorption = waferLNSi.absorption(tthetai, llamb, dist)
        
        angTp = waferLNSi.angleTrans(tthetai, llamb, waferLNSi.pPol)
        angTs = waferLNSi.angleTrans(tthetai, llamb, waferLNSi.sPol)
        
        if absorbBool == True:
            
            Tp *= absorption
            Ts *= absorption
        
        fig, ax = plot_custom(
            12,9,  [r"incident angle, $\theta_i$ ($^\circ$)", r"incident angle, $\theta_i$ ($^\circ$)"], 
            [r"wavelength, $\lambda$ (um)",r"wavelength, $\lambda$ (um)"],ncol = 2, nrow = 2, equal=False, 
            numsize = 20, commonY = True, commonX = True, axefont = 25
        )
        
        p1 = ax[0,0].pcolormesh(np.degrees(tthetai), llamb*1e6, Tp, cmap='hot', vmin = 0, vmax = 1)
        p2 = ax[0,1].pcolormesh(np.degrees(tthetai), llamb*1e6, Ts, cmap='hot', vmin = 0, vmax = 1)
        p3 = ax[1,0].pcolormesh(np.degrees(tthetai), llamb*1e6, np.degrees(angTp), cmap='hot')
        p4 = ax[1,1].pcolormesh(np.degrees(tthetai), llamb*1e6, np.degrees(angTs), cmap='hot')
        
        ax[0,0].set_title(r"p-polarized", fontsize = 25)
        ax[0,1].set_title(r"s-polarized", fontsize = 25)
        
        cbar1 = fig.colorbar(p1, ax = ax[0,0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p2, ax = ax[0,1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'transmittance, T (au)', rotation = 90, fontsize = 25, labelpad = 10)
        
        cbar1 = fig.colorbar(p3, ax = ax[1,0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p4, ax = ax[1,1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'transmission phase ($^\circ$)', rotation = 90, fontsize = 25, labelpad = 10)
        
    def TIR(wl):
        n0 = 1
        n1 = lithiumNiobate.EIndexDisp(wl)
        return np.arcsin(n0/n1)
    def TIRBBO(wl):
        n0 = 1
        n1 = BBO.OIndexDisp(wl)
        return np.arcsin(n0/n1)
    
class optics:
    
    c = 2.99792458e8 # m/s    
    
    def omega2Nu(omega):
        return omega/(2*np.pi)
    
    def nu2omega(nu):
        return 2*np.pi*nu
    
    # lambda is defined as the wavelength in material of index n
    def lamb2omega(lamb, n = 1):
        return 2*np.pi*optics.c/lamb/n
    
    def omega2lamb(omega, n = 1):
        return 2*np.pi*optics.c/omega/n
    
    def omega2lambBand(omega, omegaBand, lamb):
        return omegaBand * lamb**2 / (2*np.pi * optics.c)
    
    def lamb2omegaBand(lamb, lambBand, omega):
        return 2*np.pi*optics.c*lambBand/lamb**2
    
    def lamb2k(lamb, n):
        return 2 * np.pi * n / lamb
    

class SPDCalc:

    def plotIndicesData(filenameE, filenameO):
        # plot the index of refraction as a function of wavelength or frequency
        LNEIndex = pd.read_csv(filenameE)
        LNOIndex = pd.read_csv(filenameO)
        
        fig, ax = plot_custom(
            9, 6, r"wavelength, $\lambda$ ($\mu$m)", r"refractive index, $n$", equal=False
        )
        
        ax.plot(LNEIndex['wl'], LNEIndex['n'], label = r"$n_e$")
        ax.plot(LNOIndex['wl'], LNOIndex['n'], label = r"$n_o$")
        ax.legend()
    
        return fig, ax

    
    def plotBirefringenceData(filenameE, filenameO):
        # plot the birefringence of the crystal, defined as e - o. THis is negative for 
        # LN so it is a negative uniaxial crystal
        LNEIndex = pd.read_csv(filenameE)
        LNOIndex = pd.read_csv(filenameO)
        
        fig, ax = plot_custom(
            9, 6, r"wavelength, $\lambda$ ($\mu$m)", r"refractive index, $n$", equal=False
        )
        
        ax.plot(LNEIndex['wl'], LNEIndex['n'] - LNOIndex['n'])
    
        return fig, ax
    
    
    
    def calcJSI(wp, ws, wi, sig,M = 1):
        """
        wp : pump center frequency
        ws : signal center frequency
        sig : pump bandwidth

        """
        vs = ws-wp/2
        vi = wi-wp/2
        
        return M * np.exp(-1/(2*sig**2) * (vs + vi)**2 )
    
    
    
    def plotJSI(lambs, lambBand, lambp, pLambBand, M = 1, N = 100):
        # this is the distribution of photons purely from energy conservation
        fig, ax = plot_custom(
            5, 5, r"$\lambda_i$ (nm)", r"$\lambda_s$ (nm)", equal=False
        )
        
        lambi = np.linspace(lambs - lambBand/2, lambs + lambBand/2, N)
        lambs = lambi.copy()
        llambs,llambi= np.meshgrid(lambs, lambi, indexing='ij');
    
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)
        wwi = optics.lamb2omega(llambi)
        wsig = optics.lamb2omegaBand(lambp, pLambBand, wp)
        
        ax.pcolormesh(llambs/1e-9, llambi/1e-9, 
                      SPDCalc.calcJSI(wp, wws, wwi, wsig), cmap='hot')


    def calcFreqAngSpectrum(wp, ws, thetas, L, LcohBool = False, BBOBool = False, chi = 0.0001, s = 1):
    
        wi = wp-ws
    
        lambPAir = optics.omega2lamb(wp)
        lambSAir = optics.omega2lamb(ws)
        lambIAir = optics.omega2lamb(wi)
        
        # npO = lithiumNiobate.OIndexDisp(lambPAir)
        # nsO = lithiumNiobate.OIndexDisp(lambSAir)
        # niO = lithiumNiobate.OIndexDisp(lambIAir)
        
        if BBOBool is False:
            npE = lithiumNiobate.EIndexDisp(lambPAir)
            nsE = lithiumNiobate.EIndexDisp(lambSAir)
            niE = lithiumNiobate.EIndexDisp(lambIAir)
            
            # calc the parallel component of the wavevector mismatch      
            # the process works better if the light is E-polarized
            # phase matching transverse
            # thetai = (np.arcsin(-ws*nsE*np.sin(thetas)
            #                     /(wi*niE)
            #                     ) 
            #           )
            
            thetai = -np.arctan(ws*nsE*np.sin(thetas)/(wp*npE-ws*nsE*np.cos(thetas)))
            # print(thetai)
            # phase matching longitudinal
            # thetai = (np.arccos((wp*npE - ws*nsE*np.cos(thetas))
            #                     /(wi*niE)
            #                     ) 
            #           )
            # thetai = (np.arccos(nsE*ws/(niE*wi)*np.sin(thetas))) - np.pi/2
            kp = wp*npE/optics.c
            ks = ws*nsE*np.cos(thetas)/optics.c
            ki = wi*niE*np.cos(thetai)*npE/optics.c
            
            DeltaKPar = np.abs(1/optics.c*(-wp*npE
                                    + ws*nsE*np.cos(thetas)
                                    + wi*niE*np.cos(thetai)
                                    ) 
                         )             
            
            DeltaKPerp = np.abs(1/optics.c*(ws*nsE*np.sin(thetas)
                                    +wi*niE*np.sin(thetai)
                                    )
                          )
            
            # cax = plt.matshow(DeltaKPerp, vmin=-0.1e-10, vmax=0.1e-10)
            # plt.colorbar(cax)
            # print(thetai)
            
            waistp = 0.5e-6#m
        
        if BBOBool is True:
            npE = BBO.EIndexDisp(lambPAir)
            nsE = BBO.OIndexDisp(lambSAir)
            niE = BBO.OIndexDisp(lambIAir) #type I for BBO
        
            # calc the parallel component of the wavevector mismatch      
            # the process works better if the light is E-polarized
            # BBO should be longitudinally phase matched
            #transverse phase matching
            
            thetai = (np.arccos(nsE*ws/(niE*wi)*np.sin(thetas))) - np.pi/2
            
            kp = wp*npE/optics.c
            ks = ws*nsE*np.cos(thetas)/optics.c
            ki = wi*niE*np.cos(thetai)*npE/optics.c
            
            thetai = (np.arcsin(-ws*nsE*np.sin(thetas)
                               /(wi*niE)
                               ) 
                      )
            # phase matching longitudinal
            thetai = (np.arccos((wp*npE - ws*nsE*np.cos(thetas))
                                /(wi*niE)
                                ) 
                      ) * -np.sign(thetas)
            # thetai = (np.arccos(nsE*ws/(niE*wi)*np.sin(thetas))) - np.pi/2
            DeltaKPar = (1/optics.c*(-wp*npE
                                    + ws*nsE*np.cos(thetas)
                                    + wi*niE*np.cos(thetai)
                                    ) 
                         )             
            
            DeltaKPerp = (1/optics.c*(ws*nsE*np.sin(thetas)
                                    +wi*niE*np.sin(thetai)
                                    )
                          )

            # cax = plt.matshow(thetai)
            # plt.colorbar(cax)
            # calc the phase matching function
            
            waistp = 5000e-9 #m
        
        Fp = np.exp(-(waistp*DeltaKPerp)**2)
        # print(DeltaKPerp)
        
        beta = (2 * np.pi * ws * wi * chi * L * s 
                              / (optics.c**2 * np.sqrt(ks*ki)))#
        
        FPar = -1j * beta
        
        Fpm = np.abs(np.sinc(DeltaKPar*L/2/np.pi)**2)
        
        # cax = plt.matshow(Fpm)
        # plt.colorbar(cax)
        # Fpm[Fpm == 1] = 0
        
        if LcohBool is True:
            return Fpm, np.pi/DeltaKPar
        
        # print(waistp*DeltaKPerp)
        
        return Fpm*np.abs(FPar)**2*Fp
    
    def calcAnglesFreqAngSpectrum(wp, ws, thetas, thetai, L, K, Deltan, waistp = 10e-6):
        
        def fEta(K, Deltan, n, theta, k, d, lamb):
            thetaBragg = np.arcsin(K/(2*k*n))
            # print("Internal bragg angle: ", thetaBragg*180/np.pi)
            delta = theta - thetaBragg
            # cax = plt.matshow(delta)
            # plt.colorbar(cax)
            xi = k*n*delta*d * np.sin(thetaBragg)
            nu = np.pi*Deltan*d/(lamb*np.cos(thetaBragg))
            return np.sin(np.sqrt(xi**2 + nu**2))**2/(1+ xi**2/nu**2)
            
        
        wi = wp - ws
        
        lambPAir = optics.omega2lamb(wp)
        lambSAir = optics.omega2lamb(ws)
        lambIAir = optics.omega2lamb(wi)
        
        # print('lambPAir: ', lambPAir)
        # print('lambSAir: ', lambSAir)
        # print('lambIAir: ', lambIAir)
        
        npE = lithiumNiobate.EIndexDisp(lambPAir)
        nsE = lithiumNiobate.EIndexDisp(lambSAir)
        niE = lithiumNiobate.EIndexDisp(lambIAir)
        
        ki = wi/optics.c
        
        eta = fEta(K = K, Deltan = Deltan, n = niE, theta = thetai, k = ki, d = L, lamb = lambIAir)
        etaMinus = fEta(K = -K, Deltan = Deltan, n = niE, theta = thetai, k = ki, d = L, lamb = lambIAir)
        # print('max eta: ', np.max(eta))
        # cax = plt.matshow(eta)
        # plt.colorbar(cax)
        
    
        DeltaKPar = (1/optics.c*(-wp*npE
                                + ws*nsE*np.cos(thetas)
                                + wi*niE*np.cos(thetai)
                                ) 
                    )       
            
        
        FFp = [None]*3
        for i in range(3):
            
            DeltaKPerp = ((1/optics.c*(ws*nsE*np.sin(thetas)
                                    +wi*niE*np.sin(thetai)
                                    
                                    )- K*(-1+i)#*np.sign(thetai)
                          ))
            
              
            FFp[i] = np.exp(-(waistp*DeltaKPerp)**2)
            
        Fp = FFp[0]*etaMinus + FFp[1]*(1-eta-etaMinus) + FFp[2]*eta
            
        Fpm = np.abs(np.sinc(DeltaKPar*L/2/np.pi)**2)
            
        return Fp*Fpm
    
    def plotFreqAngSpectrumLine(lambsLow, lambsHigh, lambp, theta, L, N=300):
        lambs = np.linspace(lambsLow, lambsHigh, N)
        
        wp = optics.lamb2omega(lambp)
        ws = optics.lamb2omega(lambs)

        fig, ax = plot_custom(
           textwidth/2.1, textwidth/2.15, r"signal frequency, $\nu_s$ (THz)", 
           r"emission probability (au)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1, fontType = 'sansserif')
        
        axg0 = ax.twiny()
        
        FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, ws, theta, L))**2

        ax.plot(ws*1e-12/(2*np.pi), FAS/np.max(FAS), color = 'black', lw = lw)
        ax.set_ylim([0,1])
        ax.set_xlim(optics.lamb2omega(np.array([lambsHigh, lambsLow]))*1e-12/(2*np.pi))
        axg0.set_xlim(np.array([lambsLow, lambsHigh])*1e6)
        axg0.plot(lambs*1e6, FAS/np.max(FAS), color = 'red', lw = lw, alpha = 0)
        
        axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        
        axg0.set_xlabel(r"signal wavelength, $\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 10)
        plt.tight_layout()
        
        
    def plotFreqAngSpectrum(lambsLow, lambsHigh, lambp, thetaRange, L, N=300):
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,N)
        
        ttheta,llambs = np.meshgrid(theta, lambs, indexing='ij');
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)

        fig, ax = plot_custom(
           textwidth/1.5, textwidth/2.5, r"internal emission angle, $\theta$ ($^\circ$)", 
           r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
        
        axg0 = ax.twinx()
        
        FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))

        p1 = ax.pcolormesh(np.degrees(ttheta), wws*1e-12/(2*np.pi), FAS/np.max(FAS), cmap='hot')
        axg0.pcolormesh(np.degrees(ttheta), llambs*1e6, FAS/np.max(FAS), cmap='hot', alpha = 0)
        
        axg0.set_yscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        
        axg0.set_ylabel(r"signal wavelength, $\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize)
        
        cbar = fig.colorbar(p1, ax = axg0, pad=0.2)
        cbar.ax.tick_params(labelsize=numfontsize)
        cbar.set_label(r'emission probability (au)', rotation = 90, fontsize = labelfontsize, labelpad = 10)
        
        #print lines of TIR in crystal
        ws = wws[0,:]        
        TIR = waferLNSi.TIR(optics.omega2lamb(ws))
        ax.plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax.plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        
    def plotAnglesFreqAngSpectrum(lamb, lambp, thetaRange, L, Deltan, N=2000, K = 5e6, widthPump = 10e-6):
        
        fig, ax = plot_custom(
           textwidth/1.5, textwidth/1.8, r"signal internal angle, $\theta_s$ ($^\circ$)", 
           r"idler internal angle, $\theta$ ($^\circ$)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
        
        
        theta = np.linspace(-thetaRange/2, thetaRange/2,N)
        tthetas,tthetai = np.meshgrid(theta, theta, indexing='ij');
        
        ws = optics.lamb2omega(lamb)
        wp = optics.lamb2omega(lambp)

        FAS = SPDCalc.calcAnglesFreqAngSpectrum(wp, ws, tthetas, tthetai, L, K, Deltan = Deltan, waistp = widthPump)

        p1 = ax.pcolormesh(np.degrees(tthetas), np.degrees(tthetai), np.log10(FAS/np.max(FAS)), cmap='hot', vmin = -9, vmax = 0)

        cbar = fig.colorbar(p1, ax = ax)
        cbar.ax.tick_params(labelsize=numfontsize)
        cbar.set_label(r'emission probability (au)', rotation = 90, fontsize = labelfontsize, labelpad = 10)
        
    def plotAnglesFreqAngSpectrumRangeWavelength(lambLow, lambHigh, lambp, 
                                                 thetaRange, L, Deltan, N=2000, 
                                                 K = 5e6, widthPump = 100e-6, M = 10):
        
        fig, ax = plot_custom(
           textwidth/1.5, textwidth/1.8, r"signal internal angle, $\theta_s$ ($^\circ$)", 
           r"idler internal angle, $\theta$ ($^\circ$)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
        
        
        theta = np.linspace(-thetaRange/2, thetaRange/2,N)
        tthetas,tthetai = np.meshgrid(theta, theta, indexing='ij');
        
        
        wsHigh = optics.lamb2omega(lambLow)
        wsLow = optics.lamb2omega(lambHigh)
        
        wws = np.linspace(wsLow, wsHigh, M)
        wp = optics.lamb2omega(lambp)
        
        FAS = tthetas.copy()*0
        
        for ws in tqdm(wws):
            FAS += np.nan_to_num(SPDCalc.calcAnglesFreqAngSpectrum(wp, ws, tthetas, tthetai, L, K, Deltan = Deltan, waistp = widthPump))
            
        FAS[FAS==0] = 1e-200
        p1 = ax.pcolormesh(np.degrees(tthetas), np.degrees(tthetai), np.log10(np.abs(FAS/np.max(FAS))), cmap='hot', vmin = -13, vmax = 0)

        cbar = fig.colorbar(p1, ax = ax)
        cbar.ax.tick_params(labelsize=numfontsize)
        cbar.set_label(r'$\log_{10}$ normalized emission probability', rotation = 90, fontsize = labelfontsize, labelpad = 10)
        
        
    def plotFreqAngSpectrumLNTwo(lambsLow, lambsHigh, lambp, thetaRange, L, N=1000):
        

        fig, ax = plot_custom(
           textwidth, textwidth/2.5, [r"internal emission angle, $\theta$ ($^\circ$)", r"internal emission angle, $\theta$ ($^\circ$)"], 
           r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 8, nrow = 1, ncol = 2, commonY = True, 
           widthBool = True, heightRatios = [1], widthRatios = [1,1], 
        )
        plt.subplots_adjust(wspace=0.05)
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,N)
        
        ttheta,llambs = np.meshgrid(theta, lambs, indexing='ij');
        
        axg0 = ax[1].twinx()
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)
        
        LBBO = 1e-3
        
        FASLN = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))
        FASBBO = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L = 306e-6, BBOBool = False))

        p0 = ax[0].pcolormesh(np.degrees(ttheta), wws*1e-12/(2*np.pi), FASLN/np.max(FASLN), cmap='hot')
        p1 = ax[1].pcolormesh(np.degrees(ttheta), wws*1e-12/(2*np.pi), FASBBO/np.max(FASBBO), cmap='hot')
        
        axg0.pcolormesh(np.degrees(ttheta), llambs*1e6, FASBBO/np.max(FASBBO), cmap='hot', alpha = 0)
        
        axg0.set_yscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        
        axg0.set_ylabel(r"signal wavelength, $\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize)
        
        cbar0_ax = fig.add_axes([1.07, 0.22, 0.02, 0.7])
        cbar0 = fig.colorbar(p1, cax = cbar0_ax)
        cbar0.ax.tick_params(labelsize=numfontsize)
        cbar0.set_label( r'emission probability (au)', rotation=90, fontsize=labelfontsize)

        
        #print lines of TIR in crystal
        ws = wws[0,:]        
        TIR = waferLNSi.TIR(optics.omega2lamb(ws))
        ax[0].plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax[0].plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax[1].plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax[1].plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        
    def plotFreqAngSpectrumLNBBO(lambsLow, lambsHigh, lambp, thetaRange, L, N=100):
        

        fig, ax = plot_custom(
           textwidth, textwidth/2.5, [r"internal emission angle, $\theta$ ($^\circ$)", r"internal emission angle, $\theta$ ($^\circ$)"], 
           r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
           numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
           labelPadX = 8, labelPadY = 8, nrow = 1, ncol = 2, commonY = True, 
           widthBool = True, heightRatios = [1], widthRatios = [1,1], 
        )
        plt.subplots_adjust(wspace=0.05)
        
        lambs = np.linspace(lambsLow, lambsHigh, N)
        theta = np.linspace(-thetaRange/2, thetaRange/2,N)
        thetaBBO = np.linspace(-thetaRange/20, thetaRange/20,N)
        
        ttheta,llambs = np.meshgrid(theta, lambs, indexing='ij');
        tthetaBBO,_ = np.meshgrid(thetaBBO, lambs, indexing='ij');
        
        axg0 = ax[1].twinx()
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llambs)
        
        LBBO = 1e-3
        
        FASLN = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L))
        FASBBO = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, tthetaBBO, L = 1e3, BBOBool = True))

        cax = plt.matshow(FASBBO/np.max(FASBBO))
        plt.colorbar(cax)
        p0 = ax[0].pcolormesh(np.degrees(ttheta), wws*1e-12/(2*np.pi), FASLN, cmap='hot')
        p1 = ax[1].pcolormesh(np.degrees(tthetaBBO), wws*1e-12/(2*np.pi), FASBBO/np.max(FASBBO), cmap='hot')
        
        # axg0.pcolormesh(np.degrees(ttheta), llambs*1e6, FASBBO/np.max(FASBBO), cmap='hot', alpha = 0)
        
        # axg0.set_yscale('function', functions=(optics.lamb2omega, optics.omega2lamb))
        
        # axg0.set_ylabel(r"signal wavelength, $\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize)
        
        # cbar0_ax = fig.add_axes([1.07, 0.22, 0.02, 0.7])
        # cbar0 = fig.colorbar(p1, cax = cbar0_ax)
        # cbar0.ax.tick_params(labelsize=numfontsize)
        # cbar0.set_label( r'emission probability (au)', rotation=90, fontsize=labelfontsize)

        
        #print lines of TIR in crystal
        ws = wws[0,:]        
        TIR = waferLNSi.TIR(optics.omega2lamb(ws))
        ax[0].plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax[0].plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        # ax[1].plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        # ax[1].plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        
   
    def plotPhaseMatchingVsNonPhaseMatching(chi = 1, saveBool = False):
        fig, ax = plot_custom(
            textwidth/1.4, textwidth/3.5, r"normalized crystal length, $L/L_{\text{coh}}$", 
            "probability of \n emission (au)",  axefont = labelfontsize, 
            numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
            labelPadX = 5, labelPadY = 18
        )
        
        
        colorArray = thesisColorPalette(2, paletteType = "smooth")
        
        figHandles = [None]*2
        
        
        lambP = 788e-9
        lambS = 2*lambP
        
        wp = optics.lamb2omega(lambP)
        ws = optics.lamb2omega(lambS)
        
        ### Plot the phase matched function
        L = np.linspace(0, 100e-6,1000)
        PPM = chi**2*L**2
        
        
        ### Plot the non-phase matched function
        FpmNPM, LcohNPM = SPDCalc.calcFreqAngSpectrum(wp, ws, 0, L, LcohBool = True)
        PNPM = FpmNPM*chi**2*L**2
        
        maxPNPM = np.max(PNPM)
        
        figHandles[0], = ax.plot(np.abs(L/LcohNPM), PNPM/maxPNPM, lw = lw, color = colorArray[-1])
        figHandles[1], = ax.plot(np.abs(L/LcohNPM), PPM/maxPNPM, lw = lw, color = colorArray[0])
        

        ax.set_yscale('log')
        ax.set_ylim([0.01, 10000])
        ax.set_xlim([0, 10])
        
        labels = [r"phase-matched", r"non-phase-matched"]
        
        leg = ax.legend(handles = figHandles, labels = labels, 
                  loc = "upper left", ncol = 2)
        
        leg.get_frame().set_linewidth(0.0)
        
        
    def plotAngularCorrelationExperiment(lambWrite, angleWriteExt, lambPump, Deltan,
                                         widthPump, L = 300e-6, degBool = True, lambSig = 1e-6, N = 1000,
                                         lambLow = 500e-9, lambHigh = 1000e-9, M = 100, wsRangeBool = False):
               
        if degBool is True:
            lambSig = 2*lambPump
            
        nWrite = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        angleWrite = np.arcsin(1/nWrite(lambWrite)*np.sin(angleWriteExt))
        kWrite = optics.lamb2k(lambWrite, nWrite(lambWrite))
        
        
        K = 2*kWrite*np.sin(angleWrite)
        
        print("K: ", K)
        print("Internal writing angle: ", angleWrite*180/np.pi)
        print("lambSig: ", lambSig)
        if wsRangeBool is False:
            SPDCalc.plotAnglesFreqAngSpectrum(lamb = lambSig, lambp = lambPump, thetaRange = np.pi/2, L=L, 
                                              K = K, N = N, Deltan = Deltan, widthPump= widthPump)
        if wsRangeBool is True:
            SPDCalc.plotAnglesFreqAngSpectrumRangeWavelength(lambLow=lambLow, lambHigh=lambHigh, lambp = lambPump, thetaRange = np.pi/2, L=L, 
                                              K = K, N = N, Deltan = Deltan, widthPump= widthPump, M = M)
            
        
        
        
class SPDCResonator:
    def calcProbEmission(lambLow, lambHigh, lambp, thetaRange, L, N):
    
        lamb = np.linspace(lambLow, lambHigh, N)
        thetai = np.linspace(-thetaRange/2, thetaRange/2,N)
    
        tthetai,llamb = np.meshgrid(thetai, lamb, indexing='ij');        
        
        # calculate reflectivity
        # rp = waferLNSi.rpCalc(tthetai, llamb)
        rs = waferLNSi.rsCalc(tthetai, lamb)
        
        wp = optics.lamb2omega(lambp)
        wws = optics.lamb2omega(llamb)
        
        # calculate frequency angular spectrum
        FAS = np.nan_to_num(SPDCalc.calcFreqAngSpectrum(wp, wws, tthetai, L))
        FAS = np.abs(FAS)/np.max(FAS)
    
        return tthetai, wws, tthetai, FAS*np.abs(rs)**2
    def plotProbEmission(lambLow, lambHigh, lambp, thetaRange, L, N=300):
        
        tthetai, wws, tthetai, probEmission = SPDCResonator.calcProbEmission(
            lambLow, lambHigh, lambp, thetaRange, L, N=N)
        
        fig, ax = plot_custom(
            10, 6, r"internal emission angle, $\theta$ ($^\circ$)", r"signal frequency, $\nu_s$ (THz)", equal=False, 
            numsize = 20, labelPad = 17
        )
        
        p1 = ax.pcolormesh(np.degrees(tthetai), wws*1e-12/(2*np.pi), probEmission, cmap='hot')
        
        cbar = fig.colorbar(p1, ax = ax)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(r'reflectance probability (au)', rotation = 90, fontsize = 25, labelpad = 10)
        
        #print lines of TIR in crystal
        ws = wws[0,:]
        TIR = waferLNSi.TIR(optics.omega2lamb(ws))
        ax.plot(np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        ax.plot(-np.degrees(TIR),ws*1e-12/(2*np.pi), 'b--', zorder = 10)
        
    def calcInteractionMatrix(lambP=0.788e-6, theta = 0, L = 10.1e-6,chi = 0.0001, s =1, etalonBool = True, gainBool = False):
        if etalonBool:
            def calct(n1, n2, phase, theta):
                return 2*np.sqrt(n1 * n2)*np.cos(theta)/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))
                
            def calcr(n1, n2, phase, theta):
                t = calct(n1, n2, phase, theta)
                return np.sqrt(1-t**2) * np.exp(1j*phase)
        else:
            
            def calct(n1, n2, phase, theta):
                return 1+ 0*theta
                
            def calcr(n1, n2, phase, theta):
                t = calct(n1, n2, phase, theta)
                return 0*theta
            
        
        theta = np.abs(theta)
                
        # def calcr(n1, n2, phase, theta):
        #     return np.abs((n1*np.cos(theta) - n2*np.sqrt(1-n1/n2*np.sin(theta)))/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))) * np.exp(1j*phase)
            
        # def calct(r):
        #     return 1-np.abs(r)**2
            
        # def calct(n1, n2, phase, theta):
        #     return 2*np.sqrt(n1*n2)/(n1 + n2)
            
        # def calcr(n1, n2, phase, theta):
        #     return (n1 - n2)/(n1 + n2) * np.exp(1j*phase)
        
        # def calcr(n1, n2, phase, theta):
        #     return ((n1*np.cos(theta) - n2*np.sqrt(1-n1/n2*np.sin(theta)))/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))) * np.exp(1j*(phase))
            
        # def calct(n1, n2, phase, theta):
        #     r = calcr(n1, n2, phase, theta)
        #     return 2*np.sqrt(n1*n2)*np.cos(theta)/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))
        
        lambConj = lambda lamb: 1/(1/lambP - 1/lamb)
        
        npump = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        ns = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        
        if etalonBool is True:
            n2 = lambda lamb: silicon.sellmaierReal(lamb*1e6)
            n1 = lambda lamb: 1
        else:
            n2 = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
            n1 = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        
        # kp = lambda lamb: optics.lamb2k(lamb, npump(lamb))
        # ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))
        
        
        # thetai = lambda lamb: (np.arcsin(-ks(lamb)*np.sin(theta)
        #                     /(ks(lambConj(lamb)))
        #                     ) 
        #           )
        
        ki = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))
        ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))
        kp = optics.lamb2k(lambP, npump(lambP))
        
        thetai = lambda lamb: -np.arctan(np.abs(ks(lamb))*np.sin(theta)/(np.abs(kp)-np.abs(ks(lamb))*np.cos(theta)))
        
        deltas = lambda lamb: L*ks(lamb)*np.cos(theta)
        deltai = lambda lamb: L*ki(lamb)*np.cos(thetai(lamb))
        

        ksz = lambda lamb: optics.lamb2k(lamb, ns(lamb))*np.cos(theta)
        kiz = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))*np.cos(thetai(lamb))
        
        deltakz = lambda lamb: -1* (ksz(lamb) + kiz(lamb) - kp)
        deltaz = lambda lamb: L * deltakz(lamb)
        # deltas = lambda lamb: L*ksz(lamb)
        deltap = L*kp
        
        
        # t1s = lambda lamb: calct(n1(lamb), ns(lamb))
        # t2s = lambda lamb: calct(ns(lamb), n2(lamb))
        # r1s = lambda lamb: calcr(t1s(lamb), deltas(lamb) )
        # r2s = lambda lamb: calcr(t2s(lamb), deltas(lamb) - np.pi)
        
        t1p = calct(n1(lambP), ns(lambP), deltap, 0)
        t2p = calct(ns(lambP), n2(lambP), deltap, 0)
        r1p = calcr(n1(lambP), ns(lambP), deltap, 0)
        r2p = calcr(ns(lambP), n2(lambP), deltap+np.pi, 0)
        
        t1s = lambda lamb: calct(n1(lamb), ns(lamb), deltas(lamb), theta)
        t2s = lambda lamb: calct(ns(lamb), n2(lamb), deltas(lamb), theta)
        r1s = lambda lamb: calcr(n1(lamb), ns(lamb), deltas(lamb), theta)
        r2s = lambda lamb: calcr(ns(lamb), n2(lamb), deltas(lamb)+np.pi, theta)
        
        t1i = lambda lamb: calct(n1(lambConj(lamb)), ns(lambConj(lamb)), deltai(lamb), thetai(lamb))
        t2i = lambda lamb: calct(ns(lambConj(lamb)), n2(lambConj(lamb)), deltai(lamb),  thetai(lamb))
        r1i = lambda lamb: calcr(n1(lambConj(lamb)), ns(lambConj(lamb)), deltai(lamb),  thetai(lamb))
        r2i = lambda lamb: calcr(ns(lambConj(lamb)), n2(lambConj(lamb)), deltai(lamb)+np.pi,  thetai(lamb))
        
        # r1s = lambda lamb, theta: calcr(n1(lamb), ns(lamb), deltas(lamb), theta)
        # r2s = lambda lamb, theta: calcr(ns(lamb), n2(lamb), deltas(lamb) - np.pi, theta)
        # t1s = lambda lamb, theta: calct(r1s(lamb), theta)
        # t2s = lambda lamb, theta: calct(r2s(lamb), theta)
        
        E0p = t1p/(1 - r1p*r2p)*s;
        E0m = E0p*r2p*1;
        
        # E0p = t1p/(1 - r1p*r2p)*s;
        # E0m = E0p*r2p*1;
        
        betaP = lambda lamb: (2 * np.pi * optics.lamb2omega(lamb) * optics.lamb2omega(lambConj(lamb)) * chi * L * E0p 
                              / (optics.c**2 * np.sqrt(ksz(lamb)*kiz(lamb))))       #
        betaM = lambda lamb: (2 * np.pi * optics.lamb2omega(lamb) * optics.lamb2omega(lambConj(lamb)) * chi * L * E0m
                              / (optics.c**2 * np.sqrt(ksz(lamb)*kiz(lamb))))        #*n1(lambP)/npump(lambP)
        
        gammaP = lambda lamb: np.emath.sqrt((betaP(lamb))**2 - deltaz(lamb)**2/4)
        gammaM = lambda lamb: np.emath.sqrt((betaM(lamb))**2 - deltaz(lamb)**2/4) #np.abs
    
        mu = lambda lamb: 1j*deltaz(lamb)/2
        eta = mu
        
        # see Kitaeva 2004 Eq (11)
        W11p = lambda lamb:      np.exp(-              mu(lamb))  * ( np.cosh(gammaP(lamb)) + eta(lamb) * np.sinh(gammaP(lamb)) / gammaP(lamb) )
        W22p = lambda lamb:      np.exp(- np.conjugate(mu(lamb))) * ( np.cosh(gammaP(lamb)) -  mu(lamb) * np.sinh(gammaP(lamb)) / gammaP(lamb) )
        W12p = lambda lamb: (-1j *              betaP(lamb)  * np.sinh(gammaP(lamb))/gammaP(lamb))
        W21p = lambda lamb: ( 1j * np.conjugate(betaP(lamb)) * np.sinh(gammaP(lamb))/gammaP(lamb))
        W11m = lambda lamb: (      np.exp(-              mu(lamb))  * ( np.cosh(gammaM(lamb)) + eta(lamb) * np.sinh(gammaM(lamb)) / gammaM(lamb) ))
        W22m = lambda lamb: (      np.exp(- np.conjugate(mu(lamb))) * ( np.cosh(gammaM(lamb)) -  mu(lamb) * np.sinh(gammaM(lamb)) / gammaM(lamb) ))
        W12m = lambda lamb: (-1j *              betaM(lamb)  * np.sinh(gammaM(lamb))/gammaM(lamb))
        W21m = lambda lamb: ( 1j * np.conjugate(betaM(lamb)) * np.sinh(gammaM(lamb))/gammaM(lamb))
        
        
        W = lambda lamb: np.array([[W11p(lamb), W12p(lamb), 0, 0],[W21p(lamb), W22p(lamb), 0, 0],[0, 0, W11m(lamb), W12m(lamb)],[0, 0, W21m(lamb), W22m(lamb)]])
        # W = lambda lamb: np.array([[W11p(lamb), 0, 0, 0],[0, W22p(lamb), 0, 0],[0, 0, W11m(lamb), 0],[0, 0, 0, W22m(lamb)]])
        
        # tau1 = lambda lamb: np.array([[t1s(lamb), 0, 0, 0], [0, np.conjugate(t1s(lambConj(lamb))), 0, 0], [0, 0, t2s(lamb), 0], [0, 0, 0, np.conjugate(t2s(lambConj(lamb)))]])
        # tau2 = lambda lamb: np.array([[t2s(lamb), 0, 0, 0], [0, np.conjugate(t2s(lambConj(lamb))), 0, 0], [0, 0, t1s(lamb), 0], [0, 0, 0, np.conjugate(t1s(lambConj(lamb)))]])

        # rho1 = lambda lamb: np.array([[0, 0, r1s(lamb), 0], [0, 0, 0, np.conjugate(r1s(lambConj(lamb)))], [r2s(lamb), 0, 0, 0], [0, np.conjugate(r2s(lambConj(lamb))), 0, 0]])
        # rho2 = lambda lamb: np.transpose(np.conjugate(rho1(lamb)))
        
        tau1 = lambda lamb: np.array([[t1s(lamb), 0, 0, 0], [0, np.conjugate(t1i(lamb)), 0, 0], [0, 0, t2s(lamb), 0], [0, 0, 0, np.conjugate(t2i(lamb))]])
        tau2 = lambda lamb: np.array([[t2s(lamb), 0, 0, 0], [0, np.conjugate(t2i(lamb)), 0, 0], [0, 0, t1s(lamb), 0], [0, 0, 0, np.conjugate(t1i(lamb))]])

        rho1 = lambda lamb: np.array([[0, 0, r1s(lamb), 0], [0, 0, 0, np.conjugate(r1i(lamb))], [r2s(lamb), 0, 0, 0], [0, np.conjugate(r2i(lamb)), 0, 0]])
        rho2 = lambda lamb: np.transpose(np.conjugate(rho1(lamb)))
        
        I = np.eye(4)
        
        U = lambda lamb: tau2(lamb).dot(np.linalg.inv(I - np.matmul(W(lamb), rho1(lamb)))).dot(W(lamb)).dot(tau1(lamb)) - (rho2(lamb))
        if gainBool == True:
            gainM = lambda lamb: gammaM(lamb)
            gainP = lambda lamb: gammaP(lamb)
            return thetai
        else:
            return U
        
        
    def validateModel(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6,chi = 0.0001, s =1 ):
        Ufunc = SPDCResonator.calcInteractionMatrix(lambP, L = L, chi = chi, s =s )
        U = lambda lamb: Ufunc(lamb)
        def UTest(lamb, U=U):
            UArr = U(lamb)
            UTest = np.abs(UArr[0,0])**2 + np.abs(UArr[0,2])**2 - np.abs(UArr[0,1])**2 - np.abs(UArr[0,3])**2
            return UTest
        
        llamb = np.linspace(lambMin, lambMax, numPoints)

        UU = llamb.copy()
        for i in range(len(llamb)):
            UU[i] = UTest(llamb[i])
        print(UU)
        fig, ax = plot_custom(
            3, 2, r"wavelength, $\lambda$ (um)", r"sanity check", equal=False, labelPad = 25
        )
        
        ax.plot(llamb*1e6, UU)
        # ax.set_ylim([0.5,1.5])

        
    def calcSpectrumSPDCTrans(lamb, lambP, L = 10.1e-6, s = 1, chi = 1E-10, theta = 0, etalonBool = True):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L, s = s, chi=chi, etalonBool = etalonBool, gainBool = False)(lamb)
        # UIm = lambda lamb: np.imag(SPDCResonator.calcInteractionMatrix(lambP, L = L)(lamb))
        # URe = lambda lamb: np.real(SPDCResonator.calcInteractionMatrix(lambP, L = L)(lamb))
        # UAb = lambda lamb: np.absolute(U(lamb))
        # gain = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L, s = s, chi=chi, etalonBool = etalonBool, gainBool = True)(lamb)
        # plt.plot(np.imag(gain(lamb)))
        # plt.plot(np.real(gain(lamb)))
        # print(gain(lamb))
        def St(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return  (
                UAb[1,0]**2 * (UAb[0,0]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                      UAb[1,2]**2 * (UAb[0,2]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                      2 * np.real(UArr[0,0]*UArr[1,2]*np.conjugate(UArr[1,0])*np.conjugate(UArr[0,2]))
                     )
        
        if not(isinstance(lamb, np.ndarray)):
            return St(lamb)
        
        else:
            SSt = np.zeros_like(lamb)
            
            if lamb.ndim == 1:  # 1D array
                for i in range(lamb.shape[0]):
                    SSt[i] = St(lamb[i])
            
            elif lamb.ndim == 2:  # 2D 
                for i in range(lamb.shape[0]):
                    for j in range(lamb.shape[1]):
                        SSt[i,j] = St(lamb[i,j])
                        
            return SSt
            
    
    def calcSpectrumSPDCRefl(lamb, lambP, L = 10.1e-6, s =1, chi = 1E-10, theta = 0):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L,s = s, chi=chi)(lamb)
        
        def Sr(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (UAb[3,0]**2 * (UAb[2,0]**2 + UAb[2,1]**2 + UAb[2,3]**2) + 
                    UAb[3,2]**2 * (UAb[2,2]**2 + UAb[2,1]**2 + UAb[2,3]**2) + 
                    2 * np.real(UArr[2,0]*UArr[3,2]*np.conjugate(UArr[3,0])*np.conjugate(UArr[2,2]))
                    )
        
        if not(isinstance(lamb, np.ndarray)):
            return Sr(lamb)
        
        else:
            SSr = np.zeros_like(lamb)
            
            if lamb.ndim == 1:  # 1D array
                for i in range(lamb.shape[0]):
                    SSr[i] = Sr(lamb[i])
            
            elif lamb.ndim == 2:  # 2D 
                for i in range(lamb.shape[0]):
                    for j in range(lamb.shape[1]):
                        SSr[i,j] = Sr(lamb[i,j])
                        
            return SSr
    
    def calcSpectrumSPDCTransRefl(lamb, lambP, L = 10.1e-6, s =1, chi = 1E-10, theta = 0):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L,s = s, chi=chi)(lamb)
        
        def Stsri(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (UAb[3,0]**2 * (UAb[0,0]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                    UAb[3,2]**2 * (UAb[0,2]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                    2 * np.real(UArr[0,0]*UArr[3,2]*np.conjugate(UArr[3,0])*np.conjugate(UArr[0,2]))
                    )
        def Stirs(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (UAb[1,0]**2 * (UAb[2,0]**2 + UAb[2,1]**2 + UAb[2,3]**2) + 
                    UAb[1,2]**2 * (UAb[2,2]**2 + UAb[2,1]**2 + UAb[2,3]**2) +
                    2 * np.real(UArr[2,0]*UArr[1,2]*np.conjugate(UArr[1,0])*np.conjugate(UArr[2,2]))
                    )
        
        def StrCoh(lamb):
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (
                  np.real(np.conjugate(UAb[3,0])*np.conjugate(UAb[2,0])*UAb[0,0]*UAb[1,0]) 
                + np.real(np.conjugate(UAb[3,0])*np.conjugate(UAb[2,2])*UAb[0,0]*UAb[1,2]) 
                + np.real(np.conjugate(UAb[3,0])*np.conjugate(UAb[2,1])*UAb[1,0]*UAb[0,1]) 
                + np.real(np.conjugate(UAb[3,0])*np.conjugate(UAb[2,3])*UAb[1,0]*UAb[0,3]) 
                + np.real(np.conjugate(UAb[3,2])*np.conjugate(UAb[2,1])*UAb[0,1]*UAb[1,2]) 
                + np.real(np.conjugate(UAb[3,2])*np.conjugate(UAb[2,0])*UAb[0,2]*UAb[1,0]) 
                + np.real(np.conjugate(UAb[3,2])*np.conjugate(UAb[2,2])*UAb[0,2]*UAb[1,2]) 
                + np.real(np.conjugate(UAb[3,2])*np.conjugate(UAb[2,3])*UAb[1,2]*UAb[0,3]) 
                )
            
        
        if not(isinstance(lamb, np.ndarray)):
            
            Sfb = lambda lamb: Stsri(lamb)
            Sbf = lambda lamb: Stirs(lamb)
            Scoh = lambda lamb: StrCoh(lamb)
            
            return Sfb(lamb) + Sbf(lamb) + Scoh(lamb)

        else:
            SSfb = np.zeros_like(lamb)
            SSbf = np.zeros_like(lamb)
            SScoh = np.zeros_like(lamb)
            
            if lamb.ndim == 1:  # 1D array
                for i in range(lamb.shape[0]):
                    SSfb[i] = Stsri(lamb[i])
                    SSbf[i] = Stirs(lamb[i])
                    SScoh[i] = StrCoh(lamb[i])
            
            elif lamb.ndim == 2:  # 2D 
                for i in range(lamb.shape[0]):
                    for j in range(lamb.shape[1]):
                        
                        SSfb[i,j] = Stsri(lamb[i,j])
                        SSbf[i,j] = Stirs(lamb[i,j])
                        SScoh[i,j] = StrCoh(lamb[i,j])
                        
            return SSfb + SSbf #+ SScoh
        

        
        
    
    def calcPumpEnhancement(lambP = 788e-9, L = 10e-6):
        
        def calct(n1, n2):
            return 2*np.sqrt(n1 * n2)/(n1 + n2)
            
        def calcr(t, phase):
            return np.sqrt(1-t**2) * np.exp(1j*phase)
        
        npump = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        n2 = lambda lamb: silicon.sellmaierReal(lamb*1e6)
        n1 = lambda lamb: 1
        
        kp = lambda lamb: optics.lamb2k(lamb, npump(lamb))
        
        
        deltap = lambda lamb: L*kp(lamb) #this is the propagation phase for the pump
        
        t1 = lambda lamb: calct(n1(lamb), npump(lamb))
        t2 = lambda lamb: calct(npump(lamb), n2(lamb))
        r1 = lambda lamb: calcr(t1(lamb), deltap(lamb))
        r2 = lambda lamb: calcr(t2(lamb), deltap(lamb) - np.pi)
        
        PEForwards = t1(lambP)/(1-r1(lambP)*r2(lambP))
        PEBacks =  PEForwards*r2(lambP)
        
        # r1s = lambda lamb: calcr(t1s(lamb), deltas(lamb))
        # r2s = lambda lamb: calcr(t2s(lamb), deltas(lamb) - np.pi)
        
        # E0p = t1s(lambP)/(1 - r1s(lambP)*r2s(lambP));
        # E0m = E0p*r2s(lambP);
        
        return PEForwards, PEBacks
    
    def calcEtalonTrans(lamb, L = 10e-6, theta = 0,interfaceType = 2):
        # etalonType: 2: air -> LN -> silicon ; 1: custom; 3: air -> GaP -> silica; 4: GaP -> silica -> sapphire
            
        
        def calct(n1, n2):
            return 2*np.sqrt(n1*n2)/(n1 + n2)
            
        def calcr(n1, n2, phase):
            return (n1 - n2)/(n1 + n2) * np.exp(1j*phase)
        
        if interfaceType == 1:
            ns = lambda lamb: 3.1
            n2 = lambda lamb: 1
            n1 = lambda lamb: 1.7
        elif interfaceType == 2:
            ns = lambda lamb: lithiumNiobate.EIndexDisp(lamb) #s for sig/idler in substrate
            n2 = lambda lamb: silicon.sellmaierReal(lamb*1e6)
            n1 = lambda lamb: 1
        elif interfaceType == 3:
            ns = lambda lamb: galliumPhosphide.sellmaierReal(lamb*1e6) #s for sig/idler in substrate
            n2 = lambda lamb: 1.7
            n1 = lambda lamb: 1
        elif interfaceType == 4:
            ns = lambda lamb: 1.7 #s for sig/idler in substrate
            n2 = lambda lamb: 2
            n1 = lambda lamb: 1
        
        
        ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))*np.cos(theta)
        
        deltas = lambda lamb: L*ks(lamb)
        
        t01 = lambda lamb: calct(n1(lamb), ns(lamb))
        t12 = lambda lamb: calct(ns(lamb), n2(lamb))
        r01 = lambda lamb: calcr(n1(lamb), ns(lamb), deltas(lamb))
        r12 = lambda lamb: calcr(ns(lamb), n2(lamb), deltas(lamb))
        t10 = lambda lamb: calct(ns(lamb), n1(lamb))
        t21 = lambda lamb: calct(n2(lamb), ns(lamb))
        r10 = lambda lamb: calcr(ns(lamb), n1(lamb), deltas(lamb))
        r21 = lambda lamb: calcr(n2(lamb), ns(lamb), deltas(lamb))
        
        FP = lambda lamb: 1 / (1-r10(lamb)*r12(lamb)) # etalon enhancement
        
        return FP(lamb)#*np.abs(t12(lamb)*t01(lamb))**2
    
    def calcEtalonEffect(lamb, lambP = 788e-9, L = 10e-6, theta = 0, interferenceBool = False, detEffTrans = 1, detEffRefl = 1, collectionBool = False, interfaceType = 2, envelopeFunc = 1):
        # etalonType: 2: air -> LN -> silicon ; 1: custom; 3: air -> GaP -> silica; 4: GaP -> silica -> sapphire
        if collectionBool == True:
            if isinstance(envelopeFunc, (int)):
                totalEfficiency = SPDCSpectrumDetection.totalEff(lamb)
                totalEfficiencyHalf = SPDCSpectrumDetection.totalEffHalf(lamb)
            else:
                totalEfficiency = envelopeFunc
                totalEfficiencyHalf = envelopeFunc
            
        else:
            totalEfficiency = 1
            totalEfficiencyHalf = 1
        
        # def calct(n1, n2, phase, theta):
        #     return 2*np.sqrt(n1*n2)/(n1 + n2)
            
        # def calcr(n1, n2, phase, theta):
        #     return (n1 - n2)/(n1 + n2) * np.exp(1j*phase)
        
        # def calcr(n1, n2, phase, theta):
        #     return ((n1*np.cos(theta) - n2*np.sqrt(1-n1/n2*np.sin(theta)))/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))) * np.exp(1j*(phase))
            
        # def calct(n1, n2, phase, theta):
        #     r = calcr(n1, n2, phase, theta)
        #     return 2*np.sqrt(n1*n2)*np.cos(theta)/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))
        
        def calct(n1, n2, phase, theta):
            return 2*np.sqrt(n1 * n2)*np.cos(theta)/(n1*np.cos(theta) + n2*np.sqrt(1-n1/n2*np.sin(theta)))
            
        def calcr(n1, n2, phase, theta):
            t = calct(n1, n2, phase, theta)
            return np.sqrt(1-t**2) * np.exp(1j*phase)
        
        npump = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        lambConj = lambda lamb: 1/(1/lambP - 1/lamb)
        
        if interfaceType == 1:
            ns = lambda lamb: 3.1
            n2 = lambda lamb: 1
            n1 = lambda lamb: 1.7
        elif interfaceType == 2:
            ns = lambda lamb: lithiumNiobate.EIndexDisp(lamb) #s for sig/idler in substrate
            n2 = lambda lamb: silicon.sellmaierReal(lamb*1e6)
            n1 = lambda lamb: 1
        elif interfaceType == 3:
            ns = lambda lamb: galliumPhosphide.sellmaierReal(lamb*1e6) #s for sig/idler in substrate
            n2 = lambda lamb: 1.7
            n1 = lambda lamb: 1
        elif interfaceType == 4:
            ns = lambda lamb: 1.7*(lamb*1e6) #s for sig/idler in substrate
            n2 = lambda lamb: 2
            n1 = lambda lamb: 1
        
        
        # ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))*np.cos(theta)
        # kp = optics.lamb2k(lambP, npump(lambP))
        
        # deltas = lambda lamb: L*ks(lamb)
        
        # thetai = lambda lamb: (np.arcsin(-ks(lamb)*np.sin(theta)
        #                     /(ks(lambConj(lamb)))
        #                     ) 
        #           )
        
        
        ki = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))
        ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))
        kp = optics.lamb2k(lambP, npump(lambP))
        
        thetai = lambda lamb: -np.arctan(np.abs(ks(lamb))*np.sin(theta)/(np.abs(kp)-np.abs(ks(lamb))*np.cos(theta)))
        # print(thetai(lamb))
        
        deltas = lambda lamb: L*ks(lamb)*np.cos(theta)
        deltai = lambda lamb: L*ki(lamb)*np.cos(thetai(lamb))
        
        # t01 = lambda lamb: calct(n1(lamb), ns(lamb))
        # t12 = lambda lamb: calct(ns(lamb), n2(lamb))
        # r01 = lambda lamb: calcr(n1(lamb), ns(lamb), deltas(lamb))
        # r12 = lambda lamb: calcr(ns(lamb), n2(lamb), deltas(lamb))
        # t10 = lambda lamb: calct(ns(lamb), n1(lamb))
        # t21 = lambda lamb: calct(n2(lamb), ns(lamb))
        # r10 = lambda lamb: calcr(ns(lamb), n1(lamb), deltas(lamb))
        # r21 = lambda lamb: calcr(n2(lamb), ns(lamb), deltas(lamb))
        
        # FP = lambda lamb: 1 / (1-r10(lamb)*r12(lamb)) # etalon enhancement
        
        # PEForwards, PEBacks = SPDCResonator.calcPumpEnhancement(lambP, L)
        
        # TTForwards = (PEForwards  # Pp->*Ps->Pi->
        #                     *t12(lamb)*FP(lamb)
        #                     *t12(lambConj(lamb))*FP(lambConj(lamb)))
        # TTBacks = (PEBacks        # Pp<-*Ps->Pi-> 
        #                  *(r10(lamb))*t12(lamb)*FP(lamb)
        #                  *(r10(lambConj(lamb)))*t12(lambConj(lamb))*FP(lambConj(lamb)))
        
        # RRForwards = (PEForwards
        #                     *r12(lamb)*t10(lamb)*FP(lamb)
        #                     *r12(lambConj(lamb))*t10(lambConj(lamb))*FP(lambConj(lamb)))
        # RRBacks = (PEBacks
        #                     *t10(lamb)*FP(lamb)
        #                     *t10(lambConj(lamb))*FP(lambConj(lamb)))
        
        # rtForwards = (PEForwards
        #               *r12(lamb)*(t10(lamb))*FP(lamb)
        #               *t12(lambConj(lamb))*FP(lambConj(lamb)))
        # rtBacks = (PEBacks
        #            *(t10(lamb))*FP(lamb)
        #            *(r10(lambConj(lamb)))*t12(lambConj(lamb))*FP(lambConj(lamb)))
        
        # trForwards = (PEForwards
        #               *t12(lamb)*FP(lamb)
        #               *r12(lambConj(lamb))*(t10(lambConj(lamb)))*FP(lambConj(lamb)))
        # trBacks = (PEBacks
        #            *(r10(lamb))*t12(lamb)*FP(lamb)
        #            *(t10(lambConj(lamb)))*FP(lambConj(lamb)))
        
        # t01 = lambda lamb, theta, delta: calct(n1(lamb), ns(lamb), delta, theta)
        t12 = lambda lamb, theta, delta: calct(ns(lamb), n2(lamb), delta, theta)
        # r01 = lambda lamb, theta, delta: calcr(n1(lamb), ns(lamb), delta, theta)
        r12 = lambda lamb, theta, delta: calcr(ns(lamb), n2(lamb), delta+np.pi, theta)
        t10 = lambda lamb, theta, delta: calct(ns(lamb), n1(lamb), delta, theta)
        # t21 = lambda lamb, theta, delta: calct(n2(lamb), ns(lamb), delta, theta)
        r10 = lambda lamb, theta, delta: calcr(ns(lamb), n1(lamb), delta, theta)
        # r21 = lambda lamb, theta, delta: calcr(n2(lamb), ns(lamb), delta, theta)
        
        
        FP = lambda lamb, theta, delta: 1 / (1-r10(lamb,theta, delta)*r12(lamb,theta, delta)) # etalon enhancement
        
        PEForwards, PEBacks = SPDCResonator.calcPumpEnhancement(lambP, L)
        
        TTForwards = (PEForwards  # Pp->*Ps->Pi->
                            *t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                            *t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        TTBacks = (PEBacks        # Pp<-*Ps->Pi-> 
                         *(r10(lamb, theta, deltas(lamb)))*t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                         *(r10(lambConj(lamb), thetai(lamb), deltai(lamb)))*t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        
        RRForwards = (PEForwards
                            *r12(lamb, theta, deltas(lamb))*t10(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                            *r12(lambConj(lamb), thetai(lamb), deltai(lamb))*t10(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        RRBacks = (PEBacks
                            *t10(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                            *t10(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        
        rtForwards = (PEForwards
                      *r12(lamb, theta, deltas(lamb))*(t10(lamb, theta, deltas(lamb)))*FP(lamb, theta, deltas(lamb))
                      *t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        rtBacks = (PEBacks
                   *t10(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                   *r10(lambConj(lamb), thetai(lamb), deltai(lamb))*t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        
        trForwards = (PEForwards
                      *t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                      *r12(lambConj(lamb), thetai(lamb), deltai(lamb))*(t10(lambConj(lamb), thetai(lamb), deltai(lamb)))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        trBacks = (PEBacks
                   *(r10(lamb, theta, deltas(lamb)))*t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                   *(t10(lambConj(lamb), thetai(lamb), deltai(lamb)))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        
        if interferenceBool is False:
            TTs = (np.abs(TTForwards)**2 + np.abs(TTBacks)**2) * detEffTrans**2
            RRs = (np.abs(RRForwards)**2 + np.abs(RRBacks)**2) * detEffRefl**2
            RTs = (np.abs(rtForwards)**2 + np.abs(rtBacks)**2) * detEffTrans*detEffRefl
            TRs = (np.abs(trForwards)**2 + np.abs(trBacks)**2) * detEffTrans*detEffRefl
            TRRTs = TRs + RTs
            TRRTs = TRs
            
        else: 
            TTs = np.abs(TTForwards+TTBacks)**2 #* detEffTrans**2
            RRs = np.abs(RRForwards+RRBacks)**2 #* detEffRefl**2
            RTs = np.abs(rtForwards+rtBacks)**2 #* detEffTrans*detEffRefl
            TRs = np.abs(trForwards+trBacks)**2 #* detEffTrans*detEffRefl
            
            # TTs = (np.abs(FP(lamb)*FP(lambConj(lamb))*FP(lambP))**2
            #        *np.abs(t12(lambConj(lamb))*t12(lamb))**2
            #        *np.abs(1 + r12(lambP)*r10(lamb)*r10(lambConj(lamb)))**2
            #        )#*np.abs(1 + r12(lambP))**2) #* detEffTrans**2
            # RRs = (np.abs(FP(lamb)*FP(lambConj(lamb))*FP(lambP))**2
            #        *np.abs(t10(lambConj(lamb))*t10(lamb))**2
            #        *np.abs(r12(lamb)*r12(lambConj(lamb)) + r12(lambP))**2
            #        ) #* detEffRefl**2
            # RTs = np.abs(FP(lamb)*FP(lambConj(lamb))*
            #               (PEForwards*r12(lamb)*t10(lamb) + PEBacks*t10(lamb))*
            #               (PEForwards*t12(lambConj(lamb)) 
            #               + PEBacks*r10(lambConj(lamb))*t12(lambConj(lamb))))**2 #* detEffTrans*detEffRefl
            # TRs = (np.abs(FP(lamb)*FP(lambConj(lamb))*FP(lambP))**2
            #        *np.abs(t10(lambConj(lamb))*t12(lamb))**2
            #        *np.abs(1*r12(lambConj(lamb)) + r12(lambP)*r10(lamb))**2
            #        )#*np.abs(1 + r12(lambP))**2) #* detEffTrans**2
            TRRTs = TRs + RTs
            # TRRTs = TRs
            # TRRTs = (np.abs(rtForwards+rtBacks + trForwards+trBacks)**2) * (detEffTrans*detEffRefl)**2
        
        return TTs*totalEfficiency, RRs*totalEfficiency, TRRTs*totalEfficiencyHalf
        
    def plotSpectraEtalon(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6, theta = 0, normBool = False, 
                                 interferenceBool = False, emissionProbBool = True, s = 1, chi = 1E-10, 
                                 detEffTrans = 1, detEffRefl = 1, emissionConjBool = False, interfaceType = 2):
        colorArray = thesisColorPalette(3, paletteType = "smooth")
        
        llamb = np.linspace(lambMin, lambMax, numPoints)
        llambConj = 1/(1/lambP - 1/ llamb)
        w = 1/llamb
        TT, RR, TRRT = SPDCResonator.calcEtalonEffect(llamb, lambP, L, theta = theta, interferenceBool = interferenceBool, interfaceType = interfaceType, detEffTrans = detEffTrans, detEffRefl = detEffRefl)
        # TTc, RRc, TRRTc = SPDCResonator.calcEtalonEffect(llambConj, lambP, L, interferenceBool = interferenceBool, detEffTrans = detEffTrans, detEffRefl = detEffRefl)
        
        if emissionProbBool is True:
            wp = optics.lamb2omega(lambP)
            ws = optics.lamb2omega(llamb)
            theta = np.linspace(0, 0,1)+theta
            # theta = np.linspace(-np.pi/10000, np.pi/10000,1)
            ttheta,llambs = np.meshgrid(theta, llamb, indexing='ij');
            _,llambsConj = np.meshgrid(theta, llambConj, indexing='ij');
            wws = optics.lamb2omega(llambs)
            wwsc = optics.lamb2omega(llambsConj)
            
            FAS = np.sum(np.nan_to_num(np.abs(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L, s = s, chi = chi))), axis = 0)
            # FASConj = np.sum(np.nan_to_num(np.abs(SPDCalc.calcFreqAngSpectrum(wp, wwsc, ttheta, L))), axis = 0)
            
            TT *= FAS
            RR *= FAS
            TRRT *= FAS
            
            # TTc *= FASConj
            # RRc *= FASConj
            # TRRTc *= FASConj
        
        # if emissionConjBool is True:
        #     TT = np.sqrt(TT*TTc)
        #     RR = np.sqrt(RR*RRc)
        #     TRRT = np.sqrt(TRRT*TRRTc)
        
        fig, ax = plot_custom(
            0.6*textwidth, 0.4*textwidth, r"wavelength, $\lambda$ (um)", r"spectrum", equal=False, labelPad = labelfontsize
        )
        if normBool is True:
            # ax.plot(w, TT/max(TT), linestyle = '-', color = colorArray[0])
            # ax.plot(w, RR/max(RR), linestyle = '--', color = colorArray[1])
            # ax.plot(w, TRRT/max(TRRT), linestyle = ':', color = colorArray[2])
            ax.plot(llamb*1e6, TT/max(TT), linestyle = '-', color = colorArray[0])
            ax.plot(llamb*1e6, RR/max(RR), linestyle = '--', color = colorArray[1])
            ax.plot(llamb*1e6, TRRT/max(TRRT), linestyle = ':', color = colorArray[2])
        else:
            ax.plot(llamb*1e6, TT/max(TT), linestyle = '-', color = colorArray[0])
            ax.plot(llamb*1e6, RR/max(TT), linestyle = '--', color = colorArray[1])
            ax.plot(llamb*1e6, TRRT/max(TT), linestyle = ':', color = colorArray[2])
        # print(max(TT))
        ax.set_ylim([0,None])           
        # ax.legend(['trans', 'refl', 'both'])
        ax.set_title("$L = $" + f"{L*1e6:.2f}" + " um", fontsize = labelfontsize)
        
    
    def plotSpectra(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6, theta = 0, normBool = True, chi = 1e-10, s = 1):
        colorArray = thesisColorPalette(3, paletteType = "smooth")

        llamb = np.linspace(lambMin, lambMax, numPoints)
        
        
        # for i in range(len(llamb)):
            # St[i] = SPDCResonator.calcSpectrumSPDCTrans(llamb[i], lambP, L = L, s= s, chi = chi)
        Sr = SPDCResonator.calcSpectrumSPDCRefl(llamb, lambP, L = L, s= s, chi = chi, theta = theta)
        Str = SPDCResonator.calcSpectrumSPDCTransRefl(llamb, lambP, L = L, s=s, chi = chi, theta = theta)
        St = SPDCResonator.calcSpectrumSPDCTrans(llamb, lambP, L = L, s= s, chi = chi, theta = theta)
            
            
        fig, ax = plot_custom(
            0.6*textwidth, 0.4*textwidth, r"wavelength, $\lambda$ (um)", r"spectrum", equal=False, labelPad = labelfontsize
        )
        if normBool is True:
            ax.plot(llamb*1e6, St/max(St), color = colorArray[0])
            # ax.plot(llamb*1e6, Sr/max(Sr), color = colorArray[1])
            # ax.plot(llamb*1e6, Str/max(Str), color = colorArray[2])
            # ax.plot(w, St/max(St), color = colorArray[0])
            # ax.plot(w, Sr/max(Sr), color = colorArray[1])
            # ax.plot(w, Str/max(Str), color = colorArray[2])
        else:
            ax.plot(llamb*1e6, St/max(St), color = colorArray[0])
            ax.plot(llamb*1e6, Sr/max(St), color = colorArray[1])
            ax.plot(llamb*1e6, Str/max(St), color = colorArray[2])
        # ax.set_ylim([0.5,1.5])
        ax.set_ylim([0,None])  
        ax.set_title("$L = $" + f"{L*1e6:.2f}" + " um", fontsize = labelfontsize)
        # ax.legend(['trans', 'refl', 'both'])

class fiberDispersion:
    
    def import_excel_data(file_path):
        """Import data from the Excel file and return as pandas DataFrame."""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error occurred while importing the Excel file: {e}")
            return None
        
    def create_interpolating_extrapolating_function(df, x_column, y_column):
        """Create an interpolating and extrapolating function using scipy's interp1d."""
        try:
            x = df[x_column]
            y = df[y_column]
    
            # Sort the data in case it is not already sorted
            sorted_indices = x.argsort()
            x_sorted = x.iloc[sorted_indices]
            y_sorted = y.iloc[sorted_indices]
    
            # Create the interpolation function
            interpolating_extrapolating_func = interp1d(x_sorted, y_sorted, kind='linear',
                                                       fill_value='extrapolate')
    
            return interpolating_extrapolating_func
        except Exception as e:
            print(f"Error occurred while creating the interpolating and extrapolating function: {e}")
            return None
        
    def create_interpolating_extrapolating_function_Array(x, y):
        """Create an interpolating and extrapolating function using scipy's interp1d."""
        try:    
    
            # Create the interpolation function
            interpolating_extrapolating_func = interp1d(x, y, kind='linear',
                                                       fill_value='extrapolate')
    
            return interpolating_extrapolating_func
        except Exception as e:
            print(f"Error occurred while creating the interpolating and extrapolating function: {e}")
            return None
        
    def interpolateFilterData(filename):
        
        filenameFilter = str(dirCurr) + "\\refractiveIndex\\"  + filename
    
        # Replace 'x_column_name' and 'y_column_name' with the names of the two columns in your Excel file.
        x_column_name = 'Wavelength (nm)'
        y_column_name = '% Transmission'
    
        # Import data from the Excel file
        data_df = fiberDispersion.import_excel_data(filenameFilter)
    
        if data_df is not None:
            # Create the interpolating and extrapolating function
            transInterp = fiberDispersion.create_interpolating_extrapolating_function(data_df, x_column_name, y_column_name)
    
        return transInterp
    
    def interpolateDetectorData():
        
        filename = str(dirCurr) + "\\SNSPD\\detectorEfficiency.csv"
    
        # Import data from the Excel file
        data = genfromtxt(filename, delimiter=',')
        lamb = data[:, 0]
        eff = data[:, 1]
    
        if data is not None:
            # Create the interpolating and extrapolating function
            transInterp = fiberDispersion.create_interpolating_extrapolating_function_Array(x = lamb, y = eff)
    
        return transInterp
    
    
    
    def funcDetectionEff(lambP=0.788e-6, half = False):
        lambConj = lambda lamb, lambP: 1/(1/lambP - 1/lamb)
        
        TLP1400 = lambda lamb: SPDCSpectrumDetection.FEL1400(lamb)
        TLP1500 = lambda lamb: SPDCSpectrumDetection.FEL1500(lamb)
        TLP = TLP1400
        TFS = 0.8
        etaFC = 0.5
        etaBS = 0.4
        if half:
            TFL = lambda lamb: 0.8 * ( np.arctan(15*(lamb-1.2))/np.pi - np.arctan(15 * (lamb - 1.680))/np.pi + 0.03)
            Gamma = 2
            lamb0 = 1.700
            TFL = lambda lamb: 1/np.pi * (0.5*Gamma)/((lamb - lamb0)**2 - (0.5*Gamma)**2)
            sig = 0.14
            center = 2*0.788
            TFL = lambda lamb: 1/(sig*np.sqrt(2*np.pi))*np.exp(-(lamb-center)**2/(2*sig**2))
        else:
            TFL = lambda lamb: (0.8 * ( np.arctan(5*(lamb-1.3))/np.pi - np.arctan(5 * (lamb - 1.680))/np.pi + 0.03))**2
            sig = 0.14
            center = 2*0.788
            TFL = lambda lamb: 1/(sig*np.sqrt(2*np.pi))*np.exp(-(lamb-center)**2/(2*sig**2))
        # llamb = np.linspace(1.1e-6, 1.9e-6, 100)
        # plt.plot(llamb*1e6, TFL(llamb*1e6))
        
        totalDetEff = lambda lamb: ((TFS * etaFC * etaBS)**2 * TFL(lamb*1e6) * TFL(lambConj(lamb, lambP)*1e6) * TLP(lamb) * TLP(lambConj(lamb, lambP)) 
                                    * SPDCSpectrumDetection.detEff(lamb) * SPDCSpectrumDetection.detEff(lambConj(lamb, lambP)))
        # fig, ax = plot_custom(
        #     9, 6, r"wavelength, $\lambda$ (um)", r"sanity check", equal=False, labelPad = 25
        # )
        
        # llamb = np.linspace(1.1e-6, 1.9e-6, 100)
        # eff =  totalDetEff(llamb)
        # plt.plot(llamb*1e6, eff/np.max(eff))      
        # ax.set_ylim(-0.05, 1.05)                      
        
        return totalDetEff
    
        
class SPDCSpectrumDetection:
    
    FEL1400f = fiberDispersion.interpolateFilterData('FEL1400.xlsx')
    FEL1500f = fiberDispersion.interpolateFilterData('FEL1500.xlsx')
    detEfff = fiberDispersion.interpolateDetectorData()
    totalEfff = fiberDispersion.funcDetectionEff(half=False)
    totalEfffHalf = fiberDispersion.funcDetectionEff(half=True)
    
    FEL1400 = lambda lamb: SPDCSpectrumDetection.FEL1400f(lamb*1e9)/100
    FEL1500 = lambda lamb: SPDCSpectrumDetection.FEL1500f(lamb*1e9)/100
    detEff = lambda lamb: SPDCSpectrumDetection.detEfff(lamb*1e6)
    totalEff = lambda lamb: SPDCSpectrumDetection.totalEfff(lamb)
    totalEffHalf = lambda lamb: SPDCSpectrumDetection.totalEfffHalf(lamb)
    
        
    def calcTotalEfficiencyArrays(lambMin=1.4e-6, lambMax=1.8e-6, numPoints=100, lambP = 0.788e-6, L = 10.1e-6):
        
        llamb = np.linspace(lambMin, lambMax, numPoints)
        
        totalEfficiency = SPDCSpectrumDetection.totalEff(llamb)
        totalEfficiencyHalf = SPDCSpectrumDetection.totalEffHalf(llamb)
        
        St = llamb.copy()
        Sr = llamb.copy()
        Str = llamb.copy()
        
        
        for i in range(len(llamb)):
            St[i] = SPDCResonator.calcSpectrumSPDCTrans(llamb[i], lambP, L = L)
            Sr[i] = SPDCResonator.calcSpectrumSPDCRefl(llamb[i], lambP, L = L)
            Str[i] = SPDCResonator.calcSpectrumSPDCTransRefl(llamb[i], lambP, L = L)
            
        St *= totalEfficiency
        Sr *= totalEfficiency
        Str *= totalEfficiencyHalf
        
        return llamb, St, Sr, Str, L
        
        
    def plotTotalEfficiencyArrays(llamb, St, Sr, Str, L):
        
        fig, ax = plot_custom(
            0.6*textwidth, 0.4*textwidth, r"wavelength, $\lambda$ (um)", r"spectrum", equal=False, labelPad = 25
        )
            
        ax.plot(llamb*1e6, St/np.max(St))
        ax.plot(llamb*1e6, Sr/np.max(St))
        ax.plot(llamb*1e6, Str/np.max(St))
        # ax.set_ylim([0.5,1.5])
        ax.set_title("$L = $" + f"{L*1e6:.2f}" + " um", fontsize = 25)
        ax.legend(['trans', 'refl', 'both'])
        
    
        


class delay2Wavelength:
    
    def createIndex2lambFunction():
        
        def fit_linear_function(x_data, y_data):
            # Perform linear regression using numpy.polyfit
            slope, intercept = np.polyfit(x_data, y_data, deg=1)
            return slope, intercept
        
        lambP = 0.788e-6
        data = np.array([[1.310e-6, 1.4652],[1.550e-6, 1.4682]])
        slope, intercept = fit_linear_function(data[:, 0], data[:, 1])
        
        nFiber = lambda lamb: slope*lamb + intercept
        
        delTMean = 14.76e3 * 1e-9 # s
        lFiber = delTMean * optics.c / nFiber(2*lambP)
        
        delTRel = lambda lamb: lFiber * nFiber(lamb)/optics.c
        
        nFiber2lamb = lambda n: (n-intercept)/slope 
        delT2nFiber = lambda delT: delT * optics.c / lFiber
        
        return nFiber2lamb, delT2nFiber
    
    nFiber2lamb, delT2nFiber = createIndex2lambFunction()
    
    delT2lamb = lambda delT: delay2Wavelength.nFiber2lamb(delay2Wavelength.delT2nFiber(delT))
    
        # rangeT = 20e-9
        # ddelT = np.linspace(-rangeT, rangeT, 100)
        # plt.plot(ddelT, nFiber2lamb(nFiber(ddelT + delTMean)))
    
if __name__ == '__main__':        
    
    #%% plot reflectance, transmittance and phase of light transmitted through the LNSi sample
    waferLNSi.plotReflectanceAndPhase(400e-9, 1500e-9, np.pi*0.99, N=200)
    waferLNSi.plotTransmittanceAndPhase(400e-9, 1500e-9, np.pi*0.99, dist = 500e-6, N=200, absorbBool = True)
    # waferLNSi.plotTransmittance(400e-9, 1500e-9, np.pi*0.99, dist = 500e-6, N = 200, absorbBool = True)
    # waferLNSi.plotReflectionPhase(400e-9, 1500e-9, np.pi*0.99, N = 300)
    
    #%% plot the frequency angular spectra of a LN crystal
    SPDCalc.plotFreqAngSpectrum(850e-9, 2500e-9, 788e-9, np.pi/2, L =10e-6)
    
    #%% plot holographic SPDC stuff
    # SPDCalc.plotAnglesFreqAngSpectrum(2*788e-9, 788e-9, np.pi, L=10e-6, Deltan = 0.2)
    # lambWrite = 532e-9
    # angleWriteExt = np.pi/6
    # lambPump = 405e-9
    # widthPump = 10e-6
    # Deltan = 6e-6
    # lambSig = 750e-9
    # SPDCalc.plotAngularCorrelationExperiment(lambWrite = lambWrite, angleWriteExt = angleWriteExt, 
    #                                           lambPump = lambPump, widthPump = widthPump,
    #                                           Deltan = Deltan, degBool = False, 
    #                                           lambSig = lambSig)
    # lambLow = 500e-9
    # lambHigh = 1000e-9
    # M = 400
    # N = M
    # SPDCalc.plotAngularCorrelationExperiment(lambWrite = lambWrite, angleWriteExt = angleWriteExt, 
    #                                           lambPump = lambPump, widthPump = widthPump,
    #                                           Deltan = Deltan, degBool = False, 
    #                                           lambSig = lambSig, lambLow = lambLow, N = N,
    #                                           lambHigh = lambHigh, wsRangeBool = True, M = M)
    # SPDCalc.plotPhaseMatchingVsNonPhaseMatching()
    
    #%% SPDC etalon/resonator calculations
    lambPump = 0.788e-6
    lambMin = 2*lambPump-0.7e-6#1.2e-6    #m
    lambMax  = 2*lambPump+3e-6#=1.5e-6     #m
    lambMax = 1/(1/lambPump-1/lambMin)
    L = 10.145e-6       #m
    # L = 1e-6
    chi = 34e-12        #m/V
    s = 0.0000125*1e-20/chi   #V/m    0.0125/chi has considerable gain, 0.001/chi does not.  
    theta = np.pi/180*1
    # checks that the quantum model is working (should be equal to 1 for all points)
    # SPDCResonator.validateModel(lambMin, lambMax, 100, 0.788e-6,s = s, chi = chi) 
    
    # plots spectra for collinear SPDC calculated using the simple     
    SPDCResonator.plotSpectraEtalon(lambMin, lambMax, 500, L = L, theta = theta, normBool = False, interfaceType=2,
                                            interferenceBool=True, emissionProbBool=True, s = s, chi = chi)
    # plots spectra for collinear SPDC calculated using the quantum model
    SPDCResonator.plotSpectra(lambMin, lambMax, 500, L = L, theta = theta, normBool = False, s = s, chi = chi)
    
    # SPDCResonator.plotSpectra(lambMin, lambMax, 300, L = 150e-9, normBool = True, s = s, chi = chi)
    
    
    # SPDCResonator.plotSpectraEtalon(1100e-9, 1300e-9, 300, lambP = 0.594e-6, L = 150e-9, normBool = False, 
    #                                         interferenceBool=False, emissionProbBool=False, s = s, chi = chi)
    # lambMin = 0.594e-6
    # lambMax  = 2e-6#=1.5e-6     #m
    # lamb = np.linspace(lambMin, lambMax, 20000) 
    # T = SPDCResonator.calcEtalonTrans(lamb, L = L, theta = 0, interfaceType = 2)
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2))
    # ax.plot(lamb, T)
    # ax = plt.gca()
    # ax.set_ylabel("FP enhancement spectrum")
    # ax.set_xlabel("wavelength um")
    plt.show()
    
    
    # for metasurface
    # lambPump = 0.594e-6
    # lambMin = 2*lambPump-0.3e-6#1.2e-6    #m
    # lambMax  = 2*lambPump+0.3e-6#=1.5e-6     #m
    # L = 0.155e-6       #m
    # chi = 34e-12        #m/V
    # s = 0.0000125/chi   #V/m    0.0125/chi has considerable gain, 0.001/chi does not.  
    # SPDCResonator.plotSpectraEtalon(lambMin, lambMax, 300, L = L, normBool = False, interfaceType=3,
    #                                         interferenceBool=True, emissionProbBool=True, s = s, chi = chi)
    
    # lambMin = 0.594e-6
    # lambMax  = 2e-6#=1.5e-6     #m
    # lamb = np.linspace(lambMin, lambMax, 20000)
    
    
    
    # T = SPDCResonator.calcEtalonTrans(lamb, L = L, theta = 0, interfaceType = 3)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2))
    # plt.plot(lamb, T)
    # ax = plt.gca()
    # ax.set_ylabel("FP enhancement spectrum")
    # ax.set_xlabel("wavelength um")
    # plt.show()
    # L = 4e-6       #m
    # chi = 34e-12        #m/V
    # s = 0.0000125/chi   #V/m    0.0125/chi has considerable gain, 0.001/chi does not.  
    # T = SPDCResonator.calcEtalonTrans(lamb, L = L, theta = 0, interfaceType = 4)
    # plt.plot(lamb, T)
    # plt.show()
    


