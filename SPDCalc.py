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
from plot_custom import plot_custom, thesisColorPalette
from numpy import genfromtxt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pathlib import Path
from datetime import datetime

dirCurr = Path.cwd()

# Constants for plotting
labelfontsize = 12
numfontsize = 10 
textwidth = 8.5 - 2 # in inches
lw = 1

class lithiumNiobate:
    indexE = str(dirCurr) + '\\indexData\\lithiumNiobate/Zelmon-e.csv'
    indexO = str(dirCurr) + '\\indexData\\lithiumNiobate/Zelmon-o.csv'
    
    # Dispersion formulae for extraordinary light
    @staticmethod
    def EIndexDisp(lamb):
        lambLocal = lamb * 1e6
        return np.sqrt(1 
                        + 2.9804 * lambLocal**2 / (lambLocal**2 - 0.02047) 
                        + 0.5981 * lambLocal**2 / (lambLocal**2 - 0.0666)
                        + 8.9543 * lambLocal**2 / (lambLocal**2 - 416.08))
    
    # Dispersion formulae for ordinary light
    @staticmethod
    def OIndexDisp(lamb):
        lambLocal = lamb * 1e6
        return np.sqrt(1 
                        + 2.6734 * lambLocal**2 / (lambLocal**2 - 0.01764) 
                        + 1.2290 * lambLocal**2 / (lambLocal**2 - 0.05914)
                        + 12.614 * lambLocal**2 / (lambLocal**2 - 474.60))

class BBO:
    # Dispersion formulae for extraordinary light
    @staticmethod
    def EIndexDisp(x):
        x = x * 1e6 # x is lamb
        return (1 + 1.151075 / (1 - 0.007142 / x**2) + 0.21803 / (1 - 0.02259 / x**2) + 0.656 / (1 - 263 / x**2))**.5

    # Dispersion formulae for ordinary light
    @staticmethod
    def OIndexDisp(x):
        x = x * 1e6 # x is lamb
        return (1 + 0.90291 / (1 - 0.003926 / x**2) + 0.83155 / (1 - 0.018786 / x**2) + 0.76536 / (1 - 60.01 / x**2))**.5

class silicon:
    dataDir = str(dirCurr) + "\\refractiveIndex\\silicon\\"

    @staticmethod
    def loadData():
        return genfromtxt(silicon.dataDir + 'green-2008.csv', delimiter=',')

    @staticmethod
    def sellmaier(lamb, B1, B2, B3, C1, C2, C3, A):
        return 1 + A + B1 * lamb**2 / (lamb**2 - C1) + B2 * lamb**2 / (lamb**2 - C2) + B3 * lamb**2 / (lamb**2 - C3) 

    @staticmethod
    def sell2Index(lamb, B1, B2, B3, C1, C2, C3, A):
        return np.sqrt(np.abs(silicon.sellmaier(lamb, B1, B2, B3, C1, C2, C3, A)))**2

    @staticmethod
    def fitIndexData(wl, index, poptGuess=[4, 10, 0, 0.05, 400, 0, -1]):
        popt, _ = curve_fit(silicon.sellmaier, wl, index, p0=poptGuess)
        return popt

    @staticmethod
    def loadIndexData():
        myData = silicon.loadData()
        myData = myData[myData[:, 0] >= 0.38]
        wl = myData[:, 0]
        n = myData[:, 1]
        k = myData[:, 2]
        return wl, n, k

    @staticmethod
    def findSiliconRegression():
        wl, n, k = silicon.loadIndexData()
        poptn = silicon.fitIndexData(wl, n)
        poptk = silicon.fitIndexData(wl, k, poptGuess=[0.012, 3000, 0, 0.1424, 4.65e6, 0, -1.02])
        return poptn, poptk

    @staticmethod
    def savePoptAsCSV():
        poptn, poptk = silicon.findSiliconRegression()
        np.savetxt(silicon.dataDir + "poptnk.csv", np.vstack([poptn, poptk]), delimiter=",")

    @staticmethod
    def sellmaierReal(wl):
        popt = genfromtxt(silicon.dataDir + 'poptnk.csv', delimiter=',')
        return silicon.sell2Index(wl, *popt[0, :])

    @staticmethod
    def sellmaierImag(wl):
        popt = genfromtxt(silicon.dataDir + 'poptnk.csv', delimiter=',')
        return silicon.sell2Index(wl, *popt[1, :])

    @staticmethod
    def interpIndex(wl, wlData, nData):
        return np.interp(wl, wlData, nData)

    @staticmethod
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

    @staticmethod
    def indexComplex(wl):
        return silicon.sellmaierReal(wl * 1e6) + 1j * silicon.sellmaierImag(wl * 1e6)

    @staticmethod
    def indexComplexInterp(wl):
        wlData, nData, kData = silicon.loadIndexData()
        return silicon.interpIndex(wl * 1e6, wlData, nData) + 1j * silicon.interpIndex(wl * 1e6, wlData, kData)

class waferLNSi:
    sPol = 'sPol'
    pPol = 'pPol'
    
    @staticmethod
    def cosThetat(n1, n2, thetai):
        """
        Calculate the cosine of the transmitted angle using Snell's law.
        
        Parameters:
        n1 (float): Refractive index of the first medium.
        n2 (float): Refractive index of the second medium.
        thetai (float): Incident angle in radians.
        
        Returns:
        float: Cosine of the transmitted angle.
        """
        return np.sqrt(1 - (n1 / n2 * np.sin(thetai))**2)
    
    @staticmethod
    def rpCalc(thetai, wl):
        """
        Calculate the reflection coefficient for p-polarized light.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Reflection coefficient for p-polarized light.
        """
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = n2 * np.cos(thetai) - n1 * cosThetat
        den = n2 * np.cos(thetai) + n1 * cosThetat
        return num / den
    
    @staticmethod
    def tpCalc(thetai, wl):
        """
        Calculate the transmission coefficient for p-polarized light.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Transmission coefficient for p-polarized light.
        """
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = 2 * n1 * np.cos(thetai)
        den = n2 * np.cos(thetai) + n1 * cosThetat
        return num / den
    
    @staticmethod
    def rsCalc(thetai, wl):
        """
        Calculate the reflection coefficient for s-polarized light.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Reflection coefficient for s-polarized light.
        """
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = n1 * np.cos(thetai) - n2 * cosThetat
        den = n1 * np.cos(thetai) + n2 * cosThetat
        return num / den
    
    @staticmethod
    def tsCalc(thetai, wl):
        """
        Calculate the transmission coefficient for s-polarized light.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Transmission coefficient for s-polarized light.
        """
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = 2 * n1 * np.cos(thetai)
        den = n1 * np.cos(thetai) + n2 * cosThetat
        return num / den
    
    @staticmethod
    def rpCalcAir(thetai, wl):
        """
        Calculate the reflection coefficient for p-polarized light at an air interface.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Reflection coefficient for p-polarized light at an air interface.
        """
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = n2 * np.cos(thetai) - n1 * cosThetat
        den = n2 * np.cos(thetai) + n1 * cosThetat
        return num / den
    
    @staticmethod
    def tpCalcAir(thetai, wl):
        """
        Calculate the transmission coefficient for p-polarized light at an air interface.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Transmission coefficient for p-polarized light at an air interface.
        """
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = 2 * n1 * np.cos(thetai)
        den = n2 * np.cos(thetai) + n1 * cosThetat
        return num / den
    
    @staticmethod
    def rsCalcAir(thetai, wl):
        """
        Calculate the reflection coefficient for s-polarized light at an air interface.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Reflection coefficient for s-polarized light at an air interface.
        """
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = n1 * np.cos(thetai) - n2 * cosThetat
        den = n1 * np.cos(thetai) + n2 * cosThetat
        return num / den
    
    @staticmethod
    def tsCalcAir(thetai, wl):
        """
        Calculate the transmission coefficient for s-polarized light at an air interface.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        
        Returns:
        complex: Transmission coefficient for s-polarized light at an air interface.
        """
        n1 = 1
        n2 = lithiumNiobate.EIndexDisp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        num = 2 * n1 * np.cos(thetai)
        den = n1 * np.cos(thetai) + n2 * cosThetat
        return num / den
    
    @staticmethod
    def reflectance(thetai, wl, pol):
        """
        Calculate the reflectance for a given polarization.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        pol (str): Polarization ('sPol' or 'pPol').
        
        Returns:
        float: Reflectance.
        """
        if pol == waferLNSi.sPol:
            r = waferLNSi.rsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            r = waferLNSi.rpCalc(thetai, wl)
        return np.abs(r)**2
    
    @staticmethod
    def angleReflectivity(thetai, wl, pol):
        """
        Calculate the phase angle of the reflectivity for a given polarization.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        pol (str): Polarization ('sPol' or 'pPol').
        
        Returns:
        float: Phase angle of the reflectivity.
        """
        if pol == waferLNSi.sPol:
            r = waferLNSi.rsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            r = waferLNSi.rpCalc(thetai, wl)
        return np.angle(r)
    
    @staticmethod
    def transmittance(thetai, wl, pol, dist):
        """
        Calculate the transmittance for a given polarization.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        pol (str): Polarization ('sPol' or 'pPol').
        dist (float): Distance in meters.
        
        Returns:
        float: Transmittance.
        """
        if pol == waferLNSi.pPol:
            t = waferLNSi.tpCalc(thetai, wl)
        elif pol == waferLNSi.sPol:
            t = waferLNSi.tsCalc(thetai, wl)
        
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        T = np.real(n2) * cosThetat / (n1 * np.cos(thetai)) * np.abs(t)**2
        return T
    
    @staticmethod
    def angleTrans(thetai, wl, pol):
        """
        Calculate the phase angle of the transmittance for a given polarization.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        pol (str): Polarization ('sPol' or 'pPol').
        
        Returns:
        float: Phase angle of the transmittance.
        """
        if pol == waferLNSi.sPol:
            t = waferLNSi.tsCalc(thetai, wl)
        elif pol == waferLNSi.pPol:
            t = waferLNSi.tpCalc(thetai, wl)
        return np.angle(t)
    
    @staticmethod
    def absorption(thetai, wl, dist):
        """
        Calculate the absorption.
        
        Parameters:
        thetai (float): Incident angle in radians.
        wl (float): Wavelength in meters.
        dist (float): Distance in meters.
        
        Returns:
        float: Absorption.
        """
        n1 = lithiumNiobate.EIndexDisp(wl)
        n2 = silicon.indexComplexInterp(wl)
        omega = optics.lamb2omega(wl)
        kappa = np.imag(n2)
        cosThetat = waferLNSi.cosThetat(n1, np.real(n2), thetai)
        a = np.exp(-2 * omega * kappa / optics.c * dist / cosThetat)
        return a
    
    @staticmethod
    def plotReflectanceAndPhase(lambLow, lambHigh, thetaRange, N=100):
        """
        Plot the reflectance and phase for a range of wavelengths and angles.
        
        Parameters:
        lambLow (float): Lower wavelength limit in meters.
        lambHigh (float): Upper wavelength limit in meters.
        thetaRange (float): Range of incident angles in radians.
        N (int): Number of points for the plot.
        """
        lamb = np.linspace(lambLow, lambHigh, N)
        thetai = np.linspace(-thetaRange / 2, thetaRange / 2, N)
        tthetai, llamb = np.meshgrid(thetai, lamb, indexing='ij')
        
        Rp = waferLNSi.reflectance(tthetai, llamb, waferLNSi.pPol)
        Rs = waferLNSi.reflectance(tthetai, llamb, waferLNSi.sPol)
        angRp = waferLNSi.angleReflectivity(tthetai, llamb, waferLNSi.pPol)
        angRs = waferLNSi.angleReflectivity(tthetai, llamb, waferLNSi.sPol)
        
        fig, ax = plot_custom(
            12, 9, [r"incident angle, $\theta_i$ ($^\circ$)", r"incident angle, $\theta_i$ ($^\circ$)"], 
            [r"wavelength, $\lambda$ (um)", r"wavelength, $\lambda$ (um)"], ncol=2, nrow=2, equal=False, 
            numsize=20, commonY=True, commonX=True, axefont=25
        )
        
        p1 = ax[0, 0].pcolormesh(np.degrees(tthetai), llamb * 1e6, Rp, cmap='hot', vmin=0, vmax=1)
        p2 = ax[0, 1].pcolormesh(np.degrees(tthetai), llamb * 1e6, Rs, cmap='hot', vmin=0, vmax=1)
        p3 = ax[1, 0].pcolormesh(np.degrees(tthetai), llamb * 1e6, np.degrees(angRp), cmap='hot')
        p4 = ax[1, 1].pcolormesh(np.degrees(tthetai), llamb * 1e6, np.degrees(angRs), cmap='hot')
        
        ax[0, 0].set_title(r"p-polarized", fontsize=25)
        ax[0, 1].set_title(r"s-polarized", fontsize=25)
        
        cbar1 = fig.colorbar(p1, ax=ax[0, 0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p2, ax=ax[0, 1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'reflectance, $R$ (au)', rotation=90, fontsize=25, labelpad=10)
        
        cbar1 = fig.colorbar(p3, ax=ax[1, 0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p4, ax=ax[1, 1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'reflection phase ($^\circ$)', rotation=90, fontsize=25, labelpad=10)
    
    @staticmethod
    def plotTransmittanceAndPhase(lambLow, lambHigh, thetaRange, dist, N=100, absorbBool=False):
        """
        Plot the transmittance and phase for a range of wavelengths and angles.
        
        Parameters:
        lambLow (float): Lower wavelength limit in meters.
        lambHigh (float): Upper wavelength limit in meters.
        thetaRange (float): Range of incident angles in radians.
        dist (float): Distance in meters.
        N (int): Number of points for the plot.
        absorbBool (bool): Whether to include absorption in the calculation.
        """
        lamb = np.linspace(lambLow, lambHigh, N)
        thetai = np.linspace(-thetaRange / 2, thetaRange / 2, N)
        tthetai, llamb = np.meshgrid(thetai, lamb, indexing='ij')
        
        Tp = waferLNSi.transmittance(tthetai, llamb, waferLNSi.pPol, dist)
        Ts = waferLNSi.transmittance(tthetai, llamb, waferLNSi.sPol, dist)
        absorption = waferLNSi.absorption(tthetai, llamb, dist)
        
        angTp = waferLNSi.angleTrans(tthetai, llamb, waferLNSi.pPol)
        angTs = waferLNSi.angleTrans(tthetai, llamb, waferLNSi.sPol)
        
        if absorbBool:
            Tp *= absorption
            Ts *= absorption
        
        fig, ax = plot_custom(
            12, 9, [r"incident angle, $\theta_i$ ($^\circ$)", r"incident angle, $\theta_i$ ($^\circ$)"], 
            [r"wavelength, $\lambda$ (um)", r"wavelength, $\lambda$ (um)"], ncol=2, nrow=2, equal=False, 
            numsize=20, commonY=True, commonX=True, axefont=25
        )
        
        p1 = ax[0, 0].pcolormesh(np.degrees(tthetai), llamb * 1e6, Tp, cmap='hot', vmin=0, vmax=1)
        p2 = ax[0, 1].pcolormesh(np.degrees(tthetai), llamb * 1e6, Ts, cmap='hot', vmin=0, vmax=1)
        p3 = ax[1, 0].pcolormesh(np.degrees(tthetai), llamb * 1e6, np.degrees(angTp), cmap='hot')
        p4 = ax[1, 1].pcolormesh(np.degrees(tthetai), llamb * 1e6, np.degrees(angTs), cmap='hot')
        
        ax[0, 0].set_title(r"p-polarized", fontsize=25)
        ax[0, 1].set_title(r"s-polarized", fontsize=25)
        
        cbar1 = fig.colorbar(p1, ax=ax[0, 0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p2, ax=ax[0, 1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'transmittance, T (au)', rotation=90, fontsize=25, labelpad=10)
        
        cbar1 = fig.colorbar(p3, ax=ax[1, 0])
        cbar1.ax.tick_params(labelsize=20)
        
        cbar2 = fig.colorbar(p4, ax=ax[1, 1])
        cbar2.ax.tick_params(labelsize=18)
        cbar2.set_label(r'transmission phase ($^\circ$)', rotation=90, fontsize=25, labelpad=10)
    
    @staticmethod
    def TIR(wl):
        """
        Calculate the total internal reflection angle for lithium niobate.
        
        Parameters:
        wl (float): Wavelength in meters.
        
        Returns:
        float: Total internal reflection angle in radians.
        """
        n0 = 1
        n1 = lithiumNiobate.EIndexDisp(wl)
        return np.arcsin(n0 / n1)
    
    @staticmethod
    def TIRBBO(wl):
        """
        Calculate the total internal reflection angle for BBO.
        
        Parameters:
        wl (float): Wavelength in meters.
        
        Returns:
        float: Total internal reflection angle in radians.
        """
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
        
        
        ax.set_title("Joint Spectral Intensity", fontsize = labelfontsize)


    def calcFreqAngSpectrum(wp, ws, thetas, L, LcohBool = False, BBOBool = False, chi = 0.0001, s = 1):
    
        wi = wp-ws
    
        lambPAir = optics.omega2lamb(wp)
        lambSAir = optics.omega2lamb(ws)
        lambIAir = optics.omega2lamb(wi)
        
        
        if BBOBool is False:
            npE = lithiumNiobate.EIndexDisp(lambPAir)
            nsE = lithiumNiobate.EIndexDisp(lambSAir)
            niE = lithiumNiobate.EIndexDisp(lambIAir)
            
            # calc the parallel component of the wavevector mismatch      
            # the process works better if the light is E-polarized
            # phase matching transverse
            thetai = (np.arcsin(-ws*nsE*np.sin(thetas)
                                /(wi*niE)
                                ) 
                      )
            
            # phase matching minimize total
            # thetai = -np.arctan(ws*nsE*np.sin(thetas)/(wp*npE-ws*nsE*np.cos(thetas)))
            
            # phase matching longitudinal
            # thetai = (np.arccos((wp*npE - ws*nsE*np.cos(thetas))
            #                     /(wi*niE)
            #                     ) 
            #           )
            # thetai = (np.arccos(nsE*ws/(niE*wi)*np.sin(thetas))) - np.pi/2
            # kp = wp*npE/optics.c
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
            
            waistp = 0.001e-6#m
        
        if BBOBool is True:
            npE = BBO.EIndexDisp(lambPAir)
            nsE = BBO.OIndexDisp(lambSAir)
            niE = BBO.OIndexDisp(lambIAir) #type I for BBO
        
            # calc the parallel component of the wavevector mismatch      
            # the process works better if the light is E-polarized
            # BBO should be longitudinally phase matched
            #transverse phase matching
            
            # thetai = (np.arccos(nsE*ws/(niE*wi)*np.sin(thetas))) - np.pi/2
            
            # kp = wp*npE/optics.c
            ks = ws*nsE*np.cos(thetas)/optics.c
            ki = wi*niE*np.cos(thetai)*npE/optics.c
            
            thetai = (np.arcsin(-ws*nsE*np.sin(thetas)
                               /(wi*niE)
                               ) 
                      )
            # phase matching longitudinal
            # thetai = (np.arccos((wp*npE - ws*nsE*np.cos(thetas))
            #                     /(wi*niE)
            #                     ) 
            #           ) * -np.sign(thetas)
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
        
        
        ax.set_title("Frequency Spectrum", fontsize = labelfontsize)
        
        
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
        
        
        ax.set_title("Frequency Angular Spectrum", fontsize = labelfontsize)
        
class SPDCResonator:
        
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
                # t = calct(n1, n2, phase, theta)
                return 0*theta
            
        
        theta = np.abs(theta)
        
        lambConj = lambda lamb: 1/(1/lambP - 1/lamb)
        
        npump = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        ns = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        
        if etalonBool is True:
            n2 = lambda lamb: silicon.sellmaierReal(lamb*1e6)
            n1 = lambda lamb: 1
        else:
            n2 = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
            n1 = lambda lamb: lithiumNiobate.EIndexDisp(lamb)
        
        
        ki = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))
        ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))
        kp = optics.lamb2k(lambP, npump(lambP))
        
        # for total phase minimization
        # thetai = lambda lamb: -np.arctan(np.abs(ks(lamb))*np.sin(theta)/(np.abs(kp)-np.abs(ks(lamb))*np.cos(theta)))
        
        # for transverse phase matching
        thetai = lambda lamb: -np.arcsin(np.abs(ks(lamb))*np.sin(theta)/(np.abs(ki(lamb))))
        
        deltas = lambda lamb: L*ks(lamb)*np.cos(theta)
        deltai = lambda lamb: L*ki(lamb)*np.cos(thetai(lamb))
        

        ksz = lambda lamb: optics.lamb2k(lamb, ns(lamb))*np.cos(theta)
        kiz = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))*np.cos(thetai(lamb))
        
        deltakz = lambda lamb: -1* (ksz(lamb) + kiz(lamb) - kp)
        deltaz = lambda lamb: L * deltakz(lamb)
        deltap = L*kp
        
        
        
        t1p = calct(n1(lambP), ns(lambP), deltap, 0)
        # t2p = calct(ns(lambP), n2(lambP), deltap, 0)
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
        
        E0p = t1p/(1 - r1p*r2p)*s;
        E0m = E0p*r2p*1;
        
        
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
        
        tau1 = lambda lamb: np.array([[t1s(lamb), 0, 0, 0], [0, np.conjugate(t1i(lamb)), 0, 0], [0, 0, t2s(lamb), 0], [0, 0, 0, np.conjugate(t2i(lamb))]])
        tau2 = lambda lamb: np.array([[t2s(lamb), 0, 0, 0], [0, np.conjugate(t2i(lamb)), 0, 0], [0, 0, t1s(lamb), 0], [0, 0, 0, np.conjugate(t1i(lamb))]])

        rho1 = lambda lamb: np.array([[0, 0, r1s(lamb), 0], [0, 0, 0, np.conjugate(r1i(lamb))], [r2s(lamb), 0, 0, 0], [0, np.conjugate(r2i(lamb)), 0, 0]])
        rho2 = lambda lamb: np.transpose(np.conjugate(rho1(lamb)))
        
        I = np.eye(4)
        
        U = lambda lamb: tau2(lamb).dot(np.linalg.inv(I - np.matmul(W(lamb), rho1(lamb)))).dot(W(lamb)).dot(tau1(lamb)) - (rho2(lamb))
        if gainBool == True:
            gainM = lambda lamb: gammaM(lamb)
            gainP = lambda lamb: gammaP(lamb)
            return gainM, gainP
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

        
    def calcSpectrumSPDCff(lamb, lambP, L = 10.1e-6, s = 1, chi = 1E-10, theta = 0, etalonBool = True):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L, s = s, chi=chi, etalonBool = etalonBool, gainBool = False)(lamb)

        
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
            
    
    def calcSpectrumSPDCbb(lamb, lambP, L = 10.1e-6, s =1, chi = 1E-10, theta = 0):
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
        
    def calcSpectrumSPDCfb(lamb, lambP, L = 10.1e-6, s =1, chi = 1E-10, theta = 0):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L,s = s, chi=chi)(lamb)
        
        def Stsri(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (UAb[3,0]**2 * (UAb[0,0]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                    UAb[3,2]**2 * (UAb[0,2]**2 + UAb[0,1]**2 + UAb[0,3]**2) + 
                    2 * np.real(UArr[0,0]*UArr[3,2]*np.conjugate(UArr[3,0])*np.conjugate(UArr[0,2]))
                    )
        

        SSfb = np.zeros_like(lamb)
        
        if lamb.ndim == 1:  # 1D array
            for i in range(lamb.shape[0]):
                SSfb[i] = Stsri(lamb[i])
        
        elif lamb.ndim == 2:  # 2D 
            for i in range(lamb.shape[0]):
                for j in range(lamb.shape[1]):
                    SSfb[i,j] = Stsri(lamb[i,j])
                    
        return SSfb
    
    def calcSpectrumSPDCbf(lamb, lambP, L = 10.1e-6, s =1, chi = 1E-10, theta = 0):
        U = lambda lamb: SPDCResonator.calcInteractionMatrix(lambP, theta = theta, L = L,s = s, chi=chi)(lamb)
        
        def Stirs(lamb): 
            UArr = U(lamb)
            UAb = np.absolute(UArr)
            return (UAb[1,0]**2 * (UAb[2,0]**2 + UAb[2,1]**2 + UAb[2,3]**2) + 
                    UAb[1,2]**2 * (UAb[2,2]**2 + UAb[2,1]**2 + UAb[2,3]**2) +
                    2 * np.real(UArr[2,0]*UArr[1,2]*np.conjugate(UArr[1,0])*np.conjugate(UArr[2,2]))
                    )
        
        SSbf = np.zeros_like(lamb)
        
        if lamb.ndim == 1:  # 1D array
            for i in range(lamb.shape[0]):
                SSbf[i] = Stirs(lamb[i])
        
        elif lamb.ndim == 2:  # 2D 
            for i in range(lamb.shape[0]):
                for j in range(lamb.shape[1]):
                    SSbf[i,j] = Stirs(lamb[i,j])
                    
        return SSbf 
        

        
        
    
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
        
        
        return PEForwards, PEBacks
    
    
    def calcEtalonEffect(lamb, lambP = 788e-9, L = 10e-6, theta = 0, interferenceBool = False, detEffTrans = 1, detEffRefl = 1, collectionBool = False, interfaceType = 2, envelopeFunc = 1):
        # etalonType: 2: air -> LN -> silicon ; 1: custom; 3: air -> GaP -> silica; 4: GaP -> silica -> sapphire
        if collectionBool == True:
            
            totalEfficiency = envelopeFunc
            totalEfficiencyHalf = envelopeFunc
            
        else:
            totalEfficiency = 1
            totalEfficiencyHalf = 1
        
        
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
        elif interfaceType == 4:
            ns = lambda lamb: 1.7*(lamb*1e6) #s for sig/idler in substrate
            n2 = lambda lamb: 2
            n1 = lambda lamb: 1
        
        
        
        ki = lambda lamb: optics.lamb2k(lambConj(lamb), ns(lambConj(lamb)))
        ks = lambda lamb: optics.lamb2k(lamb, ns(lamb))
        kp = optics.lamb2k(lambP, npump(lambP))
        
        thetai = lambda lamb: -np.arctan(np.abs(ks(lamb))*np.sin(theta)/(np.abs(kp)-np.abs(ks(lamb))*np.cos(theta)))
        
        deltas = lambda lamb: L*ks(lamb)*np.cos(theta)
        deltai = lambda lamb: L*ki(lamb)*np.cos(thetai(lamb))
        
        t12 = lambda lamb, theta, delta: calct(ns(lamb), n2(lamb), delta, theta)
        r12 = lambda lamb, theta, delta: calcr(ns(lamb), n2(lamb), delta+np.pi, theta)
        t10 = lambda lamb, theta, delta: calct(ns(lamb), n1(lamb), delta, theta)
        r10 = lambda lamb, theta, delta: calcr(ns(lamb), n1(lamb), delta, theta)
        
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
        
        RTForwards = (PEForwards
                      *r12(lamb, theta, deltas(lamb))*(t10(lamb, theta, deltas(lamb)))*FP(lamb, theta, deltas(lamb))
                      *t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        RTBacks = (PEBacks
                   *t10(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                   *r10(lambConj(lamb), thetai(lamb), deltai(lamb))*t12(lambConj(lamb), thetai(lamb), deltai(lamb))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        
        TRForwards = (PEForwards
                      *t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                      *r12(lambConj(lamb), thetai(lamb), deltai(lamb))*(t10(lambConj(lamb), thetai(lamb), deltai(lamb)))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        TRBacks = (PEBacks
                   *(r10(lamb, theta, deltas(lamb)))*t12(lamb, theta, deltas(lamb))*FP(lamb, theta, deltas(lamb))
                   *(t10(lambConj(lamb), thetai(lamb), deltai(lamb)))*FP(lambConj(lamb), thetai(lamb), deltai(lamb)))
        

        TTs = np.abs(TTForwards+TTBacks)**2 #* detEffTrans**2
        RRs = np.abs(RRForwards+RRBacks)**2 #* detEffRefl**2
        RTs = np.abs(RTForwards+RTBacks)**2 #* detEffTrans*detEffRefl
        TRs = np.abs(TRForwards+TRBacks)**2 #* detEffTrans*detEffRefl
            
        # TRRTs = TRs + RTs
        # 
        return TTs*totalEfficiency, RRs*totalEfficiency, TRs*totalEfficiencyHalf, RTs*totalEfficiencyHalf
        
    def plotSpectraEtalon(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6, theta = 0, normBool = False, 
                                 interferenceBool = False, emissionProbBool = True, s = 1, chi = 1E-10, 
                                 detEffTrans = 1, detEffRefl = 1, emissionConjBool = False, interfaceType = 2):
        colorArray = thesisColorPalette(3, paletteType = "smooth")
        
        llamb = np.linspace(lambMin, lambMax, numPoints)
        llambConj = 1/(1/lambP - 1/ llamb)

        TT, RR, TR, RT = SPDCResonator.calcEtalonEffect(llamb, lambP, L, theta = theta, interferenceBool = interferenceBool, interfaceType = interfaceType, detEffTrans = detEffTrans, detEffRefl = detEffRefl)
        # TTc, RRc, TRRTc = SPDCResonator.calcEtalonEffect(llambConj, lambP, L, interferenceBool = interferenceBool, detEffTrans = detEffTrans, detEffRefl = detEffRefl)
        
        if emissionProbBool is True:
            wp = optics.lamb2omega(lambP)
            theta = np.linspace(0, 0,1)+theta
            # theta = np.linspace(-np.pi/10000, np.pi/10000,1)
            ttheta,llambs = np.meshgrid(theta, llamb, indexing='ij');
            _,llambsConj = np.meshgrid(theta, llambConj, indexing='ij');
            wws = optics.lamb2omega(llambs)
            # wwsc = optics.lamb2omega(llambsConj)
            
            FAS = np.sum(np.nan_to_num(np.abs(SPDCalc.calcFreqAngSpectrum(wp, wws, ttheta, L, s = s, chi = chi))), axis = 0)
            # FASConj = np.sum(np.nan_to_num(np.abs(SPDCalc.calcFreqAngSpectrum(wp, wwsc, ttheta, L))), axis = 0)
            
            TT *= FAS
            RR *= FAS
            TR *= FAS
            
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
            ax.plot(llamb*1e6, TR/max(TR), linestyle = ':', color = colorArray[2])
        else:
            ax.plot(llamb*1e6, TT/max(TT), linestyle = '-', color = colorArray[0])
            ax.plot(llamb*1e6, RR/max(TT), linestyle = '--', color = colorArray[1])
            ax.plot(llamb*1e6, TR/max(TT), linestyle = ':', color = colorArray[2])
        # print(max(TT))
        ax.set_ylim([0,None])           
        # ax.legend(['trans', 'refl', 'both'])
        ax.set_title("$L = $" + f"{L*1e6:.2f}" + " um", fontsize = labelfontsize)
        ax.set_title("Simple Model", fontsize = labelfontsize)
        
    
    def plotSpectra(lambMin, lambMax, numPoints, lambP = 0.788e-6, L = 10.1e-6, theta = 0, normBool = True, chi = 1e-10, s = 1):
        colorArray = thesisColorPalette(3, paletteType = "smooth")

        llamb = np.linspace(lambMin, lambMax, numPoints)
        
        
        # for i in range(len(llamb)):
            # St[i] = SPDCResonator.calcSpectrumSPDCff(llamb[i], lambP, L = L, s= s, chi = chi)
        Sr = SPDCResonator.calcSpectrumSPDCbb(llamb, lambP, L = L, s= s, chi = chi, theta = theta)
        Str = SPDCResonator.calcSpectrumSPDCfb(llamb, lambP, L = L, s=s, chi = chi, theta = theta)
        St = SPDCResonator.calcSpectrumSPDCff(llamb, lambP, L = L, s= s, chi = chi, theta = theta)
            
            
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
        ax.set_title("Rigorous Model", fontsize = labelfontsize)
        # ax.legend(['trans', 'refl', 'both'])
        
    def plotFAS(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10, chi = 0.00001, s = 1, saveBool=False, etalonBool = False):
            
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
               textwidth*0.5, 0.4*textwidth, r"$\omega_s/2\pi$ (THz)",r" $\theta$ ($^\circ$)" #"coincidence rate (mHz/mW)"
               , labelPadX = 4, labelPadY = 4, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
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
                TT, RR, TR, RT = SPDCResonator.calcEtalonEffect(llambs, lambp, L, theta = np.abs(ttheta), interferenceBool=True)
                TT = np.nan_to_num(TT)
                FAS = FAS*TT
            

            # wBack = np.min( wws*1e-12/(2*np.pi))
            # backgroundw = np.array([[2*wBack,  2*wBack],[wBack, wBack]])
            # backgroundthe = np.array([[-10, 10],[-10, 10]])
            # background = np.array([[0.788,  2*0.788],[0.788,  2*0.788]])*0
            
            # ax.pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
            
            p1 = ax.pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), FAS/np.max(FAS), cmap='plasma')
            axg0.pcolormesh(llambs*1e6, np.degrees(ttheta),FAS/np.max(FAS), cmap='plasma', alpha = 0)
            
            axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))

            # axg0.xaxis.set_label_position("bottom")
            # axg0.xaxis.set_ticks_position('bottom')
            
            axg0.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 7)
            
            cbar = fig.colorbar(p1, ax = axg0, pad=0.05, aspect = 4)
            cbar.ax.tick_params(labelsize=numfontsize)
            cbar.set_label(r'', rotation = 90, fontsize = labelfontsize, labelpad = 5)
            cbar.set_ticks([0,1])
            cbar.set_label(r'emission prob. (au)', labelpad = 3)
            
            #print lines of TIR in crystal
            # axg0.set_xticks([0.788,  2*0.788])
            # axg0.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
            # ax.set_xticks([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
            # ax.set_xticklabels([r'$\omega_p/4\pi$',  r'$\omega_p/2\pi$'])
            # ax.set_xlim([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
            # ax.set_ylim([-10,10])
            plt.tight_layout()
            
            if saveBool is True:        
                dateTimeObj = datetime.now()
                preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        
                plt.savefig(
                    "figures/" + preamble + "_SPDCSpectra" + ".png", dpi=200, bbox_inches="tight",transparent=True
                )
                
    def plotNLFAS(lambsLow, lambsHigh, lambp, thetaRange, L, N=300, M=10,s = 1, chi = 1e-10,saveBool=False, etalonBool = False):
            
            lambs = np.linspace(lambsLow, lambsHigh, N)
            theta = np.linspace(-thetaRange/2, thetaRange/2,M)
            
            llambs,ttheta = np.meshgrid(lambs, theta, indexing='ij');
            
            wp = optics.lamb2omega(lambp)
            wws = optics.lamb2omega(llambs)
            
            St = llambs.copy()*0
            # Sr = llambs.copy()*0
            # Str = llambs.copy()*0
            
            for i in range(M):
                # Sr[:,i] = SPDCResonator.calcSpectrumSPDCbb(lambs, lambp, theta = theta[i], L = L, s= s, chi = chi)
                # Str[:,i] = SPDCResonator.calcSpectrumSPDCfb(lambs, lambp, theta = theta[i], L = L, s=s, chi = chi)
                St[:,i] = np.nan_to_num(SPDCResonator.calcSpectrumSPDCff(lambs, lambp, theta = theta[i], L = L, s= s, chi = chi))

            # fig, ax = plot_custom(
            #    textwidth/1.5, textwidth/2.5, r" $\theta$ ($^\circ$)", 
            #    r"signal frequency, $\nu_s$ (THz)",  axefont = labelfontsize, 
            #    numsize = numfontsize, labelPad=labelfontsize, axel=3, wSpaceSetting = 0, hSpaceSetting = 0, 
            #    labelPadX = 8, labelPadY = 10, nrow = 1, ncol = 1)
            
            fig, ax = plot_custom(
               textwidth*0.5, 0.4*textwidth, r"$\omega_s/2\pi$ (THz)",r" $\theta$ ($^\circ$)" #"coincidence rate (mHz/mW)"
               ,labelPadX = 4, labelPadY = 4, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
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
            


            # wBack = np.min( wws*1e-12/(2*np.pi))
            # backgroundw = np.array([[2*wBack,  2*wBack],[wBack, wBack]])
            # backgroundthe = np.array([[-10, 10],[-10, 10]])
            # background = np.array([[0.788,  2*0.788],[0.788,  2*0.788]])*0
            
            # ax.pcolormesh(backgroundw, backgroundthe, background, cmap='plasma')
            
            p1 = ax.pcolormesh(wws*1e-12/(2*np.pi), np.degrees(ttheta), St/np.max(St), cmap='plasma')
            axg0.pcolormesh(llambs*1e6, np.degrees(ttheta),St/np.max(St), cmap='plasma', alpha = 0)
            
            axg0.set_xscale('function', functions=(optics.lamb2omega, optics.omega2lamb))

            # axg0.xaxis.set_label_position("bottom")
            # axg0.xaxis.set_ticks_position('bottom')
            
            axg0.set_xlabel(r"$\lambda_s$ ($\unit{\micro m}$)", fontsize = labelfontsize, labelpad = 7)
            
            
            cbar = fig.colorbar(p1, ax = axg0, pad=0.05, aspect = 4)
            cbar.ax.tick_params(labelsize=numfontsize)
            cbar.set_label(r'', rotation = 90, fontsize = labelfontsize, labelpad = 5)
            
            cbar.set_label(r'emission prob. (au)', labelpad = 3)
            cbar.set_ticks([0,1])
            #print lines of TIR in crystal
            # axg0.set_xticks([0.788,  2*0.788])
            # axg0.set_xticklabels([r'$\lambda_p$', r'$2\lambda_p$'])
            # ax.set_xticks([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
            # ax.set_xticklabels([r'$\omega_p/4\pi$',  r'$\omega_p/2\pi$'])
            # ax.set_xlim([wp/1e12/(2*np.pi)/2,wp/1e12/(2*np.pi)])
            # ax.set_ylim([-10,10])
            plt.tight_layout()
        
            if saveBool is True:        
                dateTimeObj = datetime.now()
                preamble = dateTimeObj.strftime("%Y%m%d_%H%M")
        
                plt.savefig(
                    "figures/" + preamble + "_SPDCSpectra" + ".png", dpi=200, bbox_inches="tight",transparent=True
                )

