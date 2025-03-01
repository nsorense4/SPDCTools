from SPDCalc import *

if __name__ == '__main__':   

    #%% set constants for assessment

    L = 10.15e-6 # thickness of crystal in m
    lambP = 0.787e-6 # pump wavelength
    
    lambMin = 2*lambP-0.5e-6#1.2e-6    #m
    lambMax = 1/(1/lambP-1/lambMin)
    
    L = 10.15e-6       #m
    chi = 34e-12        #m/V

    s = 0.0000125*1e-20/chi   #V/m    0.0125/chi has considerable gain, 0.001/chi does not.  
        
    #%% plot JSI
    SPDCalc.plotJSI(2*lambP, 200e-9, lambP, 2e-9)
    
    #%% plot the frequency angular spectra of a non-resonant LN crystal
    SPDCalc.plotFreqAngSpectrum(850e-9, 2500e-9, lambP, np.pi/2, L =L)
    
    #%% plot frequency spectra at a given angle of a non-resonant LN crystal
    SPDCalc.plotFreqAngSpectrumLine(850e-9, 2500e-9, lambP, 0, L =L)
        
    #%% SPDC etalon/resonator calculations
    theta = np.pi/180*0
    lambMin = 1e-6#1.2e-6    #m
    lambMax = 1/(1/lambP-1/lambMin)
    
    # plots spectra for collinear SPDC calculated using the simple model
    SPDCResonator.plotSpectraEtalon(lambMin, lambMax, 500, L = L, theta = theta, normBool = False, interfaceType=2,
                                            interferenceBool=True, emissionProbBool=True, s = s, chi = chi)
    
    # plots spectra for collinear SPDC calculated using the quantum model
    SPDCResonator.plotSpectra(lambMin, lambMax, 500, L = L, theta = theta, normBool = False, s = s, chi = chi)
    plt.show()    
    #%% Plot FAS for both simple and rigorous models across different values
    thetaRange = 20*np.pi/180

    SPDCResonator.plotFAS(lambMin, lambMax, lambP, thetaRange, L, N=1000, M = 101, chi = chi, s = s, etalonBool = True)
    SPDCResonator.plotNLFAS(lambMin, lambMax, lambP, thetaRange, L, N=301, M = 101, chi = chi, s = s)
    # sArr = [s, s, s*1e8]
    # plotFASGridForPaper(lambMin, lambMax, lambp, thetaRange, L, N=3000, M = 101, chi = chi, s = s, saveBool=saveBool)
    
    plt.show()
    


