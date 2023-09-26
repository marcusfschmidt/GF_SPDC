#%%

import sys
import time
import numpy as np
from math import factorial
import scipy as sp
from scipy.integrate import complex_ode, ode
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import special
from scipy import signal
import matplotlib.pyplot as plt
import timeit


class CoupledModes(object):

    def __init__(self, n, dt, dz, offset, L, beta, gamma, lambda_p, omega_s = 0, omega_i = 0, alpha_s = 0, alpha_i = 0, printBool = False, rtol = 1e-3, nsteps = 10000) -> None:

        """
        n: number of points in the grid
        dt: time step
        dz: propagation step
        offset: time offset to be added to the time axis
        L: Propagation length
        beta: Dispersion coefficients or the full beta function
        gamma: Nonlinear coefficient
        lambda_p: Pump wavelength expansion point. Depending on the input, the pump can be centered somewhere else, but the expansion point is always lambda_p
        omega_s: The s-pump frequency expansion point with respect to the expansion point of the pump
        omega_i: The i-pump frequency expansion point with respect to the expansion point of the pump
        alpha_s and alpha_i: The attenuation coefficients of the signal and idler
        QPM_bool: If true, the program will impose quasi phase matching with the period QPM
        printBool: If true, the program will print the progress of the simulation
        rtol: Relative tolerance for the integrator
        nsteps: Number of steps for the integrator

        """

        self.N = 2**int(n)
        self.dt = dt
        self.dz = dz
        self.offset = offset
        self.L = L
        self.beta = beta
        self.gamma = gamma
        self.lambda_p = lambda_p
        self.alpha_s = alpha_s
        self.alpha_i = alpha_i
        self.initialConditionsFlag = False
        self.printBool = printBool 

        self.QPMPeriod = 0
        self.QPM_bool = beta.QPMbool
        if self.QPM_bool:
            self.QPMPeriod = beta.QPMPeriod
        
        self.solver = ode(self.odeNL)
        self.solver.set_integrator('dopri5', rtol = rtol, nsteps = nsteps)

        #speed of light in m/ps
        self.c = 299792458*1e-12
        #hbar in J*ps
        self.hbar = 1.054571817*1e-34*1e12

        #Make axes
        self.t = np.arange(-self.N/2, self.N/2)*self.dt + offset
        self.domega = 2*np.pi/(self.N*self.dt)
        self.omega = np.arange(-self.N/2, self.N/2)*self.domega

        #pump frequencies
        self.fp = self.c/self.lambda_p
        self.omega_p = 2*np.pi*self.fp

        #Define the numerical and physical frequencies
        self.omegaReal = self.omega + self.omega_p
        self.omega_s = self.omega_p + omega_s
        self.omega_i = self.omega_p + omega_i
        self.omegaRealS = self.omega + self.omega_s
        self.omegaRealI = self.omega + self.omega_i

        #Define physical wavelengths
        self.lambdaReal = self.c/((self.omega + self.omega_p)/(2*np.pi))*1e9 #wavelength in nm
        self.lambdaRealS = self.c/((self.omega + self.omega_s)/(2*np.pi))*1e9 #wavelength in nm
        self.lambdaRealI = self.c/((self.omega + self.omega_i)/(2*np.pi))*1e9 #wavelength in nm
        
        # #Various checks for the validity of the input
        if self.omegaReal[0] < 0:
            print("øv")
        #     sys.exit("Center wavelength too high for the omega-array to be non-negative. Try increasing the frequency or increasing dt.")

        self.lamReal = self.c/((self.omega + self.omega_p)/(2*np.pi))*1e9 #wavelength in nm
        # if self.lamReal[0] < self.lambda_p*1e9:
        #     sys.exit("Pump wavelength is larger than the largest value in the wavelength warray.")

        # if self.lamReal[-1] > self.lambda_p*1e9:
        #     sys.exit("Pump wavelenght is smaller than the smallest value in the wavelength array.")


        #If using beta function, transform it appropriately
        try:
            #load wave vectors from the beta class
            kp = beta.kp
            ks = beta.ks
            ki = beta.ki
            om = beta.om

            #Interpolate and extract beta1 of the pump
            kp_fit = InterpolatedUnivariateSpline(om, kp)
            self.kp_fit = kp_fit
            kpFull = kp_fit(self.omegaReal)
            beta1 = kp_fit.derivative()(self.omega_p)
            self.beta1 = beta1
            self.kpFull = kpFull

            #Interpolate ks and ki
            ks_fit = InterpolatedUnivariateSpline(om, ks)
            ksFull = ks_fit(self.omegaRealS)
            self.ks = ks_fit.derivative()(self.omega_s)
            self.ksFull = ksFull

            ki_fit = InterpolatedUnivariateSpline(om, ki)
            kiFull = ki_fit(self.omegaRealI)
            self.ki = ki_fit.derivative()(self.omega_i)
            self.kiFull = kiFull

            #Define the moving frame
            self.k_reference = beta1

        #If using beta coefficients, make the beta function from taylor series and transform
        except AttributeError:
            self.k_reference = beta[0][1]

            kpFull = self.makeSimpleBeta(beta[0], expansionPoint = self.omega_p)
            ksFull = self.makeSimpleBeta(beta[1], expansionPoint = self.omega_s)
            self.ks = beta[1][1]
            kiFull = self.makeSimpleBeta(beta[2], expansionPoint = self.omega_i)
            self.ki = beta[2][1]

        

        #Transform the beta coefficients
        self.betap = self.transformBeta(kpFull, self.k_reference, self.omegaReal, self.omega_p)
        self.betas = self.transformBeta(ksFull, self.k_reference, self.omegaRealS, self.omega_s)
        self.betai = self.transformBeta(kiFull, self.k_reference, self.omegaRealI, self.omega_i)

        timeShiftS = (self.ks - self.k_reference)*self.L
        timeShiftI = (self.ki - self.k_reference)*self.L
        self.timeShiftArray = np.array([timeShiftS, timeShiftI])


    
    ###############################################################################################################
    ### Utility functions begin ###
    #Functions to declutter the fft/ifft shifts
    #Note that in the outcommented code, the factors ensure satisfying the Parseval theorem
    def fft(self, field):
        return sp.fft.fftshift(sp.fft.fft(sp.fft.ifftshift(field)))
        # return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))*self.dt/np.sqrt(2*np.pi)
    def ifft(self, field):
        return sp.fft.ifftshift(sp.fft.ifft(sp.fft.fftshift(field)))
        # return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field)))*np.sqrt(2*np.pi)/self.dt


    #Function to calculate the taylor series of a functiion given its coefficients
    def taylorSum(self, coefficients, variable):
        sum = 0
        for i in range(len(coefficients)):
            sum += coefficients[i]/factorial(i)*variable**(i)
        return sum

    #Function to find the nearest index in an array to a given value
    def findNearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    ### Utility functions end ###
    ###############################################################################################################

    #Functions for determining the beta functions

    #Simple dispersion curve using only coefficients in the transformed picture (i.e. moving frame of reference with pump)
    def makeSimpleBeta(self, coefficients, expansionPoint):
        freq = self.omega + self.omega_p
        beta = self.taylorSum(coefficients, freq - expansionPoint)
        return beta

    #Function to transform the beta function to the moving reference
    def transformBeta(self, beta, beta1, freq, expansionPoint):
        beta = beta - beta1*(freq - expansionPoint)
        return beta

    ###############################################################################################################
    #Input types begin

    def normalizeInput(self, field):
        norm = np.sum(field)*self.domega
        return field/norm

    #Make a standard Gaussian pump input in frequency domain
    def makeGaussianInput(self, T0, Toff = 0):
        field = np.zeros_like(self.t, dtype = complex)
        field += 1/(T0*np.sqrt(2*np.pi))*np.exp(-4*np.log(2)*((self.t + Toff)/T0)**(2))
        field = self.fft(field)
        return self.normalizeInput(field)

    #sech input
    def makeSechInput(self, P0, Toff, T0):
        field = np.zeros_like(self.t, dtype = complex)
        
        def invacosh(x):
            res = [1/np.cosh(i) if np.abs(i) < 710.4 else 0 for i in x]
            return np.array(res)

        argument = (self.t + Toff)/T0
        field += np.sqrt(P0)*invacosh(argument)*np.exp(-1j*(self.t+Toff)**2/(2*T0**2))
        field = self.fft(field)

        return self.normalizeInput(field)

    #CW input
    def makeCWInput(self, P0=1):
        omega = self.omega
        field = np.zeros_like(self.t, dtype = complex)
        df = (omega[1] - omega[0])/(2*np.pi)
        centerIdx = int(self.N/2)
        field[centerIdx] = np.sqrt(P0/(2*np.pi))/df
        return self.normalizeInput(field)

    def makeHermiteGaussianBasisFunctions(self, Toff, T0, n, fftBool = True):
        field = np.zeros_like(self.t, dtype = complex)
        field += self.hermiteGaussianFunction(n, self.t + Toff, T0)
        norm = np.sum(np.abs(field)**2)*self.dt
        field = field/np.sqrt(norm)
        if fftBool:
            field = self.fft(field)
        return field

    def hermiteGaussianFunction(self, n, t, width):
        field = 1/np.exp(n)*np.exp(-t**2/(2*width**2))*special.eval_hermite(n, t/width) 
        return field

        
    #Add random noise corresponding to one photon per mode (probably not rigurous in crystals)
    def addNoise(self, field):

        df = (self.omega[1] - self.omega[0])/(2*np.pi)
        rnd = np.random.random(field.size)
        field += np.exp(1j*rnd*2*np.pi)*np.sqrt(self.hbar*(self.omegaReal)/df)

        return field

    #Input types end
    ###############################################################################################################

    #Function to get the pump field at a given z
    def getPump(self, z):
        return self.Ap_0*np.exp(1j*self.betap*z)

    #The QPM function for the corresponding structure
    def QPM(self, z):
        if self.QPMPeriod == 0:
            return 1
        return np.sign(np.sin(2*np.pi/self.QPMPeriod*z))

    #Function to set the initial conditions for the solver
    def setInitialConditions(self, inputFields):
        self.initialConditionsFlag = True
        self.As_0, self.Ai_0, self.Ap_0  = inputFields
        initialValues = np.hstack((np.real(self.As_0), np.imag(self.As_0), np.real(self.Ai_0), np.imag(self.Ai_0))) 
        self.solver.set_initial_value(initialValues, 0)

    #Function to save the fields at a given z
    def saveVariables(self, i, field):
        z = i*self.dz

        As = (field[0] + 1j*field[1])*np.exp(1j*self.betas*z)
        Ai = (field[2] + 1j*field[3])*np.exp(1j*self.betai*z)

        fieldTime = np.zeros((self.N, 2), dtype = complex)
        fieldSpec = np.zeros_like(fieldTime)

        fieldTime[:, 0] = self.ifft(As)
        fieldTime[:, 1] = self.ifft(Ai)

        fieldSpec[:, 0] = As
        fieldSpec[:, 1] = Ai

        return z, fieldSpec, fieldTime
    

    #Function to solve the coupled differential equations
    def odeNL(self, z: float, fieldInteraction: np.ndarray) -> np.ndarray:

        #Obtain the interaction form of the fields
        fieldInteraction = np.reshape(fieldInteraction, (4, self.N))
        As_IReal, As_IImag, Ai_IReal, Ai_IImag = fieldInteraction
        As_Interaction = As_IReal + 1j*As_IImag
        Ai_Interaction = Ai_IReal + 1j*Ai_IImag

        signalExponentialFactor = 1j*(self.betas + 1j*self.alpha_s)*z
        idlerExponentialFactor = 1j*(self.betai + 1j*self.alpha_i)*z

        As = self.ifft(As_Interaction*np.exp(signalExponentialFactor))
        Ai = self.ifft(Ai_Interaction*np.exp(idlerExponentialFactor))
        Ap = self.ifft(self.getPump(z))
        QPM = self.QPM(z)

        # #Non-linear operator
        NAs = 1j*self.gamma*np.conjugate(Ai)*Ap*QPM
        NAi = 1j*self.gamma*np.conjugate(As)*Ap*QPM


        dAs = np.exp(-signalExponentialFactor)*self.fft(NAs)
        dAi = np.exp(-idlerExponentialFactor)*self.fft(NAi)

        return np.hstack((np.real(dAs), np.imag(dAs), np.real(dAi), np.imag(dAi)))





    #Function to solve the coupled differential equations
    def run(self):

        if not self.initialConditionsFlag:
            print("No initial conditions given. Exiting.")
            sys.exit()

        #Define number of steps and step size from input parameters
        Nz = self.L/self.dz
        Nsave = int(np.round(Nz))
        self.dz = self.L/Nsave

        #Preallocate output arrays
        zOut = np.zeros(Nsave + 1)
        fieldSpec = np.zeros((Nsave + 1, self.N, 2), dtype = complex)
        fieldTime = np.zeros_like(fieldSpec)
        
        #Save snapshot at t = 0
        fields_0 = np.array([np.real(self.As_0), np.imag(self.As_0), np.real(self.Ai_0), np.imag(self.Ai_0)])
        zOut[0], fieldSpec[0,:,:], fieldTime[0,:,:] = self.saveVariables(0, fields_0)
        Ti = time.time()

        timeLeft = 0
        #Solve the differential equations
        for i in range(1, Nsave + 1):
            t1 = time.time()

            #Approximate time left
            if self.printBool:
                print("", end = "\r")
                print("Step {} of {}, approximate time left [s]: {}".format(i, Nsave + 1, np.round(timeLeft,2)), end = "")

            self.solver.integrate(zOut[i-1] + self.dz)
            fields = self.solver.y 

            fields = np.reshape(fields, (4, self.N))

            zOut[i], fieldSpec[i, :, :], fieldTime[i, :, :] = self.saveVariables(i, fields)
            t2 = time.time() - t1
            timeLeft = (Nsave - i)*t2

        Tf = time.time() - Ti
        if self.printBool:
            print("\nSimulation time [s]: {}".format(np.round(Tf, 2)))

        return zOut, self.omega, self.omega_p, self.t, fieldSpec, fieldTime


# #Define the parameters
# n = 13
# dt = 0.1
# dz = 0.1e-4

# #speed of light in m/ps
# c = 299792458e-12


# #define wavelengths
# lambda_p = 532e-9
# lambda_s = 1064e-9

# #define frequencies
# om_p = 2*np.pi*c/lambda_p
# om_s = 2*np.pi*c/lambda_s
# om_i = om_p - om_s

# #define idler wavelength from energy conservation
# lambda_i = 2*np.pi*c/om_i

# #define numerical frequencies for the solver
# omega_s = -(om_p - om_s)
# omega_i = -(om_p - om_i)


# import type0_beta_5perMgO_LN as betaFunctionType0
# import typeII_beta_5perMgO_LN as betaFunctionTypeII
# #define parameters for the crystal
# L = 4000e-6

# #qpm parameters
# T = 36
# # ordinaryAxisBool = True
# # QPMPeriod = 5.916450343734758e-6
# QPMPeriod = 6.96e-6
# # beta = betaFunctionType0.type0(lambda_s, lambda_i, lambda_p, ordinaryAxisBool=ordinaryAxisBool, temperature=T)
# # QPM_bool = True

# #type 2 parameters
# beta = betaFunctionTypeII.typeII(lambda_s, lambda_i, lambda_p)
# QPM_bool = False

# #coupling strength
# gamma = 1e-1
# alpha_s = 0
# alpha_i = 0
# printBool = True
# rtol = 1e-4
# nsteps = 10000

# cnlse = CoupledModes(n, dt, dz, L, beta, gamma, lambda_p, omega_s, omega_i, alpha_s, alpha_i, printBool, rtol, nsteps)
# #Initial conditions
# T0 = 500
# Ap_0 = cnlse.makeGaussianInput(T0)
# As_0 = cnlse.makeGaussianInput(T0)
# # As_0 = cnlse.makeCWInput(1)
# Ai_0 = np.zeros_like(Ap_0)

# cnlse.setInitialConditions(np.array([As_0, Ai_0, Ap_0]))

# def makeGaussianInputTest(T0, Toff = 0):
#     field = np.zeros_like(cnlse.t, dtype = complex)
#     field += 1/(T0*np.sqrt(2*np.pi))*np.exp(-4*np.log(2)*((cnlse.t + Toff)/T0)**(2))
#     return field

# def makeGaussianInputTest2(T0, Toff = 0):
#     field = np.zeros_like(cnlse.t, dtype = complex)
#     field += 1/(T0*np.sqrt(2*np.pi))*np.exp(-4*np.log(2)*((cnlse.t + Toff)/T0)**(2))*np.kaiser(2**n,0)
#     return field

# def makeGaussianInputTest3(T0, Toff, timeAxis):
#     field = np.zeros_like(timeAxis, dtype = complex)
#     field += 1/(T0*np.sqrt(2*np.pi))*np.exp(-4*np.log(2)*((timeAxis + Toff)/T0)**(2))
#     return field

# newN = 2**18
# newT = np.arange(-newN/2, newN/2)*cnlse.dt
# newdOmega = 2*np.pi/(newN*cnlse.dt)
# newOmega = np.arange(-newN/2, newN/2)*newdOmega

# newOmegaReal = newOmega + cnlse.omega_p
# newbetap = cnlse.kp_fit(newOmegaReal)
# newbetap = cnlse.transformBeta(newbetap, cnlse.k_reference, newOmegaReal, cnlse.omega_p)


# newGauss = makeGaussianInputTest3(T0,0, timeAxis = newT)

# # plt.figure()
# # plt.plot(cnlse.t, np.abs(makeGaussianInputTest(1000))**2)
# # plt.plot(cnlse.t, np.abs(makeGaussianInputTest2(1000))**2)
# # plt.plot(newT, np.abs(newGauss)**2)
# plt.xlim(-10,10)
# plt.figure()

# nowindow = cnlse.fft(makeGaussianInputTest(T0))
# window = cnlse.fft(makeGaussianInputTest2(T0))
# accurate = cnlse.fft(newGauss)
# z = 0

# nowindow = nowindow * np.exp(1j*cnlse.betap*z)
# window = window * np.exp(1j*cnlse.betap*z)
# accurate = accurate * np.exp(1j*newbetap*z)

# nowindow = cnlse.ifft(nowindow)
# window = cnlse.ifft(window)
# accurate = cnlse.ifft(accurate)

# plt.plot(cnlse.t, np.abs(nowindow)**2, label = "no window")
# plt.plot(cnlse.t, np.abs(window)**2, label = "window")
# plt.plot(newT, np.abs(accurate)**2, label = "accurate")
# plt.xlim(-10,10)

# plt.legend()





# #%%
# z, omega, omega0, t, fieldSpec, fieldTime = cnlse.run()

# plt.plot(t, np.abs(fieldTime[-1, :, 1])**2)