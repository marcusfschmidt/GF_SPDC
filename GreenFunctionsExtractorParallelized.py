#%%
#reload all module
import sys
import importlib
import scipy as sp
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from numpy.linalg import svd
import typeII_beta_5perMgO_LN as betaFunctionTypeII
import type0_beta_5perMgO_LN as betaFunctionType0
import timeit
from multiprocessing import Pool

class GreenFunctionsExtractor(object):

    #Initialize the Green's function extractor object
    def __init__(self, kmax, debugBool = True):
        self.kmax = kmax
        self.debugBool = debugBool

    #Utility functions for symmetric fft and ifft
    def fft(self, field):
        field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 0), axis = 0), axes = 0)
        field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 1), axis = 1), axes = 1)
        # field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 0), axis = 0), axes = 0)*self.dt/np.sqrt(2*np.pi)
        # field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 1), axis = 1), axes = 1)*self.dt/np.sqrt(2*np.pi)
        return field

    def ifft(self, field):
        field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 0), axis = 0), axes = 0)
        field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 1), axis = 1), axes = 1)
        # field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 0), axis = 0), axes = 0)*np.sqrt(2*np.pi)/self.dt
        # field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 1), axis = 1), axes = 1)*np.sqrt(2*np.pi)/self.dt
        return field

    def debugPrint(self, *args, **kwargs):
        if self.debugBool:
            print(*args, **kwargs)

    #Function to make a generic solver object and save the parameters as a class attribute
    #Note that the solver objects are not used in the Green's function extraction, since a new solver object is made for each basis function in order to parallelize the extraction 
    def makeSolverParameters(self, parametersArray, solverObject):
        #Save the parameters as a class attribute
        self.parametersArray = parametersArray

        #Generic solver object
        self.solverObject = solverObject
        self.timeLen = self.solverObject.N
        self.t = self.solverObject.t
        self.omega = self.solverObject.omega
        self.lambdaAxis = self.solverObject.lambdaRealS
        self.dt = self.solverObject.dt
        self.domega = self.solverObject.domega

        self.timeShiftArray = self.solverObject.timeShiftArray

    #Function to make the pump field, typically using the solver object methods
    def makePump(self, field):
        self.Ap_0 = field

    #Define basis functions to be used in the Green's function extraction
    def makeBasisFunctions(self, T0, basisFunctionOffset = 0):
        self.A_basis = np.zeros((self.kmax, self.timeLen, 2), dtype = complex)
        self.T0 = T0
        self.basisFunctionOffset = basisFunctionOffset

        betaspl = self.solverObject.timeShiftArray[0]/2
        betaipl = self.solverObject.timeShiftArray[1]/2
        self.initOffset = np.array([-betaipl + basisFunctionOffset, -betaspl + basisFunctionOffset])

        try:
            for k in range(self.kmax):
                self.A_basis[k,:,0] = self.solverObject.makeHermiteGaussianBasisFunctions(self.initOffset[0], T0, k)
                self.A_basis[k,:,1] = self.solverObject.makeHermiteGaussianBasisFunctions(self.initOffset[1], T0, k)

        except:
            print("Warning: temporal widths too small for the amount of basis functions requested, error at k = {}".format(k))
            print("Exiting...")
            sys.exit()
    
    def runCNLSE(self, initConditions, cnlse):
        cnlse.setInitialConditions(initConditions)

        #Run the solver
        _,_,_,_,_, fieldTime = cnlse.run()
        
        #Define the input, output, and propagated input fields
        output = fieldTime[-1, :, :]
        input = fieldTime[0, :, :]

        return input, output



  
    #Function for parallel processing of the field propagation
    def parallelExtract(self, args):
        k, basisIndex, A_basis, A_noise, Ap_0, outIdx, inIdx, parArray = args
        
        #Make the solver object for each basis function by copying the generic solver object
        cnlse = self.solverObject
        #Set initial conditions
        if basisIndex == 0:
            initConditions = np.array([A_basis[k,:,0], A_noise, Ap_0])
        elif basisIndex == 1:
            initConditions = np.array([A_noise, A_basis[k,:,1], Ap_0])
        
        input, output = self.runCNLSE(initConditions, cnlse)

        input_out = input[:, inIdx]
        output_out = output[:, outIdx]
        propagatedInput_out = output[:, inIdx]
        print("Finished k = {}".format(k))


        return input_out, output_out, propagatedInput_out

    
    #Function for extracting the Schmidt modes
    def extractSchmidtModes(self, basisIndex):
        Ap_0 = self.Ap_0
        A_noise = np.zeros_like(Ap_0)

        #Get the time shifts needed for the basis functions decomposition
        crossTime = self.timeShiftArray[basisIndex]/2
        selfTime = self.timeShiftArray[int(not(basisIndex))]/2
        kmax = self.kmax

        inIdx = int(basisIndex)
        outIdx = int(not(basisIndex))

        #Arguments for the parallel processing extraction
        args_list = [(k, basisIndex, self.A_basis, A_noise, Ap_0, outIdx, inIdx, self.parametersArray) for k in range(kmax)]

        #Parallel processing
        pool = Pool()
        with pool as p:
            results = p.map(self.parallelExtract, args_list)
            p.close()
            p.join()

        #Initialize arrays for the time shifted basis functions
        B_cross = np.zeros((kmax, self.timeLen), dtype = complex)
        B_self = np.zeros((kmax, self.timeLen), dtype = complex)
        #Populate the basis function arrays

        for n in range(kmax):
            B_cross[n,:] = self.solverObject.makeHermiteGaussianBasisFunctions(crossTime + self.basisFunctionOffset, self.T0, n, fftBool = False)
            B_self[n,:] = self.solverObject.makeHermiteGaussianBasisFunctions(selfTime + self.basisFunctionOffset, self.T0, n, fftBool = False)
        
        #Initialize arrays for the Schmidt modes
        phi = np.zeros_like(B_self)
        psi = np.zeros_like(phi)
        phi_self = np.zeros_like(phi)
        psi_self = np.zeros_like(phi)

        #Extract the input, output, and propagated input fields 
        fieldTimeArray = np.zeros((3, kmax, self.timeLen), dtype = complex)
        fieldTimeArray[0,:,:], fieldTimeArray[1,:,:], fieldTimeArray[2,:,:] = zip(*results)

        #Define the matrices w/ basis coefficients
        G_cross = np.dot(fieldTimeArray[1], np.conjugate(B_cross.T)) * self.dt
        G_self = np.dot(fieldTimeArray[2], np.conjugate(B_self.T)) * self.dt
        self.debugPrint("")

        #Extract Schmidt modes
        u, rho, v_conjugate, = svd(G_cross)
        v = np.conjugate(v_conjugate).T

        u_self, rho_self, v_conjugate_self, = svd(G_self)
        v_self = np.conjugate(v_conjugate_self).T

        u = np.conjugate(u)
        u_self = np.conjugate(u_self)
        v = np.conjugate(v)
        v_self = np.conjugate(v_self)

        phi = np.matmul(u.T, B_self)
        phi_self = np.matmul(u_self.T, B_self)
        psi = np.matmul(v.T, B_cross)
        psi_self = np.matmul(v_self.T, B_self)
                
        outputBasis = [B_cross, B_self]
        self.fieldTimeArray = fieldTimeArray

        return phi, rho, psi, phi_self, rho_self, psi_self, fieldTimeArray, crossTime, outputBasis

    
    #Function for parallel processing of the Green's function extraction
    def parallelG(self, args):
        v, rho, u = args

        # Construct the Green's function from the Schmidt modes
        G = np.tensordot(v * rho[:, np.newaxis], np.conjugate(u), axes=([0], [0]))
        return G

    
    #Function for extracting the Green's functions, changing depending on distinguishable or indistinguishable photons
    def extractGreenFunctions(self, args, indistinguishableBool):
        self.debugPrint("Extracting Green's functions...", end = "\n")
        
        if indistinguishableBool:
            us, vi, uss, vss, rhoCross, rhoSelf = args
            argsList = [(vi, rhoCross, us), (vss, rhoSelf, uss)]

        if not indistinguishableBool:
            us, vs, ui, vi, uss, vss, uii,vii, rhoCross, rhoSelf = args
            argsList = [(vi, rhoCross, us), (vii, rhoSelf, uii), (vs, rhoCross, ui), (vss, rhoSelf, uss)]

        with Pool() as p:
            results = p.map(self.parallelG, argsList)
            p.close()
            p.join()
        
        self.debugPrint("Green's functions extracted.")
        return results


    ############################## 
    # Derived quantities and checks #
    def photonNumber(self, crossGreenTime):
        G = crossGreenTime

        #photon number n(t)
        n_t = np.sum(np.abs(G)**2,1)

        #total photon number N
        N = np.sum(n_t)*self.dt
        return N


    def parallelOverlap(self, args):
        G_cross, G_self, input, output, propagatedInput = args

        greenOutput = np.sum(G_cross*input,1)*self.dt
        overlap_cross = self.calcOverlap(greenOutput, output)

        input = np.conjugate(input)
        greenOutput = np.sum(G_self*input,1)*self.dt
        overlap_self = self.calcOverlap(greenOutput, propagatedInput)
        
        return overlap_cross, overlap_self


    #Helper function for calculating the overlap
    def calcOverlap(self, gout, out):
        gout = gout/np.sqrt(np.sum(np.abs(gout)**2)*self.dt)
        out = out/np.sqrt(np.sum(np.abs(out)**2)*self.dt)
        
        #now calculate overlap
        overlap = np.sum(np.conjugate(gout)*out)*self.dt
        overlap = np.abs(overlap)**2
        return overlap


    #Function for calculating the overlap between the output and the input integrated with the Green's functions
    def calculateGreenOverlap(self, G_cross, G_self, fieldArray):
        overlapArray = np.zeros((fieldArray.shape[1], 2))
        argslist = [(G_cross, G_self, fieldArray[0, k, :], fieldArray[1, k, :], fieldArray[2, k, :]) for k in range(len(overlapArray))]

        with Pool() as p:
            results = p.map(self.parallelOverlap, argslist)
            p.close()
            p.join()

        #transform the results into an array
        results = np.array(results)
        return results 
    

    
    #Function to check the Schmidt numbers and see if they are close to 1
    def checkSchmidtNumbers(self, rho, nu):
        schmidtArray = rho**2 - nu**2
        return schmidtArray
    
    ##############################

    def runExtractor(self, indistinguishableBool, checkBool):
        overlaps = None
        schmidtNumbers = None
        
        #Extract the Schmidt modes for both the signal and idler  
        self.debugPrint("\nPropagating signal...", end='\n')
        u_s,idlerRho,v_i,uss, selfRhos, vss, isTimeArray, ti, signalBasis = self.extractSchmidtModes(0)
        self.t_signalPropagated = 2*ti
        # self.timeArray_signalPropagated = isTimeArray
    
    #If using type II phasematching, also propagate the idler
        if not indistinguishableBool:
            self.debugPrint("Propagating idler...", end = "\n")
            u_i,signalRho,v_s, uii, selfRhoi, vii, siTimeArray, ts, idlerBasis = self.extractSchmidtModes(1)
            self.t_idlerPropagated = 2*ts
            # self.timeArray_idlerPropagated = siTimeArray

            argsList = (u_s, v_s, u_i, v_i, uss, vss, uii, vii, signalRho, selfRhos)

            #Extract the Green's functions from the Schmidt modes
            G_is, G_ii, G_si, G_ss = self.extractGreenFunctions(argsList, indistinguishableBool)
            #roll the Green's functions on the x axis to transform to input time
            G_is = np.roll(G_is, int((2*ts)/self.dt), axis = 1)
            G_ss = np.roll(G_ss, int((2*ts)/self.dt), axis = 1)


            G_ii = np.roll(G_ii, int(2*ti/self.dt), axis = 1)
            G_si = np.roll(G_si, int(2*ti/self.dt), axis = 1)

            G_tup = (G_is, G_ii, G_si, G_ss)

            if checkBool: 
                #Check the photon numbers
                print("Photon number from G_is: {}".format(self.photonNumber(G_is)))
                print("Photon number from G_si: {}".format(self.photonNumber(G_si)))
                print("Absolute difference: {}".format(np.abs(self.photonNumber(G_is) - self.photonNumber(G_si))))
                print("Calculating overlaps and Schmidt numbers...")
                
                #Check the overlaps
                overlapsSignalPropagation = self.calculateGreenOverlap(G_si, G_ii, siTimeArray)
                overlapsIdlerPropagation = self.calculateGreenOverlap(G_is, G_ss, isTimeArray)
                #concatenate the overlaps into a single array
                overlaps = np.concatenate((overlapsSignalPropagation, overlapsIdlerPropagation), axis = 1)   

                #Check the Schmidt numbers
                schmidtNumbersSignalPropagation = self.checkSchmidtNumbers(selfRhos, idlerRho)
                schmidtNumbersIdlerPropagation = self.checkSchmidtNumbers(selfRhoi, signalRho)

                #concatenate the Schmidt numbers into a single array
                schmidtNumbers = np.vstack((schmidtNumbersSignalPropagation, schmidtNumbersIdlerPropagation)).T

                print("Finished!")
            

        #If QPM is on, photons are indistinguishable and the idler is not propagated
        else:
            argsList = (u_s, v_i, uss, vss, idlerRho, selfRhos)
            #Extract the Green's functions from the Schmidt modes

            G, F = self.extractGreenFunctions(argsList, indistinguishableBool)
            #roll the Green's functions on the x axis to transform to input time
            G = np.roll(G, int(2*ti/self.dt), axis = 1)
            F = np.roll(F, int(2*ti/self.dt), axis = 1)

            G_tup = (G, F)

            if checkBool:
                #Check the photon numbers
                print("Photon number from G: {}".format(self.photonNumber(G)))
                print("Calculating overlaps and Schmidt numbers...")
                overlaps = self.calculateGreenOverlap(G, F, isTimeArray)

                #Check the Schmidt numbers
                schmidtNumbers = self.checkSchmidtNumbers(selfRhos, idlerRho)
                print("Finished!")
        return G_tup, overlaps, schmidtNumbers

            

# if __name__ == "__main__":
#     ###################################################
#     ###### CME parameters ######  
#     #Number of grid points 
#     n = 13
#     #Time step and spatial step. The spatial step will be adjusted slightly depending on the crystal length
#     dt = 0.002
#     dz = 0.2e-3
#     #Relative tolerance and number of steps for the adaptive step size
#     rtol = 1e-3
#     nsteps = 10000
#     #Print the progress of the solver
#     printBool = False

#     #Pump and signal wavelengths
#     lambda_p = 532e-9
#     lambda_s = 1064e-9
#     #Calculate the idler wavelength from energy conservation
#     c = 299792458e-12
#     om_p = 2*np.pi*c/lambda_p
#     om_s = 2*np.pi*c/lambda_s
#     om_i = om_p - om_s
#     lambda_i = 2*np.pi*c/om_i

#     #Attenuation coefficients (i.e. dA ~ -alpha*A)
#     #For QPM on, they must be identical.
#     alpha_s = 0
#     alpha_i = alpha_s

#     #Crystal length
#     L = 4000e-6

#     # # Define the beta function for type II phase matching
#     # beta = betaFunctionTypeII.typeII(lambda_s, lambda_i, lambda_p)

#     # Define the beta function for type 0 phase matching
#     QPMPeriod = 5.916450343734758e-6
#     beta = betaFunctionType0.type0(lambda_s, lambda_i, lambda_p, ordinaryAxisBool=True, temperature=36, QPMPeriod=QPMPeriod)
    

#     #Booleans for QPM and indistinguishability. Must be defined in the beta class
#     QPMbool = beta.QPMbool
#     indistinguishableBool = beta.indistinguishableBool


#     #Nonlinear coefficient
#     gamma = 1e-5
#     #Pump pulse duration in ps
#     T0p = 2

#     #Define numerical frequencies
#     omega_s = -(om_p - om_s)
#     omega_i = -(om_p - om_i)
#     #Define the parameters array for the solver
#     parametersArr = np.array([n, dt, dz, L, beta, gamma, lambda_p, omega_s, omega_i, alpha_s, alpha_i, QPMbool, printBool, rtol, nsteps])

#     #### Green's function parameters ####
#     #Define the number of basis functions to be used in the Green's function extraction
#     kmax = 50
#     jmax = kmax
   
#     #Check the photon number and overlap of the Green's functions
#     checkBool = False
#     ###################################################

#     # Make the Green's function extractor object
#     gf = GreenFunctionsExtractor(kmax, jmax, 3.7586/2)
#     # Make the solver object and pump field
#     gf.makeSolverParameters(parametersArr)
#     gf.makePump(gf.solverObject.makeGaussianInput(T0p))

#     #k 150 and divide by 11.5 is ok

#     #Define the basis functions (hermite gaussians) to be used in the Green's function extraction
#     T0pArr = np.array([T0p/11.5*(-k*(0/kmax) + 1) for k in range((kmax))])
#     gf.makeBasisFunctions(T0pArr)
#     G_array, overlaps, schmidtNumbers = gf.runExtractor(indistinguishableBool, checkBool)


#     if indistinguishableBool:
#         G2, F2 = G_array
#     else:
#         G_is, G_ii, G_si, G_ss = G_array
# #%%

# t1 = gf.timeShiftArray[0]/2 - 0
# t2 = gf.timeShiftArray[0]/2 - 3.7586/2

# newarr = np.zeros_like(G2)
# for n,t in enumerate(gf.t):
#     if np.abs(t-t1) < np.abs(t-t2):
#         newarr[:,n] = G1[:,n]
#     else:
#         newarr[:,n] = G2[:,n]


