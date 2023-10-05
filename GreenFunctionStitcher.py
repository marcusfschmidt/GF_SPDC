#%%
import sys

import SPDCNumerical_CPU as SPDCNumerical

import importlib
importlib.reload(SPDCNumerical)
import scipy as sp
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from numpy.linalg import svd
import typeII_beta_5perMgO_LN as betaFunctionTypeII
import type0_beta_5perMgO_LN as betaFunctionType0
from multiprocessing import Pool
import time   
import GreenFunctionsExtractorParallelized as gfep
importlib.reload(gfep)

from multiprocessing import set_start_method

import gc
gc.collect()


class GreenFunctionStitcher(object):

    def __init__(self, parametersArray, T0p, kmax, debugBool):
        self.parameterArray = parametersArray
        self.T0p = T0p
        self.kmax = kmax
        self.debugBool = debugBool
        self.gf = gfep.GreenFunctionsExtractor(kmax, debugBool)
        



    # def findInitialCenterTime(self, T0):
    #     solverObjectTemporary = SPDCNumerical.CoupledModes(*self.parameterArray)
    #     meanCenterTime = np.mean(solverObjectTemporary.timeShiftArray/2)*0

    #     # fastGF = gfep.GreenFunctionsExtractor(1, False)
    #     # fastGF.makeSolverParameters(self.parameterArray, SPDCNumerical.CoupledModes(*self.parameterArray))
    #     # fastGF.makePump(fastGF.solverObject.makeGaussianInput(self.T0p))
    #     # fastGF.makeBasisFunctions(T0, 0)
    #     # initialCenterTime = np.mean(fastGF.initOffset)*0
    #     # print(initialCenterTime2)
    #     # print(initialCenterTime)

    #     self.parameterArray[4] = meanCenterTime
    #     # del fastGF
    #     return meanCenterTime


    def defineSolver(self, solverParameters, T0, basisOffset):
        solverObject = SPDCNumerical.CoupledModes(*solverParameters)

        self.gf.makeSolverParameters(solverParameters, solverObject)
        # this is not modular; forcing a gaussian input. fix with string input eg
        self.gf.makePump(self.gf.solverObject.makeGaussianInput(self.T0p))
        # self.gf.makePump(self.pumpFunction(self.T0p))
        self.gf.makeBasisFunctions(T0, basisOffset)
        
        self.t = self.gf.t
        self.omega = self.gf.omega
        self.dt = self.gf.dt
        self.domega = self.gf.domega
        self.lambdaAxis = self.gf.lambdaAxis
        self.indistinguishableBool = self.gf.solverObject.beta.indistinguishableBool
        del solverObject

    

    def extractGreenFunctions(self, solverParameters, T0, basisOffset, checkBool = False):
        self.defineSolver(solverParameters, T0, basisOffset)


        G_array, overlaps, schmidtNumbers = self.gf.runExtractor(self.indistinguishableBool, checkBool)

        timeIndicesIdler, freqIndicesIdler = self.findWidth(G_array[0])
        t1x_idler, t2x_idler, _, _ = timeIndicesIdler
        timeIndices = timeIndicesIdler
        freqIndices = freqIndicesIdler
        

        offsetX1_idler = -(self.t[t1x_idler] - (-self.gf.initOffset[0]))
        offsetX2_idler = -(self.t[t2x_idler] - (-self.gf.initOffset[0]))
        width = np.abs(offsetX1_idler - offsetX2_idler)
        widthOffsetFromGlobalCenter = np.array([offsetX1_idler, offsetX2_idler])


        if not self.indistinguishableBool:
            timeIndicesSignal, freqIndicesSignal = self.findWidth(G_array[2])
            t1x_signal, t2x_signal, _, _ = timeIndicesSignal
            timeIndices = np.array([timeIndicesIdler, timeIndicesSignal])
            freqIndices = np.array([freqIndicesIdler, freqIndicesSignal])

        return G_array, overlaps, schmidtNumbers, self.gf.initOffset, widthOffsetFromGlobalCenter, width, timeIndices, freqIndices
    

    def findWidth(self, G):
        Gt = np.abs(G)**2
        Gf = np.abs(self.gf.fft(G))**2
        return self.widthHelperFunction(Gt), self.widthHelperFunction(Gf)


    def widthHelperFunction(self, array):
            threshold = np.mean(array)

            # Find the indices where the array values exceed the threshold
            non_zero_indices = np.where(array > threshold)

            # Extract the x and y coordinates
            x_coordinates = non_zero_indices[1]
            y_coordinates = non_zero_indices[0]

            # Get the bounding coordinates
            x1, x2 = np.min(x_coordinates), np.max(x_coordinates) 
            y1, y2 = np.min(y_coordinates), np.max(y_coordinates)

            return (x1, x2, y1, y2)


    def testOverlap(self, args):
        G, A_test, Ap_0, cnlse, basisIndex, newTimeAxis, timeOffset, plotbool = args

        if newTimeAxis is False:
            timeAxis = self.t
        else:
            timeAxis = newTimeAxis


        A_noise = np.zeros_like(A_test)
        if basisIndex == 0:
            initConditions = np.array([A_test, A_noise, Ap_0])
        elif basisIndex == 1:
            initConditions = np.array([A_noise, A_test, Ap_0])
        
        inIdx = int(basisIndex)
        outIdx = int(not(basisIndex))

        # parArray = self.gf.parametersArray
        # cnlse = SPDCNumerical.CoupledModes(*parArray)

        inputField, outputField = self.gf.runCNLSE(initConditions, cnlse)

        input = inputField[:, inIdx]
        output = outputField[:, outIdx]
        propagatedInput = outputField[:, inIdx]
        

        # greenOutput = np.sum(G*input,1)*self.dt
        # greenOutput = np.sum(G*input,1)*self.dt

        input = cnlse.ifft(cnlse.makeHermiteGaussianBasisFunctions(timeOffset, T0, 0, newTimeAxis = timeAxis))
        print("overlap func G size: " + str(np.shape(G)))
        print("overlap func input size: " + str(np.shape(input)))
        print("overap func timeaxis size: " + str(np.shape(self.t)))
        print("overlap func newtimeaxis size: " + str(np.shape(newTimeAxis)))

        greenOutput = np.sum(G*input,1)*self.dt
        overlap = self.gf.calcOverlap(greenOutput, output)
        if plotbool:
            # plt.figure()
            # plt.plot(timeAxis, np.abs(input)**2)
                      
            plt.figure()
            plt.plot(timeAxis, np.abs(greenOutput)**2)
            plt.plot(timeAxis, np.abs(output)**2)
            plt.xlim(0,3)
            # plt.plot(self.t, np.abs(propagatedInput)**2)

        #delete the variables to free up memory
        del input, output, propagatedInput, greenOutput, cnlse
        return overlap
    
    def validatePropagation(self, G, A_test, Ap, cnlse, threshold, basisIndex = 0, newTimeAxis = False, timeOffset = 0, plotBool = False):
        argList = (G, A_test, Ap, cnlse, basisIndex, newTimeAxis, timeOffset, plotBool)
        overlap = self.testOverlap(argList)

        returnBool = True
        if overlap < threshold:
            returnBool = False
        return returnBool, overlap

    
    def makeTestFunction(self, offset, T0, newTimeAxis = False):
        # temporaryParametersArray = np.copy(self.parameterArray)
        # temporaryParametersArray[3] = timeAxisOffset 
        # temporarySolverObject = SPDCNumerical.CoupledModes(*temporaryParametersArray)

        testFunction = self.gf.solverObject.makeHermiteGaussianBasisFunctions(offset, T0, 0, newTimeAxis = newTimeAxis)
        # del temporarySolverObject
        # del temporaryParametersArray
        return testFunction    

    
    def removeZeroValues(self, G, F, inputAxis, indexArray):
        x1, x2, y1, y2 = indexArray

        G = G[y1:y2, x1:x2]
        F = F[y1:y2, x1:x2]

        outAxisX = inputAxis[x1:x2]
        outAxisY = inputAxis[y1:y2]

        return G, F, outAxisX, outAxisY
    


    def stitchGreenFunctions(self, initOffset_old, initOffset_new, G_old, G_new):

        newTime = np.arange(self.old_t[0], self.t[-1], self.dt)

        if not self.indistinguishableBool:
            Gis_old, G_ii_old, G_si_old, G_ss_old = G_old
            Gis_new, G_ii_new, G_si_new, G_ss_new = G_new

            t1_idler, t1_signal = initOffset_old
            t2_idler, t2_signal = initOffset_new

            G_is, G_ii = self.stitchHelperFunction(Gis_old, G_ii_old, Gis_new, G_ii_new, t1_idler, t2_idler, newTime)
            G_si, G_ss = self.stitchHelperFunction(G_si_old, G_ss_old, G_si_new, G_ss_new, t1_signal, t2_signal, newTime)
            return (G_is, G_ii, G_si, G_ss), newTime
        else:
            G_old, F_old = G_old
            G_new, F_new = G_new
            t1, _ = initOffset_old
            t2, _ = initOffset_new
            G, F = self.stitchHelperFunction(G_old, F_old, G_new, F_new, t1, t2, newTime)
            return (G, F), newTime

    def stitchHelperFunction(self, G1, F1, G2, F2, t1, t2, newTime):

        # print(newTime.size - G1.shape[0])
        # print(newTime[0], newTime[-1])
        # print(self.t[0], self.t[-1])

        G1 = np.pad(G1, ((newTime.size - G1.shape[0], 0), (newTime.size - G1.shape[1], 0)), 'constant')
        G2 = np.pad(G2, ((newTime.size - G2.shape[0], 0), (newTime.size - G2.shape[1], 0)), 'constant')
        F1 = np.pad(F1, ((newTime.size - F1.shape[0], 0), (newTime.size - F1.shape[1], 0)), 'constant')
        F2 = np.pad(F2, ((newTime.size - F2.shape[0], 0), (newTime.size - F2.shape[1], 0)), 'constant')

        # print(np.shape(G1))



        abs_diff1 = np.abs(newTime + t1)  
        abs_diff2 = np.abs(newTime + t2)

        condition = abs_diff1 < abs_diff2

        G = np.where(condition, G1, G2)
        F = np.where(condition, F1, F2)

        return G, F

    
    def compareWidthArraysHelperFunction(self, array1, array2):
        arrayOut = np.copy(array1)
        arrayOut[0] = min(array1[0], array2[0])
        arrayOut[1] = max(array1[1], array2[1])
        
        arrayOut[2] = min(array1[2], array2[2])
        arrayOut[3] = max(array1[3], array2[3])

        return arrayOut 
    
    def compareWidthArrays(self, timeArray_old, timeArray_new, freqArray_old, freqArray_new):

        if self.indistinguishableBool:
            return self.compareWidthArraysHelperFunction(timeArray_old, timeArray_new), self.compareWidthArraysHelperFunction(freqArray_old, freqArray_new)
        else:
            timeArray_old_idler = timeArray_old[0]
            timeArray_new_idler = timeArray_new[0]
            timeArrayIdler = self.compareWidthArraysHelperFunction(timeArray_old_idler, timeArray_new_idler)

            timeArray_old_signal = timeArray_old[1]
            timeArray_new_signal = timeArray_new[1]
            timeArraySignal = self.compareWidthArraysHelperFunction(timeArray_old_signal, timeArray_new_signal)

            freqArray_old_idler = freqArray_old[0]
            freqArray_new_idler = freqArray_new[0]
            freqArrayIdler = self.compareWidthArraysHelperFunction(freqArray_old_idler, freqArray_new_idler)

            freqArray_old_signal = freqArray_old[1]
            freqArray_new_signal = freqArray_new[1]
            freqArraySignal = self.compareWidthArraysHelperFunction(freqArray_old_signal, freqArray_new_signal)

            return (timeArrayIdler, timeArraySignal), (freqArrayIdler, freqArraySignal)

    
    def addPaddingToWidth(self, array, paddingFactor = 0.25):
        #Extent the overall width of the arrays by a factor of paddingFactor
        arrayOut = np.copy(array)

        arrayOut[0] = array[0] - paddingFactor*(array[1]-array[0])
        arrayOut[1] = array[1] + paddingFactor*(array[1]-array[0])
        arrayOut[2] = array[2] - paddingFactor*(array[3]-array[2])
        arrayOut[3] = array[3] + paddingFactor*(array[3]-array[2])

        return arrayOut
    
    def outputWidthArray(self, widths):
        if self.indistinguishableBool:
            return widths 
        else:
            x1a, x2a, y1a, y2a = widths[0]
            x1b, x2b, y1b, y2b = widths[1]

            #take min of 1 values and max of 2 values
            x1 = min(x1a, x1b)
            x2 = max(x2a, x2b)
            y1 = min(y1a, y1b)
            y2 = max(y2a, y2b)
            return (x1, x2, y1, y2)
        
    def stitchTimeHelper(self, lowHighIndex):
        if lowHighIndex == 0:
            return 1
        else:
            return -1

    def iterativeStitch(self, G_array, centerTimeInitial, widthOffsetInitial, width, validationThreshold, timeWidthArray, freqWidthArray, lowHighIndex=0, signalIdlerTestIndex=0):
        if self.indistinguishableBool:
            testIndex = 0
        else:
            testIndex = 2 if signalIdlerTestIndex == 1 else 0

        widthOffsetFromGlobalCenter = np.copy(widthOffsetInitial)
        centerTime = np.copy(centerTimeInitial)

        print("Iteratively stitching Green's functions...")

        stitchTimes = []
        testcount = 0
        newTimeAxis = False
        while testcount < 1:
            gc.collect()

            testOffset = centerTime[signalIdlerTestIndex] + self.stitchTimeHelper(lowHighIndex)*width/2
            if testOffset < self.t[0] or testOffset > self.t[-1]:
                print("Offset exceeds time axis, stitching complete. Inspect the output to determine if the time window should be extended.")
                break

            timeAxis = self.t
            A_test = self.makeTestFunction(testOffset, T0, newTimeAxis=timeAxis)
            Ap = self.gf.Ap_0
            solverObject = self.gf.solverObject
            
            _, overlap = self.validatePropagation(G_array[testIndex], A_test, Ap, solverObject, validationThreshold, signalIdlerTestIndex, timeAxis, testOffset, plotBool=False)
            self.gf.debugPrint("Test overlap at edge of region: " + str(overlap))
            if overlap > 0.999:
                print("Test overlap is greater than 99.9%, stitching complete.")
                break

            # plt.figure()
            # plt.title("First test")
            # plt.imshow(np.abs(G_array[0]), extent=[self.t[0], self.t[-1], self.t[0], self.t[-1]], origin='lower')
            # plt.xlim(-8,0)
            # plt.ylim(-2,5)
            # plt.plot(self.t, np.abs(self.gf.solverObject.ifft(A_test)))
            # plt.show(block=False)
            # plt.pause(0.001)
       

            #### Check Green functions at the stitching points ###
            stitchMoveBool = False
            while True:

                print("Extracting new Green's functions...")
                timeAxisOffset = np.mean(centerTime)
                self.parameterArray[3] = timeAxisOffset


                G_array_new, _, _, centerTime_new, widthOffsetFromNewCenter, width_new, timeWidthArray_new, freqWidthArray_new = gs.extractGreenFunctions(self.parameterArray,T0,widthOffsetFromGlobalCenter[lowHighIndex], checkBool=False)
                stitchHalfWidth = (centerTime_new[signalIdlerTestIndex] - centerTime[signalIdlerTestIndex])/2
                stitchTime = testOffset - stitchHalfWidth

                offsetTimeAxis = self.t
                A_stitchTestOld = self.makeTestFunction(stitchTime, T0, newTimeAxis = timeAxis)
                A_stitchTestNew = self.makeTestFunction(stitchTime, T0, newTimeAxis = offsetTimeAxis)
                Ap_new = self.gf.Ap_0
                newSolverObject = self.gf.solverObject

                plt.figure()
                plt.title("new G")
                plt.imshow(np.abs(G_array_new[0]), extent=[self.t[0], self.t[-1], self.t[0], self.t[-1]], origin='lower')
                plt.xlim(-8,0)
                plt.ylim(-2,5)
                plt.plot(self.t, np.abs(self.gf.solverObject.ifft(A_stitchTestNew)))
                plt.vlines(-stitchTime, self.t[0], self.t[-1])
                plt.show(block=False)
                plt.pause(0.001)

                if stitchMoveBool:
                    stitchOffset = stitchTime
                    # A_test = A_stitchTest 
                    print("Stitching point moved, edge region redefined as center of new Green's functions. Continuing...")
                    break
                
                # ###########################################################s
                 

                ### problems here, very low overlap for the old. 
                ### the basis function offsets fed to the extract green functions fucks up when another offset is added to the time axis. so I need to adjust the widthOffsetfromglobalcenter variable
                _, stitchOverlapOld = self.validatePropagation(G_array[testIndex], A_stitchTestOld, Ap, solverObject, validationThreshold, signalIdlerTestIndex, timeAxis, stitchTime, plotBool=True)
                _, stitchOverlapNew = self.validatePropagation(G_array_new[testIndex], A_stitchTestNew, Ap_new, newSolverObject, validationThreshold, signalIdlerTestIndex, offsetTimeAxis, stitchTime, plotBool=True)
                # plt.figure()
                # plt.title("old G")
                # plt.imshow(np.abs(G_array[0]), extent=[self.t[0], self.t[-1], self.t[0], self.t[-1]], origin='lower')
                # plt.xlim(-8,0)
                # plt.ylim(-2,5)
                # plt.plot(self.t, np.abs(self.gf.solverObject.ifft(A_stitchTestOld)))
                # plt.vlines(-stitchTime, self.t[0], self.t[-1])
                # plt.show(block=False)
                # plt.pause(0.001)
                
                # plt.figure()
                # plt.title("First test")
                # self.old_t = timeAxis
                # plt.imshow(np.abs(G_array[0]), extent=[self.old_t[0], self.old_t[-1], self.old_t[0], self.old_t[-1]], origin='lower')
                # plt.xlim(-3,-2)
                # plt.ylim(0,2.5)
                # #equal aspect ratio
                # plt.gca().set_aspect('equal', adjustable='box')
                # plt.plot(self.old_t, np.abs(self.gf.solverObject.ifft(A_stitchTestOld)))
                # plt.vlines(-stitchTime, self.old_t[0], self.old_t[-1])
                # plt.show(block=False)
                # plt.pause(0.001)

                print("")
                print("overlaps old and new at stitching point: " + str(stitchOverlapOld) + ", " + str(stitchOverlapNew))
                print("")

                #if the difference between old and new stitch overlaps is within 3%, break the loop

                if abs(stitchOverlapNew - stitchOverlapOld) < 0.01:
                    print("New Green's function provides a decent solution at the stitching point, continuing.")
                    break
                

                widthOffsetFromGlobalCenter -= stitchHalfWidth
                stitchMoveBool = True
        
            #### End check Green functions at the stitching points ###
            stitchTimes.append(-stitchTime)
            # Compare the time arrays to get minimum and maximum extent of the new Green's functions
            timeWidthArray_out, freqWidthArray_out = self.compareWidthArrays(timeWidthArray, timeWidthArray_new, freqWidthArray, freqWidthArray_new)
           
            print("Stitching Green's functions...")
            G_array_updated, newTimeAxis = self.stitchGreenFunctions(centerTime, centerTime_new, G_array, G_array_new)
            del G_array_new
            
            A_test = self.makeTestFunction(testOffset, T0, newTimeAxis)
            _, overlap_new = self.validatePropagation(G_array_updated[testIndex], A_test, validationThreshold, signalIdlerTestIndex, newTimeAxis, testOffset, plotBool=False)
            gs.gf.debugPrint("New validation overlap at edge of region: " + str(overlap_new))
            # plt.figure()
            # plt.title("Second test, stitching moved: " + str(stitchMoveBool))
            # plt.imshow(np.abs(G_array_updated[0]), extent=[t[0], t[-1], t[0], t[-1]], origin='lower')
            # plt.xlim(-8,0)
            # plt.ylim(-2,5)
            # plt.plot(newTimeAxis, np.abs(self.gf.solverObject.ifft(A_test)))
            # plt.show(block=False)
            # plt.pause(0.001)


            if overlap_new < 0.025:
                print("New overlap approaching zero, stitching complete.")
                print("#################################################\n")
                # break
                # return G_array_updated, timeWidthArray_out, freqWidthArray_out, stitchTimes
                return G_array, timeWidthArray, freqWidthArray, stitchTimes
            else:
                testcount += 1
                print("Green's function extended by stitching, attempting again...")
                widthOffsetFromGlobalCenter += widthOffsetFromNewCenter
                width = width_new
                G_array = G_array_updated
                centerTime = centerTime_new
                timeWidthArray = timeWidthArray_out
                freqWidthArray = freqWidthArray_out

        return G_array, timeWidthArray, freqWidthArray, stitchTimes
    








if __name__ == '__main__':

    ###################################################
    ###### CME parameters ######  
    #Number of grid points 
    n = 12
    #Time step and spatial step. The spatial step will be adjusted slightly depending on the crystal length
    dt = 0.5e-2
    dz = 0.2e-3
    #Relative tolerance and number of steps for the adaptive step size
    rtol = 1e-4
    nsteps = 10000
    #Print the progress of the solver
    printBool = False
    #Pump and signal wavelengths
    lambda_p = 532e-9
    lambda_s = 1064e-9
    #Calculate the idler wavelength from energy conservation
    c = 299792458e-12
    om_p = 2*np.pi*c/lambda_p
    om_s = 2*np.pi*c/lambda_s
    om_i = om_p - om_s
    lambda_i = 2*np.pi*c/om_i

    #Attenuation coefficients (i.e. dA ~ -alpha*A)
    #For QPM on, they must be identical.
    alpha_s = 0
    alpha_i = alpha_s

    #Crystal length
    L = 4000e-6

    # Define the beta function for type II phase matching
    beta = betaFunctionTypeII.typeII(lambda_s, lambda_i, lambda_p)

    # Define the beta function for type 0 phase matching
    QPMPeriod = 5.916450343734758e-6
    beta = betaFunctionType0.type0(lambda_s, lambda_i, lambda_p, ordinaryAxisBool=True, temperature=36, QPMPeriod=QPMPeriod)

    #Nonlinear coefficient
    gamma = 1e-6
    #Pump pulse duration in ps
    T0p = 2

    #Define numerical frequencies
    omega_s = -(om_p - om_s)
    omega_i = -(om_p - om_i)
    #Define the parameters array for the solver
    parametersArr = np.array([n, dt, dz, 0, L, beta, gamma, lambda_p, omega_s, omega_i, alpha_s, alpha_i, printBool, rtol, nsteps])

    #### Green's function parameters ####
    #Define the number of basis functions to be used in the Green's function extraction
    kmax = 75

    # #Define the basis functions (hermite gaussians) to be used in the Green's function extraction
    # T0 = T0p/5

    # T0 = 1.288/4
    T0 = T0p/22

    gs = GreenFunctionStitcher(parametersArr, T0p, kmax, debugBool = True)

    print("Extracting initial Green's functions...")
    t1 = time.time()
    G_array, o, s, centerTime, widthOffsetInitial, initWidth, timeWidthArray, freqWidthArray = gs.extractGreenFunctions(parametersArr, T0, 0, checkBool = False)
    print("Green's function extraction took " + str(time.time() - t1) + " seconds.")
    validationThreshold = 0.98/2
    
    solverObject = gs.gf.solverObject
    A_test_init = gs.makeTestFunction(centerTime[0], T0)
    Ap_0 = gs.gf.Ap_0
    validationBool, initOverlap = gs.validatePropagation(G_array[0], A_test_init, Ap_0, solverObject, validationThreshold, 0, gs.t, centerTime[0], plotBool = True)
    print("WU\n")
    print("Calculated initial validation overlap: " + str(initOverlap))
    if not validationBool:
        print("Validation failed with threshold of " + str(validationThreshold) + ". Consider increasing the number of basis functions or the time resolution.")
        sys.exit()

    G_array, timeWidthArray, freqWidthArray, stitchTimes1 = gs.iterativeStitch(G_array, centerTime, widthOffsetInitial, initWidth, validationThreshold, timeWidthArray, freqWidthArray, 0)
    # print("######################################### Changing direction #########################################")
    # G_array, timeWidthArray, freqWidthArray, stitchTimes2 = gs.iterativeStitch(G_array, centerTime, widthOffsetInitial, initWidth, validationThreshold, timeWidthArray, freqWidthArray, 1)
    # stitchTimes = stitchTimes1 + stitchTimes2
    # timeWidthArrayOutput = gs.outputWidthArray(timeWidthArray)
    # freqWidthArrayOutput = gs.outputWidthArray(freqWidthArray)
    # saveArray = np.array([G_array, timeWidthArrayOutput, freqWidthArrayOutput, stitchTimes, gs.t, gs.omega, gs.lambdaAxis, parametersArr], dtype=object)
    # typeString = "type0" if beta.QPMbool else "typeII"
    # saveString = "stitchedGreens_" + typeString + "_gamma " + str(gamma) + "_T0p " + str(T0p) + "_L " + str(L)
    # # np.save(saveString, saveArray)


        #%%


    plt.figure()
    plt.imshow(np.abs(G_array[0]), extent=[gs.t[0], gs.t[-1], gs.t[0], gs.t[-1]], origin='lower')
    #vertical line at fuck
    # plt.vlines(fuck, gs.t[0], gs.t[-1])
    # plt.hlines(fuck, gs.t[0], gs.t[-1])

    # plt.xlim(-2 + fuck, fuck + 2)
    # plt.ylim(-2 + fuck, fuck + 2)

