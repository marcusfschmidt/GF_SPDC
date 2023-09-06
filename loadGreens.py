#%%
import sys

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#Utility functions for symmetric fft and ifft
def fft(field):
    field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 0), axis = 0), axes = 0)
    field = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes = 1), axis = 1), axes = 1)
    return field

def ifft(field):
    field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 0), axis = 0), axes = 0)
    field = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field, axes = 1), axis = 1), axes = 1)
    return field

def addPaddingToWidth(array, paddingFactor = 0.25):
    #Extent the overall width of the arrays by a factor of paddingFactor
    arrayOut = np.copy(array)

    arrayOut[0] = array[0] - paddingFactor*(array[1]-array[0])
    arrayOut[1] = array[1] + paddingFactor*(array[1]-array[0])
    arrayOut[2] = array[2] - paddingFactor*(array[3]-array[2])
    arrayOut[3] = array[3] + paddingFactor*(array[3]-array[2])

    return arrayOut

def removeZeroValues(G, F, inputAxis, indexArray):
    x1, x2, y1, y2 = indexArray

    G = G[y1:y2, x1:x2]
    F = F[y1:y2, x1:x2]
    outAxisX = inputAxis[x1:x2]
    outAxisY = inputAxis[y1:y2]

    return G, F, outAxisX, outAxisY


# filename = "stitchedGreens_type0_gamma 1e-05_T0p 2_L 0.004.npy"
filename = "stitchedGreens_typeII_gamma 0.01_T0p 100_L 0.004.npy"

G_array, timeWidthArray, freqWidthArray, stitchTimes, t, omega, lambdaaxis, parametersArr = np.load(filename, allow_pickle=True)
timeWidthArray = addPaddingToWidth(timeWidthArray, 1)
freqWidthArray = addPaddingToWidth(freqWidthArray, 0.2)
G, F, tx, ty = removeZeroValues(G_array[0], G_array[1], t, timeWidthArray)

Gf = fft(G_array[0])
Ff = fft(G_array[1])
Gf, Ff, lx, ly = removeZeroValues(Gf, Ff, lambdaaxis, freqWidthArray)



#1x3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
#plot abs, real and imag of G_array[0]
ax1.imshow(np.abs(G), origin='lower', extent=[tx[0], tx[-1], ty[0], ty[-1]])
ax1.set_title('abs')
ax2.imshow(np.real(G), origin='lower', extent=[tx[0], tx[-1], ty[0], ty[-1]])
ax2.set_title('real')
ax3.imshow(np.imag(G), origin='lower', extent=[tx[0], tx[-1], ty[0], ty[-1]])
ax3.set_title('imag')

ax1.set_xlabel("t [ps]")
ax2.set_xlabel("t [ps]")
ax3.set_xlabel("t [ps]")
ax1.set_ylabel("t [ps]")

fig.suptitle("G(t, t')")


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
ax1.imshow(np.abs(Gf), origin='lower', extent=[lx[0], lx[-1], ly[0], ly[-1]])
ax1.set_title('abs')
ax2.imshow(np.real(Gf), origin='lower', extent=[lx[0], lx[-1], ly[0], ly[-1]])
ax2.set_title('real')
ax3.imshow(np.imag(Gf), origin='lower', extent=[lx[0], lx[-1], ly[0], ly[-1]])
ax3.set_title('imag')

ax1.set_xlabel("lambda [nm]")
ax2.set_xlabel("lambda [nm]")
ax3.set_xlabel("lambda [nm]")
ax1.set_ylabel("lambda [nm]")

fig.suptitle("G(lambda, lambda')")









