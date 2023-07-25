import numpy as np

class type0(object):

    def __init__(self, lambda_s, lambda_i, lambda_p, ordinaryAxisBool, temperature, QPMPeriod):

        """
        This calculates the wave vectors for collinear type II phase matching of 5% MgO doped LiNbO3
        The input are:
        lambda_s: signal wavelength in meters
        lambda_i: idler wavelength in meters 
        lambda_p: pump wavelength in meters 
        ordinaryAxisBool: True if polarized along the ordinary axis, False if polarized along the extraordinary axis
        temperature: temperature in degrees C of the crystal

        The calculated wave vectors are given in units of 1/m to match the units used in the propagation script.
        """

        
        #speed of light in m/ps
        c = 299792458*1e-12

        wlArray = np.array([lambda_p, lambda_s, lambda_i])*1e6
        minimumWl = np.min(wlArray)
        maximumWl = np.max(wlArray)

        wl = np.linspace(maximumWl, minimumWl, 5000)
        self.om = 2*np.pi*c/wl*1e6

        f = (temperature - 24.5)*(temperature+570.82)

        if ordinaryAxisBool:
            n = self.ordinaryAxis(wl, f)
        else:
            n = self.extraordinaryAxis(wl, f)

        beta = 2*np.pi/wl*n*1e6

        self.kp = beta
        self.ki = beta
        self.ks = beta
        self.QPMPeriod = QPMPeriod

        self.indistinguishableBool = True
        self.QPMbool = True



    def ordinaryAxis(self, lam, f):
            #5% doped MgO, ordinary axis
            a1 = 5.653
            a2 = 0.1185
            a3 = 0.2091
            a4 = 89.61
            a5 = 10.85
            a6 = 1.97e-2
            b1 = 7.941e-7
            b2 = 3.134e-8
            b3 = -4.641e-9
            b4 = -2.118e-6
            no2 = a1 + b1*f + (a2 + b2*f)/(lam**2 - (a3 + b3*f)**2) + (a4 + b4*f)/(lam**2 - a5**2) - a6*lam**2
            return np.sqrt(no2)

    def extraordinaryAxis(self, lam, f):
        #5% doped MgO, extraordinary axis
        a1 = 5.756
        a2 = 0.0983
        a3 = 0.2020
        a4 = 189.32
        a5 = 12.52
        a6 = 1.32e-2
        b1 = 2.860e-6
        b2 = 4.7e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
        ne2 = a1 + b1*f + (a2 + b2*f)/(lam**2 - (a3 + b3*f)**2) + (a4 + b4*f)/(lam**2 - a5**2) - a6*lam**2
        return np.sqrt(ne2)
    


        