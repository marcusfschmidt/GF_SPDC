import numpy as np

class typeII(object):

    def __init__(self, lambda_s, lambda_i, lambda_p):

        """
        This calculates the wave vectors for collinear type II phase matching of 5% MgO doped LiNbO3
        The input are:
        lambda_s: signal wavelength in meters 
        lambda_i: idler wavelength in meters 
        lambda_p: pump wavelength in meters 

        The calculated wave vectors are given in units of 1/m to match the units used in the propagation script.
        """

        
        #speed of light in m/ps
        c = 299792458*1e-12

        #transform from meters to microns
        wlArray = np.array([lambda_p, lambda_s, lambda_i])*1e6
        minimumWl = np.min(wlArray)
        maximumWl = np.max(wlArray)

        wl = np.linspace(maximumWl, minimumWl, 5000)
        om = 2*np.pi*c/wl*1e6

        self.kp = self.k_type2(om, False)*1e6
        self.ks = self.k_type2(om, True)*1e6
        self.ki = self.k_type2(om, False)*1e6
        self.om = om


        self.indistinguishableBool = False
        self.QPMbool = False
    
    def sellmeier(self, a, b1, b2, c1, c2, lam):
        return np.sqrt(a + b1/(lam**2 - b2) + c1/(lam**2 - c2))


    def Omega_func(self, nx,ny,nz):
        return np.arcsin(nz/ny*np.sqrt((ny**2 - nx**2)/(nz**2 - nx**2)))


    def theta_mm_func(self, theta,phi):
        return np.arctan(np.cos(phi)*np.tan(theta))

    def theta1_func(lf, theta, phi,theta_mm,Omega):
        if theta != np.pi/2:
            return np.arccos(
            (np.cos(theta)/np.cos(theta_mm))*np.cos(Omega - theta_mm)
            )
        else:
            return np.arccos(np.sin(Omega)*np.cos(phi))


    def theta2_func(self, theta,phi,theta_mm,Omega):
        if theta != np.pi/2:
            return np.arccos(
            (np.cos(theta)/np.cos(theta_mm))*np.cos(Omega + theta_mm)
            )
        else:
            return np.arccos(-np.sin(Omega)*np.cos(phi))

    def effective_n(self, nx,nz,theta):
        return nx*nz/(np.sqrt(
            nz**2*np.cos(theta/2)**2 + nx**2*np.sin(theta/2)**2
        ))


    def k_type2(self, omega, pm):
        #c in microns/ps
        c = 299.792458
        lam = 2*np.pi*c/(omega)

        ax = 3.29100
        b1x = 0.04140
        b2x = 0.03978
        c1x = 9.35522
        c2x = 31.45571

        ay = 3.45018
        b1y = 0.04341
        b2y = 0.04597
        c1y = 16.98825
        c2y = 39.43799

        az = 4.59423
        b1z = 0.06206
        b2z = 0.04763
        c1z = 110.80672
        c2z = 86.12171

        nx = self.sellmeier(ax, b1x, b2x, c1x, c2x, lam)
        ny = self.sellmeier(ay, b1y, b2y, c1y, c2y, lam)
        nz = self.sellmeier(az, b1z, b2z, c1z, c2z, lam)

        Omega = self.Omega_func(nx,ny,nz)
        theta = 90*np.pi/180
        phi = 23.58*np.pi/180
        theta_mm = self.theta_mm_func(theta,phi)

        theta1 = self.theta1_func(theta,phi,theta_mm,Omega)
        theta2 = self.theta2_func(theta,phi,theta_mm,Omega)
        if pm:
            TH = theta1 + theta2
        else:
            TH = theta1 - theta2

        return self.effective_n(nx,nz,TH)*2*np.pi/lam


        