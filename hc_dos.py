#! /usr/bin/python

#import numpy as np
#import re

#from scipy.optimize import leastsq
#from scipy import constants
#import matplotlib.pyplot as plt
import numpy as np
import collections
import re

class hc_dos:
    """to to: if input dimensions do not match, e.g.  T is Nx1 and B is Mx1, understand it as mx1 per element of T"""
    def __init__(self, histfile = None, expfile = None, datasource = 'PPMS'):
        """
        Utility class for calculating the heat capacity of a gapped triplonic (Schottky-term) system based on 
            - a given density of states
            - a multilevel-system with given energies
        
        and of a phononic system based on the Einstein and Debye-approximation
        Initializes the DOS-histogram by pickle-loading it from path. Later, we will try to include simple
        string files, if needed
        
        The DoS has to be given as a function of energy in meV

        The heat capacity is calculated in J per mole (Joule per 6.022e23 particles/atoms/unit cell)
        Input:

        path_to_dos_hist: String, leading to the histogram file
        
        
        """
        from scipy import constants 

        self.fac_meVtoK =  constants.e * 1e-3 / constants.k #factor to get energies in K from energies in meV
        self.muB = constants.physical_constants['Bohr magneton in K/T'][0] #muB divided by kB in K/T
        self.kmol = constants.k * constants.N_A
        if not histfile == None:
            self.load_dos(histfile)
        if not expfile == None:
            self.load_data(expfile)

    def load_dos(self, path):
        """loading the dos histogram, expects plain text
        'energies'
        energies
        'count'
        count
        """
        #load the histogram
        with open(path) as datei:
            dos_hist = []
            for line in datei:
                if 'energies' in line or 'count' in line:
                    dos_hist.append(np.array([]))
                else:
                    dos_hist[-1] = np.append(dos_hist[-1], float(line))

        #find the lower and upper bound of the unperturbed dos
        self.Emin_raw, self.Emax_raw = dos_hist[0].min() * self.fac_meVtoK, dos_hist[0].max() * self.fac_meVtoK    
        #get the energy spacing
        self.deltaE  = (dos_hist[0][1:] - dos_hist[0][:-1]).mean() * self.fac_meVtoK 
        #get the distribution
        self.dos_pdf =  np.asarray(dos_hist[1], dtype = float)
        #and normalize it
        self.dos_pdf /= self.dos_pdf.sum()
        self.Emin = self.Emin_raw
        self.Emax = self.Emax_raw

    def load_data(self, path, datasource = 'PPMS', Nmol = 1, mass =1, method = 'replace'):
        """ takes the files in path and loads the data from there
        INPUT:
            path: string or iterable list of strings, not nested
        """
#        pdb.set_trace()
        if isinstance(path, basestring):
            path = np.array([path])
        for p in path:
            try:
                data = np.append(data, np.atleast_2d(self._load_data_single(p)), axis = 0)
            except:
                data = np.atleast_2d(self._load_data_single(p))
        data[:,2] /= Nmol #convert from Joule per K  to Joule per mole and K

        if method == 'replace':
            self.T_exp = data[:,1]
            self.B_exp = data[:,0]
            self.C_exp = data[:,2] 

        elif method == 'add':
            self.T_exp = np.append(self.T_exp, data[:,1])
            self.B_exp = np.append(self.B_exp, data[:,0])
            self.C_exp = np.append(self.C_exp, data[:,2])

    def _load_data_single(self, path, datasource = 'PPMS'):
        """loads the data from one file
        INPUT:
            path: string, path to file
        OUTPUT:
            data: N x 3 array with
            data[:,0] : field in T
            data[:,1] : temperature in K
            data[:,2] : heat capacity in J/K
        """
        C_factors = {'\xb5J/K' : 1e-6, #microJoule per Kelvin
                     }
        if datasource == 'PPMS':
            #we expect a sample heat capacity in microJoule per K
            with open(path, 'r') as f:
                for line in f:
                    if 'Time Stamp' in line:
                        self.unit = line.split(',')[9].split()[-1].strip('()')
                        break

            with open(path, 'r') as f:
                lines = (line for line in f if not re.search('[a-zA-DF-Z]', line))
                data = np.genfromtxt(lines, delimiter = ',', usecols = (5, 4, 9))
            data[:,0] /= 1e4 #convert from Oe to T
            data[:,2] *= C_factors[self.unit] #convert from microJoule to Joule
        return data

    def _dos_raw(self, energy, S):
        """returns the density of states for states with a total spin of S
        Input:
            energy: float or array of floats, energy at which the DoS should be given
            S       int or array of ints, total spin of the state to be examined
            
        Output:
            dos:    array of float, density of states
        """
        energy = np.atleast_1d(energy)
        S = np.atleast_1d(S)
#        else:  #if energy is a list or an array etc. (iterable) 
        energy = np.atleast_1d(energy)  #make sure the energy is an array
        dos = np.zeros(energy.shape[0])
        #the positions in the dos-histogram are the differences of energy and minimum energy in units of deltaE
        maske = np.where((S==0) & (energy < self.deltaE/2) & (energy >- self.deltaE/2)) #S=0 and energies close to zero
        dos[maske] = 1.0   #where S=0, the dos is 1.0 close to zero (N states for N particles) (also outside the band)

#        pdb.set_trace()
        maske = np.where((S==1) & (energy > self.Emin) & (energy < self.Emax)) #all energies with S=1 in the band
        indices = np.asarray((energy - self.Emin) / self.deltaE, dtype=int) #
        dos[maske] = self.dos_pdf[indices[maske]] #the dos is given by the values of the histogram at the "indices"-positions

        return dos

    def dos(self, energy, spinstate, field, g=1.94):
        """returns the density of states for the given energy, field, and involved spinstates
        Input:
            energy: n x 1 float array of energies, unit: K (energy divided by kB)
            spinstate: n x 4 x 2 float array of involved spin states S,m
            field: n x 1 float array of fields, unit: T

        Output:
            dos: n x 1 array, dtype=float, unit: 1/K
        """

        n = energy.shape[0]
        energy = np.atleast_1d(energy)
        energy = energy.reshape(n, 1)   #build a (n, 1)-array as it may be a (n,) array 
        energy = np.tile(energy, spinstate.shape[1])

        field = np.atleast_1d(field)
        field = field.reshape(field.shape[0], 1)
        field = np.tile(field, spinstate.shape[1])
        
        energy = energy + spinstate[:, :, 1] * g * self.muB * field #transform the energy back to the fieldless case for each m

        S = spinstate[:,:,0].reshape(energy.shape[0] * energy.shape[1]) #1-dim array of the first spin quantum number
        energy = energy.reshape(energy.shape[0] * energy.shape[1]) #convert it to a 1-dim array
        
        dos = self._dos_raw(energy, S)
        dos = dos.reshape(n, dos.shape[0]/n)
        dos = dos.sum(axis=1)
        return dos
                
    def C_el_dos(self, B, T, g = 1.94):
        """
        calculates the specific heat of the system for the given fields and temperatures
        Input:
            B: n x 1 array, dtype = float, unit: T (Tesla)
            T: n x 1 array, dtype = float, unit: K (Kelvin)
        """
#        pdb.set_trace()
        Emin = min(0, self.Emin - g * self.muB * max(B))    #get the global minimum of the energy for all fields
        Emax = self.Emax + g * self.muB * max(B)            #get the global maximum of the energy for all fields
        
        E = np.arange(Emin, Emax, self.deltaE)
        N = E.shape[0]
        e = np.tile(E, B.shape[0])

        n = B.shape[0]
        b = np.repeat(B, N)

        d = np.array([[0, 0], [1, 1], [1, 0], [1, -1]])
        d = d.reshape(1, d.shape[0], d.shape[1])
        d = np.tile(d, (b.shape[0], 1, 1))

#        return E.shape, B.shape, d.shape
        dos = self.dos(e, d, b)      
        dos = dos.reshape(n,N).T    

        E = E.reshape(E.shape[0], 1)

        bes_fak = - np.outer(E, 1/T)
        bes_fak = np.exp(bes_fak)

        A = dos * E**2 * bes_fak
        A = A.sum(axis = 0)

        B = dos * E    * bes_fak
        B = B.sum(axis = 0)

        Z = dos        * bes_fak
        Z = Z.sum(axis = 0)
        
        C = (A*Z - B**2) / Z**2

        return self.kmol * C/T**2
        
    def c_level(self, e, m, T, B = np.array([0])):
        """
        calculates the heat capacity of a multi-level system with a Zeeman-splitting
        
        input:
        e: M x 1 - array of energies for the level, given in units of K (E/kB), 
        d: M x 1 - array of degeneracies of the levels
        m: M x 1 - array of magnetic moments of the levels (-1, 1, 1)
    
        T: N x 1 - array  of sample temperatures in K
        B: N x 1 - array  of applied magnetic fields in Tesla
        """
        g = 1.94 #lande g-factor
#        muB = constants.physical_constants['Bohr magneton in K/T'][0] #muB devided by kB
        e = np.asarray(e, dtype = float) #making sure that the energies are a floating point array
        T = np.asarray(T, dtype = float) #making sure that the temperatures are a floating point array
        B = np.asarray(B, dtype = float) #making sure that the fields are a floating point array
        B = g * self.muB * B    
        energies  = e[:, np.newaxis] * np.ones((e.shape[0], B.shape[0])) #initialize the energies for each measurement point
        energies -= np.outer(m, B) 
        boltz_fac = np.exp( -energies / T )
    
        A = energies * energies * boltz_fac
        A = A.sum(axis=0)
    
        B = energies * boltz_fac
        B = B.sum(axis=0)
    
        Z = boltz_fac
        Z = Z.sum(axis=0)
    
        return self.kmol * (A * Z - B * B) / (Z * Z) / T**2

    def c_schottky(self, T, deltaE):
        """convenience function to be able to simulate a 2-level system without to much input"""
        e = np.array([0, deltaE])
        m = np.array([0, 0])
        return self.c_level(e, m, T)

    def C_ph_Debye(self, T, T_D):
        T = np.asarray(T, dtype=float)
        if isinstance(T, (collections.Sequence, np.ndarray)):
            stepsize = 1e-3
            c = np.zeros(T.shape[0])
            for index, t in enumerate(T):
                x = np.arange(1e-3, T_D / t, stepsize)
                y = x**4 * np.exp(x) / (np.exp(x) - 1)**2
                c[index] = 9 * sum(y) * stepsize * ( t / T_D )**3
        else:
            x = np.arange(1e-3, T_D / T, 1e-3)
            y = x**4 * np.exp(x) / (np.exp(x) - 1)**2
            c = 9 * sum(y) * (x[1:]-x[:-1]).mean() * ( T / T_D )**3
        return c * self.kmol
    
    def C_ph_Einstein(self, T, T_E):
        T = np.asarray(T, dtype=float)
        return 3 * self.kmol * (T_E / T)**2 * np.exp(T_E/T) / (np.exp(T_E/T) - 1)**2

if __name__ == "__main__":
    pass
