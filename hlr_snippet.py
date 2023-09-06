import numpy
import scipy.integrate

# raw input data, filename indicates contents of each column
# flow-corrected redshifts in column 1, angular sizes in column 5
raw_data = numpy.genfromtxt('id_z_mstar_smdust_ssfr_hlr_bt.txt', delimiter = ',')

c    = 299792458 # speed of light, m s^-1
matt = 0.3 # matter density fraction
lamb = 0.7 # cosmological constant
H0   = 70.0 # Hubble constant, m s^-1 kpc^-1
Dh   = c / H0 # Hubble distance, kpc

# inverse of equation 14 in Hogg (2000), denominator of integral in equation 15 in Hogg (2000)
# specific to 737 cosmology
Einv = lambda z: ((matt * (1 + z)**3) + lamb)**-0.5

Dc = [0] * N.shape(raw_data)[0]

# performing integral for each flow-corrected redshift to get line-of-sight comoving distances
# equation 15 in Hogg (2000)
for i in range(numpy.shape(raw_data)[0]):
	Eint = scipy.integrate.quad(Einv, 0, raw_data[i,1])
	Dc[i] = Dh * Eint[0] # comoving distances, kpc

Dm = Dc # transverse comoving distances, equation 16 in Hogg (2000)
Dl = (1 + raw_data[:,1]) * Dm # luminosity distances, equation 21 in Hogg (2000)
Da = Dm / (1 + raw_data[:,1]) # angular diameter distances, equation 18 in Hogg (2000)

# converting angular sizes to kpc sizes, using angular diameter distances
# then log transforming kpc sizes
raw_data[:,5] = (raw_data[:,5] / 206265) * Da
raw_data[:,5] = numpy.log10(raw_data[:,5])

