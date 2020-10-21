import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.io.fits as pft
from astropy.modeling.models import Voigt1D, Polynomial1D, Gaussian1D
from astropy.modeling import fitting
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec



def interp_fwhm_L(x_0):
    return np.interp(x_0, psf_x0, psf_fwhm_L)

def interp_fwhm_G(x_0):
    return np.interp(x_0, psf_x0, psf_fwhm_G)

def pix2wave_default(new, pix, waves):
    return np.interp(new, pix, waves)

#these work reasonable well if you use pix_fit, count_fit, uncert_fit (i.e., cut out regions around line)
#calib_lines = np.array([435.833, 546.074, 696.543, 706.722, 738.393, 
#                        750.387, 763.511, 772.376, 811.531, 826.452, 842.456, 852.144,
#                        912.297, 922.450])


# Calib lines from 
# https://www.oceaninsight.com/globalassets/images/manuals--instruction-old-logo/wavelength-calibration-products-v1.0_updated.pdf
calib_lines = np.array([435.833, 546.074, 696.543, 706.722, 738.393, 
                        750.387, 763.511, 772.376, 811.531, 826.452, 842.456, 852.144,
                        912.297, 922.450, 965.779, 576.96, 579.07, 794.818, 800.616, 866.794, 727.294])

pix, wave, counts, uncerts = np.loadtxt('test_data.txt', unpack=True)
pix = pix.astype(np.int)


fitter = fitting.LevMarLSQFitter()
offset = Polynomial1D(degree=0)
model_list = []
lower = 5
upper = 6
for line in calib_lines:
    pix_guess = np.floor(np.interp(line, wave, pix)).astype(np.int)
    g = Gaussian1D(mean=pix_guess, amplitude=counts[pix_guess-lower:pix_guess+upper].max())

    model_list.append(g)

model = offset
for m in model_list:
    model += m

# there are no calibration lines out here, but there are some emission lines with unknown wavelengths.
# mask to avoid them.
mask = np.logical_or(wave>1000, np.logical_and(
                    wave>402,
                    wave<408))
pix_masked = np.ma.masked_where(mask, pix)
counts_masked = np.ma.masked_where(mask, counts)
uncerts_masked = np.ma.masked_where(mask, uncerts)

# fit the model on masked data, not full set
bf_model = fitter(model, pix_masked, counts_masked, weights=1./uncerts_masked)

x_meas = []
for i,m in enumerate(bf_model):
    if i == 0:
        continue
    x_meas.append(m.mean.value)
x_meas = np.asarray(x_meas)

# 3rd order seems to be the optimal solution
#stds = []
#orders = [1,2,3,4,5]
#for o in orders:
#    fit = np.polynomial.chebyshev.chebfit(x_meas, calib_lines, deg=o)
#    p = np.polynomial.chebyshev.Chebyshev(fit)
#    stds.append(np.std(calib_lines-p(x_meas)))
#plt.plot(orders, stds, '-k.')

fit = np.polynomial.chebyshev.chebfit(x_meas, calib_lines, deg=3)
p = np.polynomial.chebyshev.Chebyshev(fit)



# Plotting:
# =========
fig = plt.figure(figsize=(24,12))
ax0 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
ax1 = plt.subplot2grid((3,2), (2,0), colspan=2, rowspan=1, sharex=ax0)
for ax in [ax0, ax1]:
    ax.tick_params(axis='both', which='major', labelsize=16)

xg = np.linspace(pix.min(), pix.max(), 10000)
ax0.plot(p(pix_masked), counts_masked, 'k.')
ax0.plot(p(xg), bf_model(xg), '-r')
ax0.set_ylabel('Counts', size=28)
ax0.set_xlim(350, 1100)
ax0.set_ylim(-100, 30000)

for l in calib_lines:
    ax0.axvline(l, ls='--', color='b', alpha=0.25)

resids = calib_lines - p(x_meas)
ax1.plot(calib_lines, resids, 'k.')
for i in [1,-1]:
    ax1.axhline(resids.mean()+i*resids.std(), color='r', ls='--', alpha=1)
ax1.axhline(resids.mean(), ls='--', color='k', alpha=1)
ax1.set_ylim(-0.75, 0.75)
ax1.set_xlabel('Wavelength [nm]',size=28)
ax1.set_ylabel(r'$\lambda_\mathrm{true} - \lambda_\mathrm{fit}$ [nm]', size=28)

plt.tight_layout()
if False:
    plt.savefig('./spectro_calib_plot.png')
plt.show(block=False)

if False:
    np.savetxt('./pix2wave_chebfit.txt', fit)

plt.figure()
#plt.plot(pix, wave-p(pix), '-k.')
##plt.plot(pix, p(pix), '-r,')
#plt.plot(x_meas, calib_lines-p(x_meas), 'bs')
plt.plot(wave, wave-p(pix), '-k.')
#plt.plot(pix, p(pix), '-r,')
#plt.plot(, calib_lines-p(x_meas), 'bs')

#plt.xlabel('Pixels')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Delta Wavelength [nm]')
plt.show()
