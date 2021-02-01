import numpy as np
from astropy.io import fits
import SpectrumAnalysis as SA
import SimulatedCCDData as SCD
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    one_d_spectrum_file_to_load = 'Gemini_1d_spectrum_from_california_KR1.npy' 
    one_d_spectrum_file_to_save = 'Gemini_1d_spectrum_from_california_KR1.npy'
    one_d_spectrum_file_to_load = None
    one_d_spectrum_file_to_save = None
    simulated_ccd = SCD.SimulatedCCD()

    wavelengths, wavelength_to_flux_funct = SCD.read1DSpectrumFromFile()
    if one_d_spectrum_file_to_load is None: 
    
        calib_spectrum_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calData/'
        calib_spectrum_file = 'SCI_22-08-18_12.fits'

        calib_1d_spectrum = SA.getOneDSpectrumFromFile(calib_spectrum_file, calib_spectrum_dir, summing_method = 'sum')

        start_spectrum = 'KR1'
        calib_fluxes, calib_centers, calib_seeings, calib_backgrounds, wavelength_to_pix_funct, wavelength_to_seeing_funct = SA.measureWavelengthSolution(calib_1d_spectrum, start_spectrum = start_spectrum)

    else:
        wavelength_to_pix_funct = None
        wavelength_to_seeing_funct = None 

    simulated_flux_pattern = simulated_ccd.getFluxPattern(wavelengths = wavelengths, wavelength_to_flux_funct = wavelength_to_flux_funct,
                                                          wavelength_to_pix_funct = wavelength_to_pix_funct, wavelength_to_seeing_funct = wavelength_to_seeing_funct,
                                                          load_one_d_flux_file = one_d_spectrum_file_to_load, save_one_d_flux_file = one_d_spectrum_file_to_save)

    exp_time = 300.0
    use_master_bias = 1
    use_master_dark = 1
    binning = [2,2]
    image = simulated_ccd.getImage(simulated_flux_pattern, exp_time, use_master_bias = use_master_bias, use_master_dark = use_master_dark, binning = binning)

    image_save_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calData/'
    image_save_file = 'artificial_gemini_sky' + str(exp_time) + 's_' + str(binning[0]) + 'x' + str(binning[1]) + '.fits'
    SCD.saveDataToFitsFile(image, image_save_file, image_save_dir)
    plt.imshow(image)

    

    

    
