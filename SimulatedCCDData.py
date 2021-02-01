#Gain given in CCD per ADU
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import erf 
import numpy as np
import matplotlib.pyplot as plt
from cantrips import binArray
from astropy.io import fits
from scipy import integrate
import math
import time
from cantrips import makeInterpFromDict
from cantrips import saveDataToFitsFile
from cantrips import readInDataFromFitsFile
from AstronomicalParameterArchive import AstronomicalParameterArchive


def measureSpectrum(spectrum_box, image, display_wavelength = 1, pix_to_wavelength_funct = None, show = 0, binning = [1,1], summing_method = 'median'):
    print ('np.shape(image) = ' + str(np.shape(image)) )
    print ('len(image[0,:]) = ' + str(len(image[0,:])) )
    print ('len(image[:,0]) = ' + str(len(image[:,0])) )
    print ('spectrum_box = ' + str(spectrum_box) )
    print ('image[0,0] = ' + str(image[0,0]) )
    print ('[(spectrum_box[1][0] / binning[1]),(spectrum_box[1][1] / binning[1])] = ' + str([(spectrum_box[1][0] / binning[1]),(spectrum_box[1][1] / binning[1])]) )
    print ('[(spectrum_box[0][0] / binning[0]),(spectrum_box[0][1] / binning[0])] = ' + str([(spectrum_box[0][0] / binning[0]),(spectrum_box[0][1] / binning[0])]) )
        

    if summing_method in ['median']: 
        binned_spectrum = [np.median(image[(spectrum_box[1][0] // binning[1]):(spectrum_box[1][1] // binning[1]), i], axis = 0)
                               for i in range(spectrum_box[0][0] // binning[0], spectrum_box[0][1] // binning[0])]
        print ('binned_spectrum = ' + str(binned_spectrum) )
    else:
        binned_spectrum = [np.sum(image[(spectrum_box[1][0] // binning[1]):(spectrum_box[1][1] // binning[1]), i], axis = 0)
                           for i in range(spectrum_box[0][0] // binning[0], spectrum_box[0][1] // binning[0])]
        
    x_bins = range(spectrum_box[0][0] // binning[0], spectrum_box[0][1] // binning[0])
        
    if not(pix_to_wavelength_funct) or (pix_to_wavelength_funct is None):
        print ('Do not have a wavelength solution available.  Mapping by pixels. ' )
        if show: 
            plt.plot(x_bins, binned_spectrum)
            plt.show()
        return [x_bins, binned_spectrum]
    else:
        wavelength_bins = [pix_to_wavelength_funct(x_bin) for x_bin in x_bins]
        if show: 
            plt.plot(wavelength_bins, binned_spectrum)
            plt.show()
        return [wavelength_bins, binned_spectrum]


def read1DSpectrumFromFile(spectrum_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/', spectrum_file = 'GeminiSkyBrightness.txt', n_ignore = 14):
    with open(spectrum_dir + spectrum_file, 'r') as ins:
        spectrum_dir = {}
        n_lines = 0
        for line in ins:
            n_lines = n_lines + 1 
            if n_lines > n_ignore: 
                new_vals = line.split()
                spectrum_dir[float(new_vals[0])] = float(new_vals[1])
    spectrum_keys = np.sort(list(spectrum_dir.keys()) ) 
    spectrum_interp = interp1d(spectrum_keys, [spectrum_dir[key] for key in spectrum_keys])
    return [spectrum_keys, spectrum_interp] 

def saveDataToFitsFile(image_array, file_name, save_dir, header = 'default', overwrite = True):

    print ('Saving image to file ' + save_dir + file_name ) 
    if header is 'default':
        default_file = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calData/' + 'default.fits'
        hdul  = fits.open(default_file)
        header = hdul[0].header 
    
    master_med_hdu = fits.PrimaryHDU(image_array.transpose(), header = header)
    master_med_hdul = fits.HDUList([master_med_hdu])
    master_med_hdul.writeto(save_dir + file_name, overwrite = overwrite)
    return 1

class SimulatedCCD:

    def getPixelToWavelengthFunction(self, wavelengths = 'default', x_center_pixel = 'default'):
        if wavelengths in ['default','Default','DEFAULT']:
            wavelengths = self.wavelengths
        wavelengths = sorted(wavelengths) 
        xs = [self.getColumnOfWavelength(wavelength, x_center_pixel = x_center_pixel) for wavelength in wavelengths]
        wavelength_of_x_slope = (wavelengths[-1] - wavelengths[0]) / (xs[-1] - xs[0])
        print ('wavelength_of_x_slope = ' + str(wavelength_of_x_slope)) 
        if min(xs) > 0:
            min_x = min(xs) 
            wave_of_min_x = wavelengths[np.argmin(xs)] 
            xs = [0] + xs
            wavelengths = [wave_of_min_x - wavelength_of_x_slope * (min_x - 0)] + wavelengths
        if max(xs) < self.size[0]:
            max_x = max(xs) 
            wave_of_max_x = wavelengths[np.argmax(xs)] 
            xs = xs + [self.size[0]]
            wavelengths =  wavelengths + [wave_of_max_x + wavelength_of_x_slope * (self.size[0] - max_x)]


        #print ('xs = ' + str(xs)) 
        #print ('wavelengths = ' + str(wavelengths)) 
        return interp1d(xs, wavelengths) 

    def getColumnOfWavelength(self, wavelength,  x_center_pixel = 'default'):
        if x_center_pixel in ['default','Default','DEFAULT']:
            x_center_pixel = self.x_center_pixel 
        det_x_positions, det_y_positions =  self.getFocusedImageOfSlitBounds(wavelength)
        average_position = (det_x_positions[1] + det_x_positions[0])/2.0 / self.pixel_width
        return average_position + x_center_pixel  

    def getFocusedImageOfSlitBounds(self, wavelength):
        slit_width_angle = np.arctan(self.slit_width / 2.0 * 1.0 / self.focal_length1) * 2.0 # opening angle of slit
        diff_output_angles = [np.arcsin(np.sin(-slit_width_angle / 2.0) - self.diff_order * wavelength / self.grating_sep),
                              np.arcsin(np.sin(slit_width_angle / 2.0) - self.diff_order * wavelength / self.grating_sep)
                              ] # opening angle after diffraction element of single wavelength
        det_x_positions = [self.focal_length2 * np.tan(diff_angle - self.angle_out_lens_vs_diffraction_elem) for diff_angle in diff_output_angles]# detection positions of single wavelength
        det_y_positions = [- self.slit_height / 2.0 * self.focal_length2 /self.focal_length1, self.slit_height / 2.0 * self.focal_length2 /self.focal_length1]

        return det_x_positions, det_y_positions

    #Determine how much the lack of perfect focus (the seeing) effects how much the photons are redistributed.  
    def getDefocusedScalingOfSlit(self, detector_x, wavelength, seeing, focus_rect_bounds):
        #focus_rect_bounds = getFocusedImageOfSlitBounds(wavelength)
        #print ('(focus_rect_bounds[1] - detector_x)/np.sqrt(2.0 * seeing ** 2.0) = ' + str((focus_rect_bounds[1] - detector_x)/np.sqrt(2.0 * seeing ** 2.0))) 
        intensity_scaling = 0.5 * ( erf((focus_rect_bounds[1] - detector_x)/np.sqrt(2.0 * seeing ** 2.0)) -  erf((focus_rect_bounds[0] - detector_x)/np.sqrt(2.0 * seeing ** 2.0)) )

        return intensity_scaling
    
    def defocusedSlitPhotons(self, detector_x, wavelength, photons_on_detector, seeing):
        #photon_on_detector is number of photons/s/nm hitting detector at wavelength 
        det_x_positions, det_y_positions = self.getFocusedImageOfSlitBounds(wavelength)
        defocused_scaling = self.getDefocusedScalingOfSlit(detector_x, wavelength, seeing, det_x_positions)

        #A_slit = self.slit_width * self.slit_height 
        A_det = (abs(det_y_positions[1] - det_y_positions[0])) * (abs(det_x_positions[1] - det_x_positions[0]))
        area_scaling = 1.0 / A_det

        #returns the photons flux at position x: number of photons/s/nm/nm^2 hitting detector at position x 
        return area_scaling * defocused_scaling * photons_on_detector 

    def energyPerPhoton(self, wavelength):
        astro_arch = AstronomicalParameterArchive()
        h = astro_arch.getPlancksConstant() * 10.0 ** 6.0 #convert from joules to ergs
        c = astro_arch.getc() * 10.0 ** 12.0 #convert from meters to nanometers
        return h * c / wavelength

    def getPhotonRateVsXPixel(self, pix_x_range = 'all', photon_rate_funct = 'default', x_center_pixel = 'default', wavelengths = 'default', seeing_funct = 'default'):
        if pix_x_range in ['all','ALL', 'All','full','Full','FULL']:
            pix_x_range = self.spectrum_box[0]
        if photon_rate_funct in ['default','Default','DEFAULT']:
            photon_rate_funct = self.photon_rate_funct #Function of wavelength that gives number of photons/s/nm hitting detector at input wavelength 
        if x_center_pixel in ['default','Default','DEFAULT']:
            x_center_pixel = self.x_center_pixel
        if wavelengths in ['default','Default','DEFAULT']:
            wavelengths = self.wavelengths 
        pixel_width = self.pixel_width
        pixel_height = self.pixel_height
        pixel_area = pixel_width * pixel_height 

        if seeing_funct in ['default','Default','DEFAULT']:
            #diffraction limit: sin(theta) = m * lambda / slit_width
            seeing_funct = lambda wavelength: self.pixel_width 

        pixel_to_physical_x = lambda pixel_x: (pixel_x - x_center_pixel) * self.pixel_width
        test_x = [pix_x for pix_x in range(*pix_x_range)][0]
        test_defocused_vals = [self.defocusedSlitPhotons(pixel_to_physical_x(test_x), wavelength, float(photon_rate_funct(wavelength)), seeing_funct(wavelength)) for wavelength in wavelengths]
        #integrand = lambda wavelength, pixel : scaling(wavelength) * np.exp(power(wavelength, pixel))
        integrand = lambda wavelength, pix_x: self.defocusedSlitPhotons(pixel_to_physical_x(pix_x), wavelength, photon_rate_funct(wavelength), seeing_funct(wavelength)) * self.eff_qe(wavelength) #Now a electron flux: number of electrons/s/nm/nm^2 hitting detector at position x
        delta_wavelengths = [(wavelengths[i+1] + wavelengths[i])/2 - (wavelengths[i] + wavelengths[i-1])/2 for i in range(1,len(wavelengths) - 1)]
        delta_wavelengths = [delta_wavelengths[0] ] + delta_wavelengths + [delta_wavelengths[-1]]
        print ('Beginning numerical integrations on wavelengths. ')
        det_e_rate_vect = [0 for pix_x in range(*pix_x_range)]
        for j in range(len(det_e_rate_vect)):
            pix_x = list(range(*pix_x_range))[j]
            print ('Working on pix_x ' + str(pix_x)) 
            det_e_rate_vect[j] = sum([integrand(wavelengths[i], pix_x) * delta_wavelengths[i] for i in range(len(wavelengths))]) * pixel_area
        #det_e_rate_vect = [sum([integrand(wavelengths[i], pix_x) * delta_wavelengths[i] for i in range(len(wavelengths))]) * pixel_area for pix_x in range(*pix_x_range)] #now number of electrons detected per pixel
        print ('Finished wavelengths') 
               
        #det_intensity_vect = [integrate.quad(lambda wavelength: self.defocusedSlitPhotons(pixel_to_physical_x(pix_x), wavelength / 1000.0, wavelength_funct(wavelength), seeing_funct(wavelength)), *wavelength_bounds)[0] for pix_x in range(*pix_x_range)]

        return det_e_rate_vect #returns electrons/s arriving at each pixel in the pixel range

    def getPhotonMesh(self, one_d_rate_vect = None, pix_x_range = 'all', photon_rate_funct = 'default', x_center_pixel = 'default', wavelengths = 'default', seeing_funct = 'default', y_center_pixel = 'default'):
        if y_center_pixel in ['default','Default','DEFAULT']:
            y_center_pixel = self.y_center_pixel
        if one_d_rate_vect is None: 
            one_d_rate_vect = self.getPhotonRateVsXPixel(pix_x_range = pix_x_range, photon_rate_funct = photon_rate_funct, x_center_pixel = x_center_pixel, wavelengths = wavelengths, seeing_funct = seeing_funct)
        y_pix_range = self.getFocusedImageOfSlitBounds(self.wavelengths[0])[1]
        y_pix_range = [y / self.pixel_height + y_center_pixel for y in y_pix_range]
        print ('y_pix_range = ' + str(y_pix_range)) 

        xmesh, ymesh = np.meshgrid(range(self.size[0]), range(self.size[1]))
        rate_mesh = np.zeros(np.shape(xmesh))
        
        for i in range(self.size[1]):
            if i > y_pix_range[0] and i < y_pix_range[1]: rate_mesh[i,:] = one_d_rate_vect

        return rate_mesh

    def getADUMesh(self, gain = 'default', one_d_rate_vect = None, pix_x_range = 'all', photon_rate_funct = 'default', x_center_pixel = 'default', wavelengths = 'default', seeing_funct = 'default'):
        electron_mesh = self.getPhotonMesh(one_d_rate_vect = one_d_rate_vect, pix_x_range = pix_x_range, photon_rate_funct = photon_rate_funct, x_center_pixel = x_center_pixel, wavelengths = wavelengths, seeing_funct = seeing_funct)
        if gain in ['default','Default','DEFAULT']:
            gain = self.gain 
        ADU_mesh = 1.0 / gain * electron_mesh
        return ADU_mesh 
        

    def getOneDSpecrum(self, wavelength_funct = None):
        if wavelength_funct is None:
            wavelength_funct = read1DSpectrumFromFile()
            return 1 

    def getFluxPattern(self, wavelengths = None, wavelength_to_flux_funct = None, wavelength_to_pix_funct = None, wavelength_to_seeing_funct = None, 
                       n_sig = 5.0, load_one_d_flux_file = None, save_one_d_flux_file = None, scale_eff_qe = 1):

        size = self.size 
        wavelength_range = (min(wavelengths), max(wavelengths)) 
        if load_one_d_flux_file is None:
            if wavelength_to_pix_funct is None: wavelength_to_pix_funct = self.wavelength_to_pix_funct
            if (wavelengths is None) or (wavelength_to_flux_funct is None) or (wavelength_to_pix_funct is None) or (wavelength_to_seeing_funct is None):
                print ('At least one of wavelengths,  wavelength_to_flux_funct, wavelength_to_pix_funct, or wavelength_to_seeing_funct is None and no predetermined 1dFlux given.  Returning 0. ')
                return 0
            scaling =  lambda wavelength: wavelength_to_flux_funct(wavelength) / np.sqrt(2.0 * math.pi * wavelength_to_seeing_funct(wavelength) ** 2.0)
            power = lambda wavelength, pixel: - (wavelength_to_pix_funct(wavelength) - pixel) ** 2.0 / (2.0 *  wavelength_to_seeing_funct(wavelength) ** 2.0) 
            integrand = lambda wavelength, pixel : scaling(wavelength) * np.exp(power(wavelength, pixel))
            delta_wavelengths = [(wavelengths[i+1] + wavelengths[i])/2 - (wavelengths[i] + wavelengths[i-1])/2 for i in range(1,len(wavelengths) - 1)]
            delta_wavelengths = [delta_wavelengths[0] ] + delta_wavelengths + [delta_wavelengths[-1]]
            one_d_flux_funct = lambda x: sum(integrand(wavelengths[i], x) * delta_wavelengths[i] for i in range(len(wavelengths)))
            #one_d_flux_funct = lambda x: integrate.quad(lambda wavelength: integrand(wavelength, x),
            #                                            max(wavelength_range[0], minimize_scalar(lambda wavelength: abs((wavelength_to_pix_funct(wavelength) - x) / wavelength_to_seeing_funct(wavelength) + n_sig), bounds = wavelength_range, method = 'bounded')['x']),
            #                                            min(wavelength_range[-1], minimize_scalar(lambda wavelength: abs((wavelength_to_pix_funct(wavelength) - x) / wavelength_to_seeing_funct(wavelength) - n_sig), bounds = wavelength_range, method = 'bounded')['x'])
            #                                           ) [0]
            start = time.time()
            one_d_flux_vect = np.zeros(np.shape([0.0 for elem in range(size[0])])) + 0.0
            for pix in range(size[0]):
                if scale_eff_qe:
                    eff_qe_at_pix = self.eff_qe(minimize_scalar(lambda wavelength: abs(wavelength_to_pix_funct(wavelength) - pix), bounds = wavelength_range, method = 'bounded' )['x'])
                    if pix % 100 == 0: print ('Correcting for QE with effective QE = ' + str(eff_qe_at_pix) )
                    new_val = one_d_flux_funct(pix) * eff_qe_at_pix
                else:
                    new_val = one_d_flux_funct(pix)
                one_d_flux_vect[pix] = new_val 
                if pix % 100 == 0:
                    end = time.time()
                    print ('Took ' + str(end - start) + 's for element ' + str(pix) + ' for which the wavelength is ' + str(minimize_scalar(lambda wavelength: abs(wavelength_to_pix_funct(wavelength) - pix), bounds = wavelength_range, method = 'bounded' )['x']) + ' and the result is ' + str(new_val))
                    start = time.time() 
            #one_d_flux_vect = [one_d_flux_funct(pix) for pix in range(size[0])]
 
        else:
            print ('Loading pregenerated 1d flux file. ')
            one_d_flux_vect = np.load(load_one_d_flux_file)
            print ('one_d_flux_vect = ' + str(one_d_flux_vect) )
            #one_d_flux_vect = np.load(load_one_d_flux_file).item()
        
        if not(save_one_d_flux_file is None):
            print ('saving one_d_flux_vector to ' + str(save_one_d_flux_file) )
            np.save(save_one_d_flux_file, one_d_flux_vect)
        print ('one_d_flux_vect = ' + str(one_d_flux_vect) )
        plt.plot(range(size[0]), one_d_flux_vect)
        #one_d_flux = lambda xs, height, center, width: height * np.exp(-(xs - center) ** 2.0 / (2.0 * width ** 2.0))
        xmesh, ymesh = np.meshgrid(range(size[0]), range(size[1]))
        #print 'xmesh = ' + str(xmesh) 
        #print 'ymesh = ' + str(ymesh)
        flux_mesh = np.zeros(np.shape(xmesh))
        print ('len(flux_mesh[0,:]) = ' + str(len(flux_mesh[0,:])) )
        print ('len(flux_mesh[:,0]) = ' + str(len(flux_mesh[:,0])) ) 
        for i in range(size[1]):
            flux_mesh[i,:] = one_d_flux_vect
        #flux_mesh = np.sum([one_d_flux(xmesh, line[0], line[1], line[2]) for line in lines], axis = 0)
        print ('flux_mesh = ' + str(flux_mesh) )
    
        mask_funct = lambda xs, ys: (xs > self.spectrum_box[0][0]) * (xs < self.spectrum_box[0][1]) * (ys > self.spectrum_box[1][0]) * (ys < self.spectrum_box[1][1]) 
        mask = mask_funct(xmesh, ymesh)
        masked_mesh = flux_mesh * mask

        return masked_mesh.transpose()
        

    def showCCD(self, image_array):
        fig, ax = plt.subplots()
        cax =  ax.imshow(image_array.transpose(), interpolation = 'none')
        cbar = fig.colorbar(cax) 

    def getImage(self, flux_pattern, exp_time, temp = -15.0, use_master_bias = 0, use_master_dark = 0, binning = [1,1]):
        print ( 'use_master_bias = ' + str(use_master_bias) ) 
        #print 'binning = ' + str(binning)
        #print 'self.size = ' + str(self.size) 
        binned_size = [self.size[0] // binning[0], self.size[1] // binning[1]]
        if use_master_bias:
            print ( 'Using master bias file: ' + self.master_bias_file )
            bias_array, bias_header = readInDataFromFitsFile(self.master_bias_file, self.master_bias_dir)
            bias_array = bias_array.transpose() 
            bias_signal = bias_array[0:self.size[0], 0:self.size[1]]
            bias_signal = binArray(bias_signal, binning) / (binning[0] * binning[1])
            print ( 'np.shape(bias_signal) = ' + str(np.shape(bias_signal))  )
            
            #bias_signal = np.random.normal(self.bias_level, self.rdnoise, tuple(binned_size))
        else:
            bias_signal = np.zeros(self.size) + self.bias_level
        bias_noise = np.random.normal(0.0, self.rdnoise, tuple(binned_size))
        if use_master_dark:
            print ('Using master dark file: ' + self.master_dark_file )
            dark_array, dark_header = readInDataFromFitsFile(self.master_dark_file, self.master_dark_dir)
            dark_array = dark_array.transpose()
            dark_array = dark_array[0:self.size[0], 0:self.size[1]]
            print ('np.shape(dark_array) = ' + str(np.shape(dark_array)) )
            dark_sigma_deviations = np.random.normal(0.0, 1.0, tuple(self.size))
            print ('np.shape(dark_sigma_deviations ) = ' + str(np.shape(dark_sigma_deviations )) )
            dark_signal = dark_array * exp_time + np.sqrt(dark_array * exp_time) * dark_sigma_deviations
            dark_signal = dark_signal[0:self.size[0], 0:self.size[1]]
            #dark_signal = np.random.normal(dark_array[0:self.size[0], 0:self.size[1]] * exp_time, np.sqrt(dark_array[0:self.size[0], 0:self.size[1]] * exp_time), tuple(binned_size))
            #dark_signal = np.random.normal(dark_current * exp_time, np.sqrt(dark_current * exp_time), tuple(self.size))
        else:
            dark_current = self.dark_current_interp(temp)
            dark_signal = np.random.normal(dark_current * exp_time, np.sqrt(dark_current * exp_time), tuple(self.size))

        source_sigma_deviations = np.random.normal(0.0, 1.0, tuple(self.size))
        source_signal = flux_pattern * exp_time + np.sqrt(flux_pattern * exp_time) * source_sigma_deviations 
        #source_signal = np.random.normal(flux_pattern * exp_time, np.sqrt(flux_pattern * exp_time), tuple(self.size))

        print ('np.shape(bias_signal) = ' + str(np.shape(bias_signal)) )
        print ('np.shape(dark_signal) = ' + str(np.shape(dark_signal)) )
        print ('np.shape(source_signal) = ' + str(np.shape(source_signal)) )
        ccd_signal = dark_signal + source_signal
        print ('Binning CCD signal before adding read noise...'  )
        print ('binning = ' + str(binning)) 
        ccd_signal = binArray(ccd_signal, binning) 
            
        return ((ccd_signal + bias_signal + bias_noise) / self.gain).astype(int)

    def getDarkImage(self, exp_time, temp = -15.0, use_master_bias = 0, use_master_dark = 0):
        return self.getImage(np.zeros(self.size), exp_time,
                              temp = temp, use_master_bias = use_master_bias, use_master_dark = use_master_dark)

    def getBiasImage(self, use_master_bias = 0):
        return self.getImage(np.zeros(self.size), 0.0, use_master_bias = use_master_bias)

    def getScaledSkyFlux(self): 

    #Distances are given in nm 
    def __init__(self, size = [1024, 1024], gain = 2.0, overscan = [0, 0], rdnoise = 9.0, wavelength_to_pix_funct = None, spectrum_box = [[0, 1024], [396, 628]], 
                 dark_current_dict = {-21:0.02, -15:0.02, -10:0.02, -4:0.02, 0:0.02, 2:0.02, 6:0.02, 8:0.02},
                 true_qe_dict = {250:0.08, 300:0.23, 350:0.35, 400:0.42, 450:0.51, 500:0.62, 550:0.70, 600:0.79, 650:0.87,
                                 700:0.91, 750:0.94, 800:0.96, 850:0.92, 900:0.90, 950:0.60, 1000:0.32, 1050:0.11},
                 front_lens_trans_dict = {425:0.70, 500:0.79, 550:0.78, 600:0.81, 650:0.83, 700:0.8, 750:0.81, 800:0.84, 850:0.86, 900:0.83},
                 back_lens_trans_dict = {400:0.80, 590:0.52, 780:0.67, 970:0.88, 1160:0.95, 1540:0.88, 1730:0.80, 1920:0.73, 2110:0.63, 2300:0.44}, 
                 grating_eff_dict = {300:0.005, 350:0.01, 400:0.02, 450:0.2, 500:0.34, 550:0.5, 600:0.58, 650:0.63,
                                     700:0.64, 750:0.63, 800:0.61, 850:0.57, 900:0.55, 950:0.51, 1000:0.48, 1050:0.45,
                                     1100:0.41, 1150:0.37, 1200:0.34, 1250:0.31, 1300:0.28, 1350:0.26, 1400:0.24, 1450:0.23,
                                     1500:0.21, 1550:0.2, 1600:0.18, 1650:0.16, 1700:0.15, 1750:0.14, 1800:0.12, 2000:0.05},
                 filter_eff_dict = {300:0.0001, 495:0.0001, 500:0.5, 505:0.97, 600:0.95, 2000:0.91}, 
                 scattering_loss = 0.33, wavelength_densifying_factor = 3.0, total_dist_slit_to_detector = 0.5 * 10.0 ** 9.0, 
                 focal_length1 = 50.0 * 10.0 ** 6.0, focal_length2 = 50.0 * 10.0 ** 6.0, slit_width = 10.0 * 10.0 ** 3.0, slit_height = 3.0 * 10.0 ** 6.0, front_f_number = 1.2, 
                 diff_order = 1.0, grating_sep = 1.0 / 300.0 * 10.0 ** 6.0, angle_out_lens_vs_diffraction_elem_in_deg = -12.5, pixel_dim = 13.0 * 10.0 ** 3.0, 
                 master_bias_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calData/', master_bias_file = 'MSB_med.fits',
                 master_dark_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calData/', master_dark_file = 'MSD_med.fits'):
        astro_arch = AstronomicalParameterArchive()
        deg_to_rad = astro_arch.getDegToRad()
        deg_to_arcsec = astro_arch.getDegToAsec() 
        self.wavelength_to_pix_funct = wavelength_to_pix_funct
        self.size = size
        self.gain = gain # electrons / ADU 
        self.overscan = overscan
        self.rdnoise = rdnoise
        self.bias_level = 100.0 
        self.dark_current_interp = makeInterpFromDict(dark_current_dict)
        self.spectrum_box = spectrum_box
        self.focal_length1 = focal_length1
        self.focal_length2 = focal_length2
        self.slit_width = slit_width
        self.slit_height = slit_height
        self.diff_order = diff_order
        self.grating_sep = grating_sep 
        self.angle_out_lens_vs_diffraction_elem = angle_out_lens_vs_diffraction_elem_in_deg * deg_to_rad
        self.pixel_width = pixel_dim
        self.pixel_height = pixel_dim
        self.x_center_pixel = size[0] / 2
        self.y_center_pixel = size[1] / 2
        self.front_f_number = front_f_number

        slit_area_sqr_m = self.slit_height * self.slit_width * (10.0 ** -9.0) ** 2.0
        solid_sky_angle = np.arctan(1.0 / front_f_number) ** 2.0 * np.pi * (1.0 / deg_to_rad * deg_to_arcsec) ** 2.0

        system_loss_funct = lambda wavelength: 1.0
        self.wavelength_densifying_factor = wavelength_densifying_factor
        gemini_wavelengths, gemini_photon_flux_interp = read1DSpectrumFromFile(spectrum_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/', spectrum_file = 'GeminiSkyBrightness.txt', n_ignore = 14)
        gemini_detector_photon_rate_interp = interp1d(gemini_wavelengths, [gemini_photon_flux_interp(wavelength) * slit_area_sqr_m * solid_sky_angle * system_loss_funct(wavelength) for wavelength in gemini_wavelengths])
        #self.wavelengths = [gemini_wavelengths[0]] + np.linspace(gemini_wavelengths[1], gemini_wavelengths[-2], len(gemini_wavelengths[1:]) * self.wavelength_densifying_factor).tolist() + [gemini_wavelengths[-1]]
        self.wavelengths = gemini_wavelengths 
        self.photon_rate_funct = gemini_detector_photon_rate_interp 

        if min(self.wavelengths) < min(list(true_qe_dict.keys())): true_qe_dict[int(min(self.wavelengths))] = true_qe_dict[min(list(true_qe_dict.keys()))]
        if max(self.wavelengths) > max(list(true_qe_dict.keys())): true_qe_dict[int(max(self.wavelengths)) + 1] = true_qe_dict[max(list(true_qe_dict.keys()))]
        print ('true_qe_dict = ' + str(true_qe_dict)) 
        true_qe_interp = makeInterpFromDict(true_qe_dict) 
        #print ('true_qe_interp([1.0,10000.0]) = ' + str(true_qe_interp([1.0,10000.0]))) 

        if min(self.wavelengths) < min(list(grating_eff_dict.keys())): grating_eff_dict[int(min(self.wavelengths))] = grating_eff_dict[min(list(grating_eff_dict.keys()))]
        if max(self.wavelengths) > max(list(grating_eff_dict.keys())): grating_eff_dict[int(max(self.wavelengths)) + 1] = grating_eff_dict[max(list(grating_eff_dict.keys()))]
        print ('grating_eff_dict = ' + str(grating_eff_dict)) 
        grating_eff_interp = makeInterpFromDict(grating_eff_dict)
        #print ('grating_eff_interp([1.0,10000.0]) = ' + str(grating_eff_interp([1.0,10000.0]))) 

        if min(self.wavelengths) < min(list(filter_eff_dict.keys())): filter_eff_dict[int(min(self.wavelengths))] = filter_eff_dict[min(list(filter_eff_dict.keys()))]
        if max(self.wavelengths) > max(list(filter_eff_dict.keys())): filter_eff_dict[int(max(self.wavelengths)) + 1] = filter_eff_dict[max(list(filter_eff_dict.keys()))]
        print ('filter_eff_dict = ' + str(filter_eff_dict))
        filter_eff_interp = makeInterpFromDict(filter_eff_dict)
        #print ('filter_eff_interp([1.0,10000.0]) = ' + str(filter_eff_interp([1.0,10000.0])) )

        if min(self.wavelengths) < min(list(front_lens_trans_dict.keys())): front_lens_trans_dict[int(min(self.wavelengths))] = front_lens_trans_dict[min(list(front_lens_trans_dict.keys()))]
        if max(self.wavelengths) > max(list(front_lens_trans_dict.keys())): front_lens_trans_dict[int(max(self.wavelengths)) + 1] = front_lens_trans_dict[max(list(front_lens_trans_dict.keys()))]
        print ('front_lens_trans_dict = ' + str(front_lens_trans_dict))
        front_lens_trans_interp = makeInterpFromDict(front_lens_trans_dict)

        if min(self.wavelengths) < min(list(back_lens_trans_dict.keys())): back_lens_trans_dict[int(min(self.wavelengths))] = back_lens_trans_dict[min(list(back_lens_trans_dict.keys()))]
        if max(self.wavelengths) > max(list(back_lens_trans_dict.keys())): back_lens_trans_dict[int(max(self.wavelengths)) + 1] = back_lens_trans_dict[max(list(back_lens_trans_dict.keys()))]
        print ('back_lens_trans_dict = ' + str(back_lens_trans_dict))
        back_lens_trans_interp = makeInterpFromDict(back_lens_trans_dict)

        self.eff_qe = lambda wavelength: scattering_loss * true_qe_interp(wavelength) * grating_eff_interp(wavelength) * filter_eff_interp(wavelength) * front_lens_trans_interp(wavelength) * back_lens_trans_interp(wavelength) 
        

        self.master_bias_dir = master_bias_dir
        self.master_bias_file = master_bias_file
        self.master_dark_dir = master_dark_dir 
        self.master_dark_file = master_dark_file 
        self.use_master_dark = 0
        self.use_master_bias = 0

        #self.eff_qe = lambda wavelength: 1.0 

        # I may want to actually read in the true master bias and dark images.  But not at the moment 
        # self.master_bias =
        # self.master_dark = 

        
        
