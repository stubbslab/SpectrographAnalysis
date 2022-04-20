"""
Takes a series of monochromatic images and turns them into a mapping of wavelength to
    scatter pattern.  This scatter pattern is used by the ProcessRawSpectrumClass.py
    to correct the scatter from the sky spectrum. Doing so is important, as the
    scatterred light from bright sections of the spectrum, scatterred into faint
    sections, can introduce a large systematic error.

Requires that the following data be taken as well:
    - A bias image, taken between each monochromatic image
    - A background (no light) image, taken with the monochromatic light blocked.
         These images provide a measurement of the common-source background
         illumination.
    - A series of monochromatic images, from which the scatter pattern is computed.
"""
import os
import cantrips as can
import sys
import SpectroscopyReferenceParamsObject as ref_param
import numpy as np
import cosmics_py3 as cosmics
from datetime import datetime
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import scipy
import matplotlib.patches as patches
import SashasAstronomyTools as sat
import cv2

def boxcar_funct(shape, A, h, w, x, y):
    """
    The function for a rectangle in an image.  This shape is fit to the image
       to approximate the monochromatic slit image.
    The arguments are the shape of the image containing the boxcar (shape, used
        to generate the image with the ideal boxcar), and the boxcar paramters
        (A = boxcar amplitude, h = height, w = width, x = horizontal center,
        y = vertical center). Pixels that are partially in the boxcar are
        assigned values equal to A times the fraction of pixel in the boxcar.
        This is important so that fits using this boxcar vary continuously in
        h, w, x and y (otherwise, the fit does not change until those size
        parameters shift by a full pixel).
    """
    col = range(shape[0])
    lb, rb, bb, tb = [x - w //2,  x + w //2, y - h //2,  y + h //2]
    col_frac_in = [ 1.0 if (pix > bb + 0.5 and pix < tb - 0.5) else pix - bb + 0.5 if abs(pix - bb) < 0.5 else tb - pix + 0.5 if abs(pix - tb) < 0.5 else 0.0 for pix in col]
    row = range(shape[1])
    row_frac_in = [ 1.0 if (pix > lb + 0.5 and pix < rb - 0.5) else pix - lb + 0.5 if abs(pix - lb) < 0.5 else rb - pix + 0.5 if abs(pix - rb) < 0.5 else 0.0 for pix in row]
    xs = np.transpose(np.array([col_frac_in for pix in row]))
    ys = np.array([row_frac_in for pix in col])
    boxcar = xs * ys * A

    return boxcar

def boxcar_diff_funct(current_image, A, h, w, x, y):
    """
    The function that determines the goodness of fit (just sum of square differences)
        between an image and an assumed boxcar shape.
    The arguments are the image (current_image) and the boxcar parameters (A =
        amplitude, h = height, w = width, x = horizontal center, y = vertical center).
    """
    boxcar = boxcar_funct(np.shape(current_image), A, h, w, x, y)
    diff_image = (current_image - boxcar)
    diff = np.sum(diff_image ** 2.0)
    return diff

class ScatterFunction:

    def CleanCosmics(self, image_names, image_dir = None, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = 'crc_'):
        """
        Remove cosmic rays (or sharp, bright features) from an image.  Should be applied to
           any exposure that is longer than a few seconds. Uses the LA cosmics algorithm,
           implemented in python using the cosmics_py3.py package.
        Documentation for LA cosmic is available at: http://www.astro.yale.edu/dokkum/lacosmic/
        """
        if image_dir == None:
            image_dir = self.target_dir
        print ('image_names = ' + str(image_names))
        for image_name in image_names:
            #print ('Beginning cosmic ray cleaning for image ' + image_dir + image_name)
            print ('image_name = ' + str(image_name))
            image_array, image_header = cosmics.fromfits(image_dir + image_name)
            cosmic_object = cosmics.cosmicsimage(image_array, readnoise = readnoise, sigclip = sigclip, sigfrac = sigfrac, objlim = objlim, verbose = False)
            cosmic_object.run(maxiter = maxiter)
            image_header['CRCLEANED'] = 'Cosmic rays removed by cosmics.py'
            cosmics.tofits(image_dir + new_image_prefix + image_name, cosmic_object.cleanarray, image_header)
        return 1

    def readInRawSpect(self, target_file, target_dir):
        """
        Reads in a raw spectrum file.  Currently (and far as I can see ahead),
            this means just reading in a .fits image.
        """
        return can.readInDataFromFitsFile(target_file, target_dir)


    def makeMasterBias(self, master_bias_image_file, master_bias_level_file, target_dir,
                       bias_list = 'Bias.list', bias_x_partitions = 2, bias_y_partitions = 2):
        """
        Make a master bias image from a stack of biases.  Ideally, the biases are interleaved
            between science images.  Because the stack of images can be quite large and
            median stacking requires reading in many images into memory at once, this median
            combination uses the smart medianing technique, implemented by the cantrips.py
            library.
        Bias level appears to change slowly over time.   Therefore, the master bias consists of
             two parts: the median of a stack of bias images AFTER THE MEDIAN HAS BEEN SUBTRACTED
             FROM EACH IMAGE and a plot of median bias level in time.
        """
        bias_list_exists = os.path.isfile(target_dir + bias_list)
        if not(bias_list_exists):
            print ('Unable to find bias list: ' + target_dir + bias_list)
            return 0
        bias_images=np.loadtxt(target_dir + bias_list, dtype='str')
        if len(np.shape(bias_images)) == 0:
            bias_images = [str(bias_images)]
        print ('Median combining bias images, in parts ...')
        print ('bias_images = ' + str(bias_images))
        med_bias = can.smartMedianFitsFiles(bias_images, target_dir, bias_x_partitions, bias_y_partitions, subtract_stat = 'mean')[0]
        bias_level_map = sat.measureStatisticsOfFitsImages(bias_images, ax = None, data_dir = target_dir, stat_type = 'mean', show_plot = 0, save_plot = 0, )
        m_bias_header = can.readInDataFromFitsFile(bias_images[-1], target_dir)[1]
        utc_time = datetime.utcnow()
        m_bias_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
        m_bias_header['NCOMBINE'] = (str(len(bias_images)), 'Number of raw biases stacked.')
        m_bias_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')
        #print('med_bias.data = ' + str(med_bias.data))
        #print ('med_bias_header = ' + str(m_bias_header))
        can.saveDataToFitsFile(np.transpose(med_bias), master_bias_image_file, target_dir, header = m_bias_header, overwrite = True, n_mosaic_extensions = 0)
        can.saveListsToColumns(bias_level_map, master_bias_level_file, target_dir, header = 'exp time(s), mean bias level', sep = ',')
        print ('Master bias files created: ')
        print (target_dir + master_bias_image_file)
        print (target_dir + master_bias_level_file)
        return 1

    def plotBiasLevels(self, bias_list = 'BIAS.list', bias_plot_name = 'BiasLevels.pdf'):
        """
        Make a plot of the median bias levels in time.  We read in the bias images
            listed in the text file indicated by the bias_list variable, reads
            the exposure start time parameter from the header, and plots those
            start times against the median level of the bias.  This plots is
            saved to a .pdf file.
        """
        f, axarr = plt.subplots(1,2, figsize = (12, 6))
        bias_list_exists = os.path.isfile(target_dir + bias_list)
        if not(bias_list_exists):
            print ('Unable to find bias list: ' + target_dir + bias_list)
            return 0
        print ('Making plot of bias level in time... ')
        bias_images=np.loadtxt(target_dir + bias_list, dtype='str')
        bias_obs_times, bias_mean = sat.measureStatisticsOfFitsImages(bias_images, ax = axarr[0], data_dir = target_dir, stat_type = 'mean', show_plot = 0, save_plot = 0, save_plot_name = target_dir + 'BiasLevels.pdf', ylabel = 'Mean counts in image', title = 'OSELOTS Bias Mean Level - ut' + ''.join(self.date))
        bias_obs_times, bias_std = sat.measureStatisticsOfFitsImages(bias_images, ax = axarr[1], data_dir = target_dir, stat_type = 'std', show_plot = 0, save_plot = 0, save_plot_name = target_dir + 'BiasLevels.pdf', ylabel = 'Std of counts in image', title = 'OSELOTS Bias Scatter - ut' + ''.join(self.date))
        plt.tight_layout()
        plt.savefig(target_dir + bias_plot_name)

        return 1

    def makeMasterDark(self, master_dark_file, target_dir, master_bias_image_file, master_bias_level_file,
                       dark_list = 'Dark.list', bias_sub_prefix = 'b_',
                       dark_x_partitions = 2, dark_y_partitions = 2,
                       remove_intermediate_files = 0 ):
        """
        Makes a master "dark" image, or image with the spectral source blocked.
            In other words, this is the image of the common-mode illumination
            that does nto depend on the input spectrum.
        Ideally, this images are taken with the lens cover on the entrance
            lens.
        This calculation is performed by first cosmic ray and bias correcting
            each individual no light image.  Then, these processed no light
            images are median stacked.
        This master dark image should be subtracted from each science image.
        """
        dark_list_exists = os.path.isfile(target_dir + dark_list)
        if not(dark_list_exists):
            print ('Unable to find dark list: ' + target_dir + dark_list)
            return 0
        dark_images=np.loadtxt(target_dir + dark_list, dtype='str')
        if len(np.shape(dark_images)) == 0:
            dark_images = [str(dark_images)]

        #bias correct the images
        exp_times = [-1 for dark_file in dark_images]
        for i in range(len(dark_images)):
            dark_file = dark_images[i]
            single_dark_data, single_dark_header = can.readInDataFromFitsFile(dark_file, target_dir)
            single_dark_data, single_dark_header = self.biasSubtract(single_dark_data, single_dark_header, master_bias_image_file, master_bias_level_file)
            exp_times[i] = float(single_dark_header['EXPTIME'])
            can.saveDataToFitsFile(np.transpose(single_dark_data), bias_sub_prefix + dark_file, target_dir, header = single_dark_header, overwrite = True, n_mosaic_extensions = 0)
            m_dark_header = single_dark_header

        #median combine the dark images
        med_dark = can.smartMedianFitsFiles([bias_sub_prefix + dark_image for dark_image in dark_images], target_dir, dark_x_partitions, dark_y_partitions, scalings = [1.0 / time for time in exp_times] )[0]
        if remove_intermediate_files:
            [os.remove(target_dir + bias_sub_prefix + dark_image) for dark_image in dark_images ]
        m_dark_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
        m_dark_header['NCOMBINE'] = (str(len(dark_images)), 'Number of bias-sub darks stacked.')
        m_dark_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')

        can.saveDataToFitsFile(np.transpose(med_dark), master_dark_file, target_dir, header = m_dark_header, overwrite = True, n_mosaic_extensions = 0)
        print ('Master dark file created ' + target_dir + master_dark_file)
        return 1

    def biasSubtract(self, image_data, image_header, master_bias_image_file, master_bias_level_file):
        """
        Subtracts the measured bias level (determined from the makeMasterBias command)
           from the image given by image_data. The 2d bias structure is determined by
           the median stack of the bias images and the median bias level is determined
           by interpolating between the median bias levels of the individual bias
           images, in time.
        """
        start_time_str = image_header[self.obs_time_keyword]
        bias_structure_data, bias_header = can.readInDataFromFitsFile(master_bias_image_file, self.target_dir)
        bias_levels = can.readInColumnsToList(master_bias_level_file, self.target_dir, n_ignore = 1, delimiter = ',', verbose = 0)
        bias_levels = [[float(start_time) for start_time in bias_levels[0]], [float(bias_level) for bias_level in bias_levels[1]]]
        bias_interp = scipy.interpolate.interp1d(bias_levels[0], bias_levels[1], bounds_error = False, fill_value = 'extrapolate')
        start_time_float = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%SZ').timestamp()
        bias_level = float(bias_interp(start_time_float))
        image_data = image_data - bias_structure_data - bias_level
        image_header['BIASSUB'] = (str(datetime.utcnow() ), 'UTC of Bias subtraction')
        image_header['MBIAS'] = (master_bias_image_file, 'Name of Subtracted Master Bias File')
        image_header['BIASLEVL'] = (bias_level, 'Median bias level')

        return image_data, image_header

    def darkSubtract(self, image_data, image_header, master_dark_file, exp_time_keyword = 'EXPTIME'):
        """
        Subtract the 'dark' (meaning common mode) signal from an image. The
            dark data is produced using the makeMasterDark command.  The dark
            image is scaled by the ratio of the image exposure time (read from)
            the provided header) and the dark exposure time (read from the)
            header of the master dark image.
        """
        dark_data, dark_header = can.readInDataFromFitsFile(master_dark_file, target_dir)
        exp_time = float(image_header[exp_time_keyword])
        image_data = image_data - dark_data * exp_time
        image_header['DARKSUB'] = (str(datetime.utcnow() ), 'UTC of Dark subtraction')
        image_header['MDARK'] = (master_dark_file, 'Name of Subtracted Master Dark File')

        return image_data, image_header

    def stackImage(self, current_images, current_headers, combine = 'median'):
        """
        Combine a stack of image arrays (read into RAM) into a single image file.
           The user can specify the type of averaging (mean or median).  Note
           this median combination assumes all data are read into ram (in contrast
           to the SmartMedian function in the cantrips.py library).
        """
        current_header = current_headers[0]
        if combine == 'median':
            current_image = np.median(current_images, axis = self.spec_axis)
            current_header['STACMETH'] = ( 'MEDIAN', 'Method of stacking individual spectra')
        else:
            current_image = np.mean(current_images, axis = self.spec_axis)
            current_header['STACMETH'] = ( 'MEAN', 'Method of stacking individual spectra')
        current_header['NSTACK'] = ( str(len(current_images)), 'Number of individual stacked spectra')
        current_header['EXPTIME'] = (str(sum([float(current_header['EXPTIME']) for current_header in current_headers])), 'Total exposure time of all stacked spectra')
        return current_image, current_header

    def determineSpectralMask(self, image_shape, rect_fit, buffer_in_pix = None):
        """
        Determine the mask that will be applied to image before determining the scatter
            function, based on the fitted boxcar shape.  The mask is either 1s (outside
            of the boxcar), and 0s (inside of the boxcar).
        """
        if buffer_in_pix == None:
            buffer_in_pix = self.mask_buffer_pix
        mask = np.zeros(image_shape) + 1
        h_fit, w_fit, x_fit, y_fit = rect_fit[1:]
        mask[max(y_fit - h_fit // 2 - buffer_in_pix, 0):min(y_fit + h_fit // 2 + buffer_in_pix, image_shape[0]-1),
             max(x_fit - w_fit // 2 - buffer_in_pix, 0):min(x_fit + w_fit // 2 + buffer_in_pix, image_shape[1]-1)] = 0
        return mask

    def splineFitData(self, current_image, rect_fit, n_spline_seeds_in_rect_xy = [10, 40]):
        """
        Compute the "fitted" spectrum by performing a 2d spline interpolation over the section of
           the image where the boxcar spectrum has been identified.  Also determine the mask
           around that boxcar.  This interpolated image is used determine the total number of
           counts in the measured spectrum.
        """
        rect_dims = [ (rect_fit[0][1] - rect_fit[0][0]) , (rect_fit[1][1] - rect_fit[1][0]) ]
        #spline_seed_densities_in_rect_xy = [max(2, np.round(rect_dims[0] // n_spline_seeds_in_rect_xy[0])), max(2, np.round(rect_dims[1] // n_spline_seeds_in_rect_xy[1]))]
        spline_seed_densities_in_rect_xy = [2, 20]
        n_x_seeds, n_y_seeds = [int(np.ceil(rect_dims[0] / spline_seed_densities_in_rect_xy[0])),
                                int(np.ceil(rect_dims[1] // spline_seed_densities_in_rect_xy[1] )) ]
        if n_x_seeds % 2 == 0:
            col_seeds = can.niceReverse([rect_dims[0] // 2 - rect_dims[0] // (n_x_seeds * 2) + (-i) * spline_seed_densities_in_rect_xy[0] for i in range(n_x_seeds // 2) ]) + [rect_dims[0] // 2 + rect_dims[0] // (n_x_seeds * 2) + (i) * spline_seed_densities_in_rect_xy[0] for i in range(n_x_seeds // 2) ]
        else:
            col_seeds = can.niceReverse([rect_dims[0] // 2 + (-i - 1) * spline_seed_densities_in_rect_xy[0] for i in range(n_x_seeds // 2) ]) + [rect_dims[0] // 2 ] + [rect_dims[0] // 2 + (i + 1) * spline_seed_densities_in_rect_xy[0] for i in range(n_x_seeds // 2) ]
        col_seeds = [seed + rect_fit[0][0] for seed in col_seeds]
        if n_y_seeds % 2 == 0:
            row_seeds = can.niceReverse([rect_dims[1] // 2 - rect_dims[1] // (n_y_seeds * 2) + (-i) * spline_seed_densities_in_rect_xy[1] for i in range(n_y_seeds // 2) ]) + [rect_dims[1] // 2 + rect_dims[1] // (n_y_seeds * 2) + (i) * spline_seed_densities_in_rect_xy[1] for i in range(n_y_seeds // 2) ]
        else:
            row_seeds = can.niceReverse([rect_dims[1] // 2 + (-i - 1) * spline_seed_densities_in_rect_xy[1] for i in range(n_y_seeds // 2) ]) + [rect_dims[1] // 2 ] + [rect_dims[1] // 2 + (i + 1) * spline_seed_densities_in_rect_xy[1] for i in range(n_y_seeds // 2) ]
        row_seeds = [seed + rect_fit[1][0] for seed in row_seeds]
        fitted_section = [ [col_seeds[0], col_seeds[-1]], [row_seeds[0], row_seeds[-1]] ]
        #spline_seeds = can.flattenListOfLists([[[col_seed, row_seed] for row_seed in row_seeds] for col_seed in col_seeds]  )
        spline_vals = [[current_image[row_seed, col_seed] for col_seed in col_seeds] for row_seed in row_seeds]
        #We are trying to fit the excess signal above the background.
        # So we force it to tie down to the median.
        image_median = np.median(spline_vals)
        for i in range(len(row_seeds)):
            spline_vals[i][0] = image_median
            spline_vals[i][-1] = image_median
        for i in range(len(col_seeds)):
            spline_vals[0][i] = image_median
            spline_vals[-1][i] = image_median
        """
        f, axarr = plt.subplots(2,1)
        axarr[0].imshow(current_image)
        axarr[0].scatter(can.flattenListOfLists([[col_seed for col_seed in col_seeds] for row_seed in row_seeds]),
                    can.flattenListOfLists([[row_seed for col_seed in col_seeds] for row_seed in row_seeds]), marker = 'x', c = 'r' )
        axarr[1].imshow(np.array(spline_vals))
        plt.show( )
        """
        spline_fit = scipy.interpolate.interp2d(col_seeds, row_seeds, spline_vals, )
        #We are attempting to fit the excess light.  So we subtract off the image median - the average pedastal on which the signal rests.
        fitted_image = spline_fit(range(np.shape(current_image)[0]), range(np.shape(current_image)[1])) - image_median

        return fitted_image, fitted_section

    def extractCentralSpectrum(self, current_image, rect_guess_height = 240, rect_guess_width = 5, rect_spline_scaling = 4):
        """
        Identifies the section defined by the monochromatic image of the slit and removes it from the
            core image.  The returned slit-less image is used to determine the scatter pattern
            at this particular wavelength.
        """
        image_shape = np.shape(current_image)
        boxcar_kernel = np.ones((rect_guess_height, rect_guess_width), dtype = int)
        smoothed_image = fftconvolve(current_image, boxcar_kernel, mode='same')
        smoothed_max_loc = can.find2DMax(smoothed_image[rect_guess_height:-rect_guess_height, rect_guess_width:-rect_guess_width])
        smoothed_max_loc = [smoothed_max_loc[0] + rect_guess_height, smoothed_max_loc[1] + rect_guess_width]
        funct_to_minimize = lambda box_params: boxcar_diff_funct(current_image, *box_params)
        init_guess = [current_image[smoothed_max_loc[0], smoothed_max_loc[1]], rect_guess_height, rect_guess_width, smoothed_max_loc[1], smoothed_max_loc[0]]
        bounds = [(-np.inf, np.inf), (2, image_shape[0] // 2), (2, image_shape[1] // 2), (0, image_shape[0]), (0, image_shape[1])]
        #print ('init_guess = ' + str(init_guess))
        print ('Finding spectrum to mask...')
        print ('bounds = ' + str(bounds))
        rect_fit = scipy.optimize.minimize(funct_to_minimize, init_guess, bounds = bounds)
        print ('Rectangle of spectrum = ' + str(rect_fit))
        rect_fit = rect_fit['x']
        h_init, w_init, x_init, y_init = init_guess[1:]
        h_fit, w_fit, x_fit, y_fit = rect_fit[1:]
        rect_fit = [rect_fit[0], int(rect_fit[1]), int(rect_fit[2]), int(rect_fit[3]), int(rect_fit[4])]
        expanded_rect = [ [int(x_fit - w_fit * (0.5 + rect_spline_scaling)), int(x_fit + w_fit * (0.5 + rect_spline_scaling))], [int(y_fit - h_fit * 0.5 - w_fit * (rect_spline_scaling)), int(y_fit + h_fit * 0.5 + w_fit * (rect_spline_scaling)) ] ]

        fitted_spectrum, mask_section = self.splineFitData(current_image, expanded_rect, )
        plt.close('all')
        f, axarr = plt.subplots(1,3, figsize = (15, 5), squeeze = False)
        axarr[0,0].imshow((current_image))
        axarr[0,0].set_title('Original Image')
        #axarr[0,1].imshow((smoothed_image))
        rect_patch_init = patches.Rectangle((x_init - w_init // 2, y_init - h_init // 2), w_init, h_init, linewidth=1, edgecolor='r', facecolor='none')
        rect_patch_fit = patches.Rectangle((x_fit - w_fit // 2, y_fit - h_fit // 2), w_fit, h_fit, linewidth=1, edgecolor='g', facecolor='none')
        rect_patch_expanded = patches.Rectangle((expanded_rect[0][0], expanded_rect[1][0]), expanded_rect[0][1] - expanded_rect[0][0], expanded_rect[1][1] - expanded_rect[1][0], linewidth=1, edgecolor='orange', facecolor='none')
        #axarr[0,0].add_patch(rect_patch_init)
        #axarr[0,0].add_patch(rect_patch_fit)
        axarr[0,1].imshow((fitted_spectrum))
        axarr[0,1].set_title('Spline Fit of Signal')
        axarr[0,2].imshow((current_image - fitted_spectrum))
        axarr[0,2].set_title('Residual (Scattered) Light')
        plt.tight_layout()
        plt.show(block = False)
        plt.pause(1)
        #"""
        if self.save_image_name !=None:
            can.saveDataToFitsFile(np.transpose(fitted_spectrum), 'SplineFit_' + self.save_image_name, self.target_dir, header = 'default')

        return fitted_spectrum, mask_section

    def interpolateOverSpecRegion(self, spec_diff_image, spectral_fit_mask, averaging_dist = 5, interp_type = 'edge_interp' ):
        """
        Takes an image and a mask section, and interpolates over the mask section based on the
            rest of the image.
        """
        masked_image = can.interpMaskedImage(spec_diff_image, mask_region = spectral_fit_mask, av_dist = averaging_dist, interp_type = interp_type )
        return masked_image

    def determineScatterPattern(self, image_with_scatter):
        """
        Takes in a single monochromatic image of the OSELOTS slit and determines teh scatter pattern
            by identifying the section of the image with the spectru and masking it out.  A Gaussian
            blurring is applied to the residual image, and returned as the scatter pattern.
        """
        spectrum_fit_image, spectral_fit_mask = self.extractCentralSpectrum(image_with_scatter,  )
        spec_diff_image = image_with_scatter - spectrum_fit_image
        masked_image = self.interpolateOverSpecRegion(spec_diff_image, spectral_fit_mask)
        spectral_sum = np.sum(spectrum_fit_image)

        normalized_image = masked_image / spectral_sum
        smoothed_image = cv2.GaussianBlur(normalized_image, (0, 0), self.scatter_blur_gauss_sigma, self.scatter_blur_gauss_sigma)
        #normalized_image = normalized_image * spectral_fit_mask
        return smoothed_image, spectral_sum

    def processSingleWaveImages(self, spec_files_to_reduce, scatter_key, do_bias = None, do_dark = None, crc_correct = None, cosmic_prefix = None, save_stacked_image = None, save_image_name = None, scatter_image_name = None, save_scatter_pattern = 1 ):
        """
        Take in a list of monochromatic OSELOTS image file names taken at the same wavelength and
            determine a scatter pattern from them.  If necessary, intermediate files (master bias
            and master darks) are created along the way.
        The results are written to the scatter_mapping dictionary, providing a mapping between
            wavelength to scatter pattern image files.
        """
        print ('spec_files_to_reduce = '  + str(spec_files_to_reduce ))
        if do_bias is None:
            do_bias = self.do_bias
        if do_dark is None:
            do_dark = self.do_dark
        if crc_correct is None:
            crc_correct = self.crc_correct
        if cosmic_prefix is None:
            cosmic_prefix = self.cosmic_prefix
        if save_stacked_image == None:
            save_stacked_image = self.save_stacked_image
        self.save_image_name = save_image_name

        self.current_images = [[] for spec_file in spec_files_to_reduce]
        self.current_headers = [[] for spec_file in spec_files_to_reduce]
        if crc_correct:
            self.CleanCosmics(spec_files_to_reduce, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = cosmic_prefix)
            spec_files_to_reduce = [cosmic_prefix + spec_file for spec_file in spec_files_to_reduce]
        for i in range(len(spec_files_to_reduce)):
            print ('Reading in raw spectrum from ' + self.target_dir + spec_files_to_reduce[i])
            self.current_images[i], self.current_headers[i] = self.readInRawSpect(spec_files_to_reduce[i], self.target_dir) #Read in raw spectrum
        #Overscan correct (?) ## Not currently set up to do this

        #[OPTIONAL] Make master bias
        if do_bias or do_dark:
            master_bias_exists = os.path.isfile(self.target_dir + self.master_bias_image_file)
            if not(master_bias_exists):
                print ('Making master bias files.  Will be saved to: ' )
                print(self.target_dir + self.master_bias_image_file)
                print(self.target_dir + self.master_bias_level_file)
                master_bias_exists = self.makeMasterBias(self.master_bias_image_file, self.master_bias_level_file, self.target_dir)
            if not(master_bias_exists):
                print ('Unable to find master bias file, ' + self.target_dir + self.master_bias_file + ', and also could not make it.  Returning without processing.')
                #sys.exit()

        #Make master dark (meaning common mode illumination)
        if do_dark :
            master_dark_exists = os.path.isfile(self.target_dir + self.master_dark_file)
            print ('master_dark_exists = ' + str(master_dark_exists))
            print ('self.target_dir + self.master_dark_file = ' + str(self.target_dir + self.master_dark_file))
            if not(master_dark_exists):
                print ('Making master dark file.  Will be saved to ' + self.target_dir + self.master_dark_file)
                master_dark_exists = self.makeMasterDark(self.master_dark_file, self.target_dir, self.master_bias_image_file, self.master_bias_level_file)
            if not(master_dark_exists):
                print ('Unable to find master dark file, ' + self.target_dir + self.master_dark_file + ', and also could not make it.  Returning without processing.')
                #sys.exit()

        #Bias Subtract
        for i in range(len(spec_files_to_reduce)):
            if do_bias or do_dark:
                self.current_images[i], self.current_headers[i] = self.biasSubtract(self.current_images[i], self.current_headers[i], self.master_bias_image_file, self.master_bias_level_file)
            if do_dark:
                print ('dark_subtracting....' )
                self.current_images[i], self.current_headers[i] = self.darkSubtract(self.current_images[i], self.current_headers[i], self.master_dark_file)

        exp_times = [float(header[self.fits_exp_time_keyword]) for header in self.current_headers]
        self.current_images = [ self.current_images[i] / exp_times[i] for i in range(len(self.current_images)) ]
        self.current_image, self.current_header = self.stackImage(self.current_images, self.current_headers, combine = 'mean' )
        #self.current_image = np.median(self.current_images, axis = 0)
        self.current_header = self.current_headers[0]
        self.current_header[self.fits_exp_time_keyword] = 1
        if save_stacked_image and save_image_name != None:
            can.saveDataToFitsFile(np.transpose(self.current_image), save_image_name , self.target_dir, header = self.current_header)

        scatter_pattern, scattering_flux = self.determineScatterPattern(self.current_image, )
        #Probably too much memory to hold onto these scatter images
        self.scatter_mapping[scatter_key] = scatter_pattern
        if save_scatter_pattern and scatter_image_name != None :
            scatter_header = self.current_header
            scatter_header['SCATRWAV'] = (scatter_key, 'Wavelength sourcing scatter')
            scatter_header['SCATRFLX'] = (scattering_flux, 'Central flux sourcing scatter')
            can.saveDataToFitsFile(np.transpose(scatter_pattern), scatter_image_name, self.target_dir, header = self.current_header)
        self.scatter_mapping[scatter_key] = scatter_image_name

        if self.remove_intermed_images and crc_correct:
            [os.remove(self.target_dir + file) for file in spec_files_to_reduce]
        return 1

    def saveScatterMapping(self, save_file_name, save_dir = None, header = None, sep = ',' ):
        """
        Saves the scatter mapping dictionary to a file of two columns.  The first is the
           scattering wavelength and the second is the image file containing the scatter
           pattern. That scatter pattern file is used to correct scatter in the signal
           images.
        """
        if save_dir == None:
            save_dir = self.target_dir
        save_cols = [ list(self.scatter_mapping.keys()), [self.scatter_mapping[key] for key in list(self.scatter_mapping.keys())] ]
        can.saveListsToColumns(save_cols, save_file_name, save_dir, header = header, sep = sep)
        return 1


    def initialize_params_from_ref_params(self):
        """
        Reads in start parameters for the OSELOTS instrument.  They are listed in the
            OSELOTSDefaults.txt file, and are read in using the
            SpectroscopyReferenceParamsObject.py object.
        """
        self.master_bias_image_file = self.ref_param_holder.getMasterBiasImageName()
        self.master_bias_level_file = self.ref_param_holder.getMasterBiasLevelName()
        self.master_dark_file = self.ref_param_holder.getMasterDarkName()
        self.master_bias_list = self.ref_param_holder.getBiasList()
        self.master_dark_list = self.ref_param_holder.getDarkList()
        self.ref_throughput_file = self.ref_param_holder.getRefThroughputFile()
        self.cosmic_prefix = self.ref_param_holder.getCosmicPrefix()
        self.background_buffer = self.ref_param_holder.getBackgroundBuffer()
        self.background_size = self.ref_param_holder.getBackgroundSize()
        self.background_low = self.ref_param_holder.getBackgroundLow()
        self.archival_data_dir = self.ref_param_holder.getArchivalDataDir()
        self.spec_axis = self.ref_param_holder.getSpecAxis()
        self.data_image_suffix = self.ref_param_holder.getImageSuffix()
        self.processed_file_suffix = self.ref_param_holder.getProcessedFileSuffix()
        self.figure_suffix = self.ref_param_holder.getFigureSuffix()
        self.processed_spectra_image_suffix = self.ref_param_holder.getProcessedSpectrumSuffix()
        self.processed_multistep_spectra_image_suffix = self.ref_param_holder.getMultistepSpectrumSuffix()
        self.perp_spec_image_suffix = self.ref_param_holder.getOrthogonalBinOfSpectrumSuffix()
        self.n_std_for_strong_line = self.ref_param_holder.getNStdForStrongLines()
        self.n_std_for_full_ref_line = self.ref_param_holder.getNStdForFullRefLines()
        self.n_std_for_strong_ref_line = self.ref_param_holder.getNStdForStrongRefLines()
        self.n_std_for_full_line = self.ref_param_holder.getNStdForFullLines()
        self.sig_clip_for_line_width = self.ref_param_holder.getSigClipForLineWidth()
        self.ghosts_n_sig_width = self.ref_param_holder.getNSigGhostsWidth()
        self.background_fit_order = self.ref_param_holder.getBackgroundFitOrder()
        self.ghosts_high = self.ref_param_holder.getGhostsHigh()
        self.ghosts_right = self.ref_param_holder.getGhostsRight()
        self.min_ghosts = self.ref_param_holder.getMinGhosts()
        self.removeGhostByShiftingSpectrum = self.ref_param_holder.getRemoveGhostsByShifting ()
        self.clean_buffer = self.ref_param_holder.getCleanBuffer()
        self.n_std_for_most_ghosts = self.ref_param_holder.getNStdForMostGhosts()
        self.n_std_for_first_ghosts = self.ref_param_holder.getNStdForFirstGhosts()
        self.min_detections_for_ident_as_line = self.ref_param_holder.getMinDetectionsForIdentifingALine()
        self.ghost_search_buffer = self.ref_param_holder.getLineWidthFitOrder()
        self.strong_line_search_binning = self.ref_param_holder.getStrongLineSearchBinning()
        self.strong_line_fit_binning = self.ref_param_holder.getStrongLineFitBinning()
        self.max_line_fit_width = self.ref_param_holder.getMaxFittedLineWidth()
        self.line_width_guess = self.ref_param_holder.getInitialLineWidthGuesss()
        self.max_sep_per_pixel_for_line_trace = self.ref_param_holder.getMaxSepForLineTrace()
        self.parallel_smoothing = self.ref_param_holder.getParallelSmoothing()
        self.line_mean_fit_order = self.ref_param_holder.getLineMeanFitOrder()
        self.line_amplitude_fit_order = self.ref_param_holder.getLineAmplitudeFitOrder()
        self.line_width_fit_order = self.ref_param_holder.getLineWidthFitOrder()
        self.wavelength_of_pix_solution_guess = self.ref_param_holder.getWavelengthOfPixSolutionGuess()
        self.min_sig_sep_for_distinct_lines = self.ref_param_holder.getSigSepForDistinctLines()
        self.approx_pix_scale = self.ref_param_holder.getPixScale()  #(nm/pix)
        self.crude_fit_gauss = self.ref_param_holder.getCrudeGaussianSmoothWidth()
        self.default_figsize = self.ref_param_holder.getDefaultFigSize()
        self.wavelength_solution_order = self.ref_param_holder.getWavelengthSolutionOrder()
        self.wavelength_scaling = self.ref_param_holder.getWavelengthScalings()
        self.ref_spec_solution_file = self.ref_param_holder.getRefSpecSolutionFile()
        self.bg_std_buffer = self.ref_param_holder.getBackgroundStdBuffer()
        self.continuum_smoothing = self.ref_param_holder.getContinuumSmoothing()
        self.n_continuum_fit_seeds = self.ref_param_holder.getNContinuumSeeds()
        self.continuum_seeds_file = self.ref_param_holder.getContinuumSeedFile()
        self.n_ignore_for_continuum_seeds = self.ref_param_holder.getNIgnoreContinuumSeedFile()
        self.min_line_vs_seed_sep = self.ref_param_holder.getMaxContinuumSeedLineSep()
        self.throughput_file = self.ref_param_holder.getThroughputFile()
        self.abs_throughput_wavelength, self.abs_throughput_val = self.ref_param_holder.getAbsThroughputData()
        self.ref_sky_lines_file = self.ref_param_holder.getRefSkyLinesFile()
        self.ref_sky_lines_file_n_ignore = self.ref_param_holder.getNIgnoreRefSkyLinesFile()
        self.init_seeing_guess = self.ref_param_holder.getInitSeeing()
        self.seeing_fit_order = self.ref_param_holder.getSeeingOrder()
        self.std_thresh_for_new_line = self.ref_param_holder.getStdThreshForNewLinesOnSlice()
        self.width_pix_sample_to_fit_line = self.ref_param_holder.getPixelWidthToFitALine()
        self.width_pix_sample_to_fit_ref_line = self.ref_param_holder.getPixelWidthToFitARefLine()
        self.refined_width_pix_sample_to_fit_line = self.ref_param_holder.getRefinedPixelWidthToFitALine()
        self.n_pix_above_thresh_for_new_line_in_slice = self.ref_param_holder.getNPixAboveThreshForNewLine()
        self.background_bin_width_for_line_id_in_slice = self.ref_param_holder.getBackgroundBinWidthForNewLineOnSlice()
        self.centroid_adjust_size = self.ref_param_holder.getMaxPixelsThatACentroidCanBeAdjusted()
        self.stacked_image_line_dict_id  = self.ref_param_holder.getLineDictIDOfStackedImage()
        self.bg_sig_clip = self.ref_param_holder.getBackgroundSigClip()
        self.stacked_image_name = self.ref_param_holder.getStackedImageName()
        self.final_spec_image_name = self.ref_param_holder.getStackedImageName()
        self.obs_time_keyword = self.ref_param_holder.getStartExposureKeyword()
        self.date_format_str = self.ref_param_holder.getDateFormatString()
        self.stacked_image_keyword = self.ref_param_holder.getStackedKeyword()
        self.lines_in_dict_keyword = self.ref_param_holder.getLinesDictKeyword()
        self.wavelength_target_range = self.ref_param_holder.getWavelengthRangeOfInterest()
        self.energy_to_photon_scaling = self.ref_param_holder.getWavelengthToPhotonScaling()
        self.background_cut_wavelengths = self.ref_param_holder.getBackgroundCutWavelengths()
        self.bg_fit_range = self.ref_param_holder.getBackgroundFitRegion()
        self.max_dimension = self.ref_param_holder.getImageDimensions()[self.spec_axis]
        self.pix_bin_to_meas_background_noise = self.ref_param_holder.getBinningForBackgroundNoiseMeasurement()
        self.scatter_blur_gauss_sigma = self.ref_param_holder.getScatterBlurGaussSigam()

        self.fits_exp_time_keyword = self.ref_param_holder.getExpTimeKeyword()

        self.wavelength_range = self.ref_param_holder.getWavelengthRangeOfInterest()

        self.throughput_interp = None
        self.mu_of_wavelength_solution = None
        self.wavelength_of_mu_solution = None
        self.spec_range = None

        return 1

    def __init__(self, data_dir = None, do_bias = 1, do_dark = 0, crc_correct = 1, save_stacked_image = 1, remove_intermed_images = 1, mask_buffer_pix = 5,
                 ref_params_file = 'OSELOTSDefaults.txt', ref_params_dir = '/Users/sashabrownsberger/Documents/sashas_python_scripts/skySpectrograph/', date = ['2000', '01', '01'] ):

        self.target_dir = data_dir
        self.date = date
        self.ref_param_holder = ref_param.CommandHolder(spectrograph_file = ref_params_file, defaults_dir = ref_params_dir)

        self.do_bias = do_bias
        self.do_dark = do_dark
        self.crc_correct = crc_correct
        self.save_stacked_image = save_stacked_image
        self.remove_intermed_images = remove_intermed_images
        self.scatter_mapping = {}
        self.mask_buffer_pix = mask_buffer_pix
        self.initialize_params_from_ref_params()


if __name__ == "__main__":
    """
    Run from the command line as:
    $ python DetermineScatterFunction.py

    You should update the following parameters for your specific data:
    date, f_pos, start_bias_number, cal_waves

    You should also double check that the following parameters match your data:
    n_dark_imgs_per_wave (number of no illumination images taken at each wavelength)
    n_bias_imgs_per_wave (number of bias images taken at each wavelength)
    n_exp_imgs_per_wave (number of monochromatic exposures taken at each wavelength)
    """
    date = ['2021', '12', '28']
    f_pos = '26p35'
    target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/' + 'ut' + ''.join([str(elem) for elem in date]) + '/'
    do_dark = 1
    processor = ScatterFunction(data_dir = target_dir, date = date, do_dark = do_dark )

    cal_waves = [int(wave) for wave in np.arange(600, 1000.1, 2)]
    n_dark_imgs_per_wave = 1
    n_bias_imgs_per_wave = 2
    n_exp_imgs_per_wave = 5
    start_bias_number = 36
    KR1_nums = [15, 17, 19]

    n_imgs_per_cycle = (n_dark_imgs_per_wave + n_bias_imgs_per_wave + n_exp_imgs_per_wave)
    bias_nums = [KR1_nums[0] - 1] + [num + 1 for num in KR1_nums] + can.flattenListOfLists( [[[start_bias_number + j * (n_dark_imgs_per_wave + 1) + k * n_bias_imgs_per_wave // 2 + i * n_imgs_per_cycle for k in range(n_bias_imgs_per_wave // 2)] for j in range(n_dark_imgs_per_wave + 1)] for i in range(len(cal_waves))], fully_flatten = 1 )
    print ('bias_nums = ' + str(bias_nums))
    dark_nums = can.flattenListOfLists( [[start_bias_number + n_bias_imgs_per_wave // 2 + j + i * n_imgs_per_cycle for j in range(n_dark_imgs_per_wave)] for i in range(len(cal_waves) - 1)] )
    print ('dark_nums = ' + str(dark_nums))
    cal_nums = can.flattenListOfLists([[start_bias_number + n_bias_imgs_per_wave + n_dark_imgs_per_wave + n_imgs_per_cycle * i + j for j in range(n_exp_imgs_per_wave)] for i in range(-1, len(cal_waves)-1)])
    print ('cal_nums = ' + str(cal_nums))
    bias_imgs = ['Bias_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in bias_nums]
    can.saveListsToColumns(bias_imgs, 'BIAS.list', target_dir)
    processor.plotBiasLevels(bias_list = 'BIAS.list')
    #dark images aren't literally dark (shutter closed).  They are measurements of the spectrum when the source light is off.

    #dark_nums = []
    dark_imgs = ['NoLight_f' + f_pos + '_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in dark_nums]
    can.saveListsToColumns(dark_imgs, 'DARK.list', target_dir)
    #cal_waves = [650, 700, 750, 800, 850, 900, 950, 1000]
    #cal_waves = np.arange(600, 1001, 5)[0:5]
    #cal_waves = []
    raw_cal_imgs = [['Mono_' + str(cal_waves[i]) + 'nm_f' + f_pos + '_' + '_'.join(date) + '_' + str(cal_nums[i*n_exp_imgs_per_wave + j]) + '.fits' for j in range(n_exp_imgs_per_wave)] for i in range(len(cal_waves))]
    proc_single_wave_images = ['proc_Mono_' + str(cal_waves[i]) + 'nm_f' + f_pos + '_' + '_'.join(date) + '.fits' for i in range(len(cal_waves))]
    scatter_single_wave_images = ['scatterPattern_Mono_' + str(cal_waves[i]) + 'nm_f' + f_pos + '_' + '_'.join(date) + '.fits' for i in range(len(cal_waves))]

    #sat.measureStatisticsOfFitsImages(bias_imgs + dark_imgs + can.flattenListOfLists(raw_cal_imgs), data_dir = target_dir, stat_type = 'mean')
    #sys.exit()

    processor.makeMasterBias('BIAS.fits', 'BIAS.txt', target_dir)
    [processor.processSingleWaveImages(raw_cal_imgs[i], cal_waves[i], save_image_name = proc_single_wave_images[i], scatter_image_name = scatter_single_wave_images[i]) for i in range(len(raw_cal_imgs))]

    processor.saveScatterMapping('scatter_map_Mono_' + '_'.join(date) + '.txt', header = 'Mono Wavelength (nm), Scatter fits image')
