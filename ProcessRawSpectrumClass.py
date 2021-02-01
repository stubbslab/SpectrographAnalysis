import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import cantrips as can
import time
from datetime import datetime
import scipy.optimize as optimize
import scipy.special as special
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import cosmics_py3 as cosmics
import sys
import scipy
import SpectroscopyReferenceParamsObject as ref_param

#Use reference emission lines pulled from this paper: https://ui.adsabs.harvard.edu/abs/2003A%26A...407.1157H/abstract
# at this link: http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/407/1157
#Stored in file CombinedReferenceSkyLines.txt
#More accurate, higher wavelength line intensities are available at this link: http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/581/A47
#

"""
Hi.  This file is designed to both reduce and extract spectra from image files taken with the OSELOT Sky Spectograph.
The file can be run from the Command Line.  To execute, you must provide, in order:
[the name of the file to be processed]
[the directory where the image files can be found]
[the name of the master bias file, to be looked for or made]
[the name of the master dark file, to be looked for or made]
[the name of the image fits file after it has been processed]

For example, on my machine (that of Sasha Brownsberger), I can type the following into the command line, and the code runs provided I am in the directory with this file:
$ python ProcessRawSpectrum.py sky_2020_09_23_68.fits /Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut20200925/ BIAS.fits DARK.fits sky_2020_09_23_68_proc.fits sky_2020_09_23_68_spec.png sky_2020_09_23_68_perp_spec.png

The code will first process the image file, saving it to the given processed image file name (sky_2020_09_23_68_proc.fits in my example).
It will then begin to process the spectrum, displaying plots as it does.  This takes numerous steps.  They are roughly as follows:
1. Finding which rows the spectrum occupies.
2. Binning the spectrum along columns.
3. Identifying strong-ish emission lines by looking at the derivative of the spectrum.
4. Fitting a sliding slice of each identified line with a Gaussian on a linear pedestal.  One fit for each row in the spectrum will be fit.  Every 100th or so of these fits will be shown.
5. Fit the profiles of these lines to a polynomial - i.e., how do the shape and centers of these lines as you move from the bottom of the spectrum to the top?
6. Fit the single line polynomial fit paramters to another polynomial function of column id.  This describes how the lines are displayed as a function of pixel along the spectrum.
7. REBIN the spectrum, following these curved fits.
*** The next steps are the part of the work with which I am still struggling.  ***
8. Re-identify lines in this newly binned spectrum, but with much more sensitivity.
9. Match lines to list of identified lines to get the wavelength solution.
"""

class SpectrumProcessor:

    #For Gemini sky lines:
    # ref_file = 'GeminiSkyBrightness.txt', ref_file_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/', ref_file_delimiter = ' ', n_lines_to_ignore = 14
    def importReferenceSpectrum(self, ref_file = None, ref_data = None, ref_file_delimiter = ' ', n_lines_to_ignore = 0, ref_file_dir = ''):
        if ref_data == None:
            if ref_file == None:
                print ('You must specify either a reference spectrum data file or give reference spectrum data data (wavelength, throughput).  Not redifining reference spectrum interpolator. ')
                return 1
            else:
                ref_data = can.readInColumnsToList(ref_file, file_dir = ref_file_dir, n_ignore = n_lines_to_ignore, delimiter = ref_file_delimiter, convert_to_float = 1)
        print('ref_data = ' + str(ref_data))
        self.ref_interp = scipy.interpolate.interp1d(ref_data[0], ref_data[1], fill_value = 0.0, bounds_error=False)

        return 1

    #For OSELOTS:
    # throughput_file = 'OSELOT_throughput.txt', throughput_file_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/', delimiter = ' ', n_lines_to_ignore = 0
    def importSystemThroughput(self, throughput_file = None, throughput_data = None, throughput_file_delimiter = ',', n_lines_to_ignore = 1, throughput_file_dir = None):
        if throughput_data == None:
            if throughput_file == None and self.throughput_file == None:
                print ('You must specify either a throughput data file or give throughput data (wavelength, throughput).  Not redifining throughput interpolator. ')
                return self.throughput_interp
            elif throughput_file == None:
                throughput_file = self.throughput_file
            if throughput_file_dir == None:
                throughput_file_dir = self.archival_data_dir
            throughput_data = can.readInColumnsToList(throughput_file, file_dir = throughput_file_dir, n_ignore = n_lines_to_ignore, delimiter = throughput_file_delimiter, convert_to_float = 1)
        throughput_interp = scipy.interpolate.interp1d(throughput_data[0], throughput_data[1] , fill_value = 0.0, bounds_error=False)

        return throughput_interp

    def CleanCosmics(self, image_dir, image_names, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = 'crc_'):
        for image_name in image_names:
            #print ('Beginning cosmic ray cleaning for image ' + image_dir + image_name)
            image_array, image_header = cosmics.fromfits(image_dir + image_name)
            cosmic_object = cosmics.cosmicsimage(image_array, readnoise = readnoise, sigclip = sigclip, sigfrac = sigfrac, objlim = objlim, verbose = False)
            cosmic_object.run(maxiter = maxiter)
            image_header['CRCLEANED'] = 'Cosmic rays removed by cosmics.py'
            cosmics.tofits(image_dir + new_image_prefix + image_name, cosmic_object.cleanarray, image_header)
        return 1

    def readInRawSpect(self, target_file, target_dir):
        return can.readInDataFromFitsFile(target_file, target_dir)

    def makeMasterBias(self, master_bias_file, target_dir,
                       bias_list = 'Bias.list', bias_x_partitions = 2, bias_y_partitions = 2):
        bias_list_exists = os.path.isfile(target_dir + bias_list)
        if not(bias_list_exists):
            print ('Unable to find bias list: ' + target_dir + bias_list)
            return 0
        bias_images=np.loadtxt(target_dir + bias_list, dtype='str')
        if len(np.shape(bias_images)) == 0:
            bias_images = [str(bias_images)]
        print ('Median combining bias images, in parts ...')
        print ('bias_images = ' + str(bias_images))
        med_bias = can.smartMedianFitsFiles(bias_images, target_dir, bias_x_partitions, bias_y_partitions)[0]
        m_bias_header = can.readInDataFromFitsFile(bias_images[-1], target_dir)[1]
        utc_time = datetime.utcnow()
        m_bias_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
        m_bias_header['NCOMBINE'] = (str(len(bias_images)), 'Number of raw biases stacked.')
        m_bias_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')

        print ('med_bias = ' + str(med_bias))
        #print('med_bias.data = ' + str(med_bias.data))
        #print ('med_bias_header = ' + str(m_bias_header))
        can.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = m_bias_header, overwrite = True, n_mosaic_extensions = 0)
        #c.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = 'default', overwrite = True, n_mosaic_extensions = 0)
        print ('Master bias file created ' + target_dir + master_bias_file)
        return 1

    def biasSubtract(self, image_data, image_header, master_bias_file):
        bias_data, bias_header = can.readInDataFromFitsFile(master_bias_file, self.target_dir)
        image_data = image_data - bias_data
        image_header['BIASSUB'] = (str(datetime.utcnow() ), 'UTC of Bias subtraction')
        image_header['MBIAS'] = (master_bias_file, 'Name of Subtracted Master Bias File')

        return image_data, image_header

    def darkSubtract(self, image_data, image_header, master_dark_file, exp_time_keyword = 'EXPTIME'):
        dark_data, dark_header = can.readInDataFromFitsFile(master_dark_file, target_dir)
        exp_time = float(image_header[exp_time_keyword])
        image_data = image_data - dark_data * exp_time
        image_header['DARKSUB'] = (str(datetime.utcnow() ), 'UTC of Dark subtraction')
        image_header['MDARK'] = (master_bias_file, 'Name of Subtracted Master Bias File')

        return image_data, image_header

    def makeMasterDark(self, master_dark_file, target_dir, master_bias_file,
                       dark_list = 'Dark.list', bias_sub_prefix = 'b_',
                       dark_x_partitions = 2, dark_y_partitions = 2,
                       remove_intermediate_files = 0 ):
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
            single_dark_data, single_dark_header = biasSubtract(single_dark_data, single_dark_header, master_bias_file)
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

    def determineSpecRowRanges(self, current_image,
                               sum_method = 'sum', n_sig_deriv_spikes_for_spec = 1.5, n_sig_gauss_to_define_spec_width = 2.0 , sig_step = 0.5, showIDedLines = 1, save_perp_spec_image = 0, perp_spec_image_name = 'IdentifiedSpecRange.png'):
        perp_spec_axis = (self.spec_axis + 1) % 2
        perp_spec = np.sum(current_image, axis = perp_spec_axis)
        if self.bg_std_buffer > 0:
            perp_spec_smooth = can.smoothList(perp_spec[self.bg_std_buffer:-self.bg_std_buffer], smooth_type = 'boxcar', averaging = 'median', params = [50])
        else:
            perp_spec_smooth = can.smoothList(perp_spec, smooth_type = 'boxcar', averaging = 'median', params = [50])
        perp_spec_peak_loc = np.argmax(perp_spec_smooth)
        perp_len = len(perp_spec_smooth)
        perp_spec_derivs = np.gradient(perp_spec_smooth)
        perp_derivs_med = np.median(perp_spec_derivs)
        perp_derivs_std = np.std(perp_spec_derivs)
        deriv_indeces_above_std = []
        while len(deriv_indeces_above_std) == 0:
            deriv_indeces_above_std = [i for i in range(len(perp_spec_derivs)) if abs (perp_spec_derivs[i] - perp_derivs_med) / perp_derivs_std >= n_sig_deriv_spikes_for_spec ]
            if len(deriv_indeces_above_std) == 0: n_sig_deriv_spikes_for_spec = n_sig_deriv_spikes_for_spec - sig_step
        print ('deriv_indeces_above_std = ' + str(deriv_indeces_above_std))
        print ('perp_spec_peak_loc = ' + str(perp_spec_peak_loc))
        left_slope_around_peak = np.min([index for index in deriv_indeces_above_std if index < perp_spec_peak_loc])
        right_slope_around_peak = np.max([index for index in deriv_indeces_above_std if index > perp_spec_peak_loc])
        print ('[left_slope_around_peak, right_slope_around_peak] = ' + str([left_slope_around_peak, right_slope_around_peak]))
        perp_line_step_up, perp_line_step_down = [left_slope_around_peak, right_slope_around_peak]
        fit_spect_funct = lambda xs, A, l, r, A0: A * np.where(xs < r, 1, 0 ) * np.where(xs > l, 1, 0 ) + A0
        #fit_spect_funct = lambda xs, A, mu, sig, alpha, A0: A * np.exp(-(np.abs(np.array(xs) - mu)/ (np.sqrt(2.0) * sig )) ** alpha) + A0
        init_guess = [np.max(perp_spec_smooth) , perp_line_step_up, perp_line_step_down, np.median(perp_spec_smooth)]
        #init_guess = [np.max(perp_spec) , perp_spec_peak_loc, (perp_line_step_down - perp_line_step_up) / 2.0, 2.0, 0.0]
        fitted_profile = optimize.minimize(lambda params: np.sqrt(np.sum(np.array(fit_spect_funct(list(range(perp_len)), *params) - perp_spec_smooth) ** 2.0)) / perp_len, x0 = init_guess)['x']
        print('fitted_profile = ' + str(fitted_profile))
        fitted_variance = np.sqrt(fitted_profile[2] ** 2.0 * special.gamma(3.0 / fitted_profile[3]) / special.gamma(1.0 / fitted_profile[3]))
        left_side, right_side = [fitted_profile[1], fitted_profile[2]]
        #left_side, right_side = (fitted_profile[1] - n_sig_gauss_to_define_spec_width * fitted_variance, fitted_profile[1] + n_sig_gauss_to_define_spec_width * fitted_variance)
        #print ('init_box_guess[0] * np.where(list(range(perp_len)) > perp_line_step_up, 1, 0) * np.where(list(range(perp_len)) < perp_line_step_down, 1, 0) = ' + str(init_box_guess[0] * np.where(list(range(perp_len)) > perp_line_step_up, 1, 0) * np.where(list(range(perp_len)) < perp_line_step_down, 1, 0)))
        if perp_line_step_up > perp_line_step_down:
            print ('Cannot identify location of spectrum on image.  Returning 0s.')
            return [-1, -1]
        elif showIDedLines or save_perp_spec_image:
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize )
            true_plot = axarr[0].plot(range(perp_len), perp_spec_smooth, color = 'k')[0]
            axarr[0].set_xlabel('Row number (pix)')
            axarr[0].set_ylabel('Smoothed counts binned perp. (ADU)')
            #fitted_plot = axarr[0].plot(range(perp_len), fit_spect_funct(list(range(perp_len)), *fitted_profile), c = 'orange')[0]
            spec_region = axarr[0].axvline(left_side, c = 'r')
            #axarr[0].legend([true_plot, fitted_plot, spec_region], ['True binned data', 'Generalized gauss fit', 'Spectrum region edges'])
            axarr[0].legend([true_plot, spec_region], ['Smoothed data orthogonal to spectra (ADU)', 'Chosen spectrum region edges'])
            axarr[0].axvline(right_side, c = 'r')
            derivs = axarr[1].plot(range(len(perp_spec_derivs)), perp_spec_derivs, c = 'k')[0]
            axarr[1].legend([derivs], ['Pixel derivative of true binned data'])
            axarr[1].set_xlabel('Column number (pix)')
            axarr[1].set_ylabel('Deriv of smoothed data (ADU/pix)')
            axarr[0].set_title('Determination of spectrum region by fitting to Sum Perpendicular to Spectrum')
            plt.tight_layout()
            if save_perp_spec_image:
                plt.savefig(perp_spec_image_name)
            if showIDedLines:
                plt.show()

        return [perp_line_step_up + self.bg_std_buffer, perp_line_step_down + self.bg_std_buffer]

    def fullSpecFitFunct(self, ref_xs, ref_ys, lines, ref_params):
        n_points = len(ref_xs)
        seeing_by_pixel_funct = np.poly1d(ref_params)
        #print ('[ref_xs, [line[1] for line in lines]] = ' + str([ref_xs, [line[1] for line in lines]]))
        fit_ys = np.sum([line[0] * np.exp(-(line[1] - ref_xs) ** 2.0 / (2.0 * seeing_by_pixel_funct(line[1] - n_points // 2) ** 2.0 ))  for line in lines], axis = 0)
        residual = np.sqrt(np.sum((fit_ys - np.array(ref_ys)) ** 2.0))
        #f, axarr = plt.subplots(2,1, figsize = [15,4])
        #axarr[0].plot(ref_xs, seeing_by_pixel_funct(ref_xs- n_points // 2))
        #axarr[1].plot(ref_xs, ref_ys)
        #axarr[1].plot(ref_xs, fit_ys)
        #plt.draw()
        #plt.pause(0.1)
        #plt.close('all')
        #print ('[ref_params, residual] = ' + str([ref_params, residual]))
        return residual


    def simulFitLineWidths(self, ref_xs, ref_ys, ref_emission_lines, init_guess = None):
        if init_guess == None:
            init_guess = np.zeros(self.seeing_fit_order + 1)
            init_guess[-1] = self.init_seeing_guess
        best_seeing_funct = scipy.optimize.minimize(lambda params: self.fullSpecFitFunct(ref_xs, ref_ys, ref_emission_lines, params), init_guess)
        pivot_x = ref_xs[len(ref_xs) // 2]
        rescaled_best_seeing_params = [best_seeing_funct['x'][0], best_seeing_funct['x'][1] - 2 * best_seeing_funct['x'][0] * pivot_x,  best_seeing_funct['x'][0] * pivot_x ** 2.0 - best_seeing_funct['x'][1] * pivot_x + best_seeing_funct['x'][2]]
        return rescaled_best_seeing_params

    def SingleLineFunctionToMinimize(self, xs, ys, fit_width, fit_funct, fit_params):
        central_pix = round(fit_params[1])
        if np.isnan(central_pix) or np.isnan(fit_width) or np.isnan(len(xs)) :
            print ('[fit_params, central_pix, fit_width, len(xs)] = ' + str([fit_params, central_pix, fit_width, len(xs)]))
            print ('One of the passed single-fitting line parameters was a nan.  That is not an acceptable set of parameters.  Returning an infinite fit result: ')
            return np.inf
            #if np.isnan(fit_params[1]):
            #    fit_params[1] = 0.0
            #    fit_width=0.0
            #    print ('Reassigning a nan value in the fit params to 0... ')
        fit_indeces = [max(int(central_pix - fit_width), 0), min(int(central_pix + fit_width) + 1, len(xs) - 1)]
        sub_xs = xs[fit_indeces[0]:fit_indeces[1]]
        sub_ys = ys[fit_indeces[0]:fit_indeces[1]]
        fit_ys = fit_funct(sub_xs, *fit_params)
        fit_res = np.sqrt(np.sum((fit_ys - np.array(sub_ys)) ** 2.0))

        #print ('[fit_params, fit_res] = ' + str([fit_params, fit_res] ))
        #plt.scatter(sub_xs, sub_ys)
        #plt.plot(sub_xs, fit_ys)
        #plt.draw()
        #plt.pause(0.1)
        #plt.close('all')

        return fit_res


    def fitSingleLine(self, xs, ys, init_guess, fit_width = 2, bounds = ([0.0, 0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]), show_fit = 0, pedestal = 1, width_from_focus_funct = 0, seeing_fit_funct = None, verbose = 0):
        #print ('!!!fit width = ' + str(fit_width) + '!!!')
        if width_from_focus_funct and seeing_fit_funct == None:
            seeing_fit_funct = self.seeing_fit_funct
        #print ('bounds = ' + str(bounds))
        if pedestal:
            if width_from_focus_funct:
                init_guess = [init_guess[0], init_guess[1], init_guess[3] ]
                bounds =  [[bounds[0][0], bounds[0][1], bounds[0][3], ], [bounds[1][0], bounds[1][1], bounds[1][3]]]
                fit_funct = lambda xs, A, mu, shift, : A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * seeing_fit_funct(mu) ** 2.0)) + shift + 0.0 * (np.array(xs) - np.mean(xs))
            else:
                init_guess = [init_guess[0], init_guess[1], init_guess[2], init_guess[3] ]
                bounds =  [[bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3], ], [bounds[1][0], bounds[1][1], bounds[1][2], bounds[1][3]]]
                fit_funct = lambda xs, A, mu, sig, shift, : A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift + 0.0 * (np.array(xs) - np.mean(xs))
        else:
            if width_from_focus_funct:
                fit_funct = lambda xs, A, mu: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * seeing_fit_funct(mu) ** 2.0)) + 0.0 + 0.0 * (np.array(xs) - np.mean(xs))
                init_guess = [init_guess[0], init_guess[1]]
                bounds = [[bounds[0][0], bounds[0][1]], [bounds[1][0], bounds[1][1]]]
            else:
                fit_funct = lambda xs, A, mu, sig, : A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + 0.0 + 0.0 * (np.array(xs) - np.mean(xs))
                init_guess = init_guess[0:-2]
                bounds = [bounds[0][0:-2], bounds[1][0:-2]]
        #if show_fit:
        #    just_gauss_fit = plt.plot(xs, fit_funct(xs, *init_guess), c = 'red')

        bounds = [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
        #print ('[init_guess, bounds] = ' + str([init_guess, bounds] ))
        if np.any(np.isnan(init_guess)): print ('[init_guess, bounds] = ' + str([init_guess, bounds]))
        #if verbose: print ('[fit_width, fit_funct, fit_params, init_guess] = ' + str([fit_width, fit_funct, init_guess] ))
        fit_res = optimize.minimize(lambda fit_params: self.SingleLineFunctionToMinimize(xs, ys, fit_width, fit_funct, fit_params), init_guess, bounds = bounds)
        #fit_res = optimize.minimize(lambda params: np.sqrt(np.sum((fit_funct(fit_xs[params[1] - fit_width:params[1] + fit_width + 1]) - np.array(fit_ys[params[1] - fit_width:params[1] + fit_width + 1])) ** 2.0)), init_guess)
        fit_res = fit_res['x'].tolist()
        if np.any(np.isnan(fit_res)):
            print ('Fitted line returned a nan.  We will reassign to the initial guess: ')
            fit_res = init_guess
        #print ('fit_res = ' + str(fit_res))
        """
        try:
            #fit_res = optimize.curve_fit(fit_funct, fit_xs, fit_ys, p0 = init_guess, bounds = bounds)[0].tolist()
            fit_res = optimize.minimize(lambda fit_params: self.SingleLineFunctionToMinimize(fit_xs, fit_ys, fit_width, fit_funct, fit_params), init_guess)
            #fit_res = optimize.minimize(lambda params: np.sqrt(np.sum((fit_funct(fit_xs[params[1] - fit_width:params[1] + fit_width + 1]) - np.array(fit_ys[params[1] - fit_width:params[1] + fit_width + 1])) ** 2.0)), init_guess)
            fit_res = fit_res['x']
            print ('fit_res = ' + str(fit_res))
        except (RuntimeError, TypeError) :
            print ('Failed to fit one possible line with initial guess: ' + str(init_guess))
            fit_res = init_guess[:]
            #plt.plot(fit_xs, fit_ys, c = 'k')
            #plt.plot(fit_xs, fit_funct(fit_xs, *init_guess), c = 'red')
            #plt.show()
        """
        init_central_pix = round(init_guess[1])
        init_indeces = [int(max(init_central_pix - fit_width, 0)), int(min(init_central_pix + fit_width, len(xs) - 1))]
        init_xs = xs[init_indeces[0]:init_indeces[1]]
        init_ys = ys[init_indeces[0]:init_indeces[1]]
        fit_central_pix = round(fit_res[1])
        fit_indeces = [max(fit_central_pix - fit_width, 0), min(fit_central_pix + fit_width + 1, len(xs) - 1)]
        fit_indeces = [int(fit_indeces[0]), int(fit_indeces[1])]
        fit_xs = xs[fit_indeces[0]:fit_indeces[1]]
        fit_ys = ys[fit_indeces[0]:fit_indeces[1]]
        if show_fit:
            just_gauss_fit = plt.plot(init_xs, fit_funct(init_xs, *init_guess), c = 'red')[0]
            gauss_and_pedestal_fit = plt.plot(fit_xs, fit_funct(fit_xs, *fit_res), c = 'green')[0]
        fit_sum_of_sqrs = np.sum((np.array(fit_ys) - np.array(fit_funct(fit_xs, *fit_res))) ** 2.0)
        mean_sum_of_sqrs = np.sum((np.array(fit_ys) - np.mean(fit_ys)) ** 2.0)
        fit_res = fit_res + [fit_sum_of_sqrs, mean_sum_of_sqrs]
        #print ('fit_res = ' + str(fit_res))
        #if fit_res[0] <0.0: print ('[bounds, fit_res] = ' + str([bounds, fit_res]))
        if pedestal and width_from_focus_funct:
            fit_res = [fit_res[0], fit_res[1], seeing_fit_funct(fit_res[1]), fit_res[2], fit_res[3], fit_res[4]]
        elif not(pedestal) and width_from_focus_funct:
            fit_res = [fit_res[0], fit_res[1], seeing_fit_funct(fit_res[1]), 0.0, fit_res[2], fit_res[3]]
        elif not(pedestal) and not(width_from_focus_funct):
            fit_res = [fit_res[0], fit_res[1], fit_res[2], 0.0, fit_res[3], fit_res[4]]

        if show_fit:
            return [just_gauss_fit, gauss_and_pedestal_fit, fit_res]
        else:
            return fit_res

    def identifyLinesOnSlice(self, xs, ys,
                             max_line_fit_width = None, peak_guesses = [], show_spec = 0, verbose = 1 ,
                             fit_lines_with_pedestal = 1, fit_line_width_with_seeing_funct = 0,
                             seeing_fit_funct = None  ):

        std_thresh = self.std_thresh_for_new_line
        init_fit_width_guess = self.init_seeing_guess

        if max_line_fit_width == None:
            max_line_fit_width = self.width_pix_sample_to_fit_line
        n_pix_above_thresh = self.n_pix_above_thresh_for_new_line_in_slice
        background_bin_width = self.background_bin_width_for_line_id_in_slice
        search_width = self.centroid_adjust_size
        if len(peak_guesses) == 0:
            n_pix = len(xs)
            bg_from_binning = [np.median(ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
            bg_ys = ([ys[i] - bg_from_binning[i] for i in range(n_pix)])
            bg_ys_med = np.median(bg_ys)
            bg_ys_std = np.std(bg_ys)
            bg_ys_std = [np.std(bg_ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
            pix_vals_above_std = [pix for pix in xs[1:-1] if bg_ys[pix] > bg_ys_med + std_thresh * bg_ys_std[pix]]
            #Our guesses for the centers of ghost lines are those pixels that are both part of a group of pixels that are some threshold above the noise and are local extremum
            peak_guesses = ([0] if ((np.mean(bg_ys[0:int(n_pix_above_thresh/2 + 0.5)]) > bg_ys_med + std_thresh * bg_ys_std[0]) and (ys[0] > ys[1])) else [] )
            peak_guesses = peak_guesses + [pix for pix in pix_vals_above_std if ((ys[pix] > ys[pix-1] and ys[pix] > ys[pix+1]) and (np.mean(bg_ys[max(0, pix-int(n_pix_above_thresh/2)):min(pix+int(n_pix_above_thresh + 0.5), n_pix-1)]) > bg_ys_med + std_thresh * bg_ys_std[pix])) ]
            peak_guesses = peak_guesses + ([xs[-1]] if ((np.mean(bg_ys[-int(n_pix_above_thresh/2 + 0.5):]) > bg_ys_med + std_thresh * bg_ys_std[-1]) and (ys[-1] > ys[-2])) else [] )
        if len(peak_guesses) == 0:
            print ('No significant peaks detected on slice.' )
        n_peak_guesses = len(peak_guesses)
        if verbose:
            print ('peak_guesses = ' + str(peak_guesses))
            print ('xs = ' + str(xs))
            print ('n_peak_guesses = ' + str(n_peak_guesses))
        line_fits = [0 for peak in peak_guesses]
        if show_spec:
            fig = plt.figure(figsize = self.default_figsize)
            spec_plot = plt.plot(xs, ys, c = 'blue')[0]
            plt.xlabel('Pixel number (column)')
            plt.ylabel('Binned spectrum (ADU)')
            plt.title('Fits to IDed lines on spectrum slice')
        for j in range(0, n_peak_guesses ):
            peak_guess = peak_guesses[j]
            #if n_peak_guesses == 1:
            #    fit_xs = list( range(max(int(peak_guess - max_line_fit_width), xs[0]),
            #                         min(int(peak_guess + max_line_fit_width) + 1, n_pix)) )
            #elif j == 0:
                #fit_xs = list( range(max(int(peak_guess - max_line_fit_width), 0),
                #                     min(int(peak_guess + max_line_fit_width), int(peak_guess + peak_guesses[j+1]) // 2, xs[-1])) )
            #elif j == n_peak_guesses - 1:
            #    fit_xs = list( range(max(int(peak_guess - max_line_fit_width), int(peak_guesses[j-1] + peak_guesses[j]) // 2, xs[0]),
            #                     min(int(peak_guess + max_line_fit_width), xs[-1])) )
            #else:
            #    fit_xs = list( range(max(int(peak_guess - max_line_fit_width), int(peak_guesses[j-1] + peak_guess) // 2, xs[0]),
            #                     min(int(peak_guess + max_line_fit_width), int(peak_guess + peak_guesses[j+1]) // 2, xs[-1])) )
            fit_xs = list( range(max(int(peak_guess - max_line_fit_width), xs[0]),
                                      min(int(peak_guess + max_line_fit_width) + 1, xs[-1])) )
            fit_index_range = [fit_xs[0] - xs[0], fit_xs[-1] - xs[0]]
            #print('[fit_xs, ys] = ' + str([fit_xs, ys]))
            if len(fit_xs) <= 1:
                print ('fit_xs = ' + str(fit_xs))
                print ('xs = ' + str(xs))
                print ('[peak_guess, max_line_fit_width, j, n_peak_guesses, peak_guesses, len(xs)] = ' + str([peak_guess, max_line_fit_width, j, n_peak_guesses, peak_guesses, len(xs)]))
            fit_ys = ys[fit_index_range[0]:fit_index_range[-1] + 1]
            #print ('[fit_xs, fit_ys] = ' + str([fit_xs, fit_ys]))
            init_guess = [max(fit_ys), peak_guess, init_fit_width_guess, 0.0, 0.0 ]
            lower_bounds = [-np.inf, init_guess[1] - search_width, 0.2, -np.inf, -np.inf ]
            lower_bounds = [min(lower_bounds[i], init_guess[i]) for i in range(len(init_guess))]
            upper_bounds = [2.0 * init_guess[0], init_guess[1] + search_width, 10.0, init_guess[1] + init_guess[3], np.inf]
            upper_bounds = [max(upper_bounds[i], init_guess[i]) for i in range(len(init_guess))]
            if show_spec:
                line_fits_and_plots = self.fitSingleLine(xs, ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
                just_gauss_fit, gauss_on_pedestal_fit, line_fits[j] = [line_fits_and_plots, line_fits_and_plots, line_fits_and_plots[-1]]
                #fit_funct = lambda xs, A, mu, sig, shift, : A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift + 0.0 * (np.array(xs) - np.mean(xs))
                #print ('init_guess[0:4] = ' + str(init_guess[0:4]))
                #plt.plot(fit_xs, fit_funct(fit_xs, *(init_guess[0:4])), c = 'orange')
            else:
                #if verbose: print('init_guess = ' + str(init_guess))
                line_fits[j] = self.fitSingleLine(xs, ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
        if n_peak_guesses < 1:
            just_gauss_fit = plt.plot(np.nan, np.nan, '-', color = 'red')[0]
            gauss_on_pedestal_fit = plt.plot(np.nan, np.nan, '-', color = 'green')[0]
            line_fits = []

        if show_spec:
            print ('[spec_plot, just_gauss_fit, gauss_on_pedestal_fit] = ' + str([spec_plot, just_gauss_fit, gauss_on_pedestal_fit]))
            if n_peak_guesses >= 1:
                plt.legend([spec_plot, just_gauss_fit, gauss_on_pedestal_fit], ['Spectrum on slice', 'Just line fits', 'Line + pedestal fits'])
            else:
                plt.legend([spec_plot, just_gauss_fit, gauss_on_pedestal_fit], ['Spectrum on slice', 'Just line fits (none detected)', 'Line + pedestal fits (none detected)'])
            #plt.show()
            #plt.draw()
            #plt.pause(0.1)
            #plt.close('all')
            plt.show()
        #print ('line_fits = ' + str(line_fits))
        return line_fits

    def extendLinesIntoImage(self, range_to_extend, line_extensions, data_to_search, ref_line_ends,
                             binning_search_width = 3, max_sep_per_pix = 5,
                             max_frac_intensity_per_pix = 0.1, line_bg_width = 10,
                             max_seq_fit_failures = 3, max_seq_extreme_amps = 5,
                             bound_tolerance = 0.01):

        max_frac_intensity_per_pix = 0.2
        n_failures = [0 for line in line_extensions]
        n_extreme_amps = [0 for line in line_extensions]

        for i in range(len(range_to_extend)):
            pix = range_to_extend[i]
            if self.spec_axis == 0:
                spec_slice = np.median(data_to_search[max(0, pix - int(binning_search_width/2)):pix + int(binning_search_width/2 + 0.5), :], axis = self.spec_axis)
            else:
                spec_slice = np.median(data_to_search[:, max(0, pix - int(binning_search_width/2)):pix + int(binning_search_width/2 + 0.5)], axis = self.spec_axis)

            for j in range(len(line_extensions)):
                if (n_failures[j] <= max_seq_fit_failures and n_extreme_amps[j] <= max_seq_extreme_amps):
                    line_extension = line_extensions[j][i]
                    ref_line_end = ref_line_ends[j]
                    if (i == 0 or line_extensions[j][i-1] is None):
                        prev_line = ref_line_end
                    else:
                         prev_line = line_extensions[j][i-1]
                     #Search the slice for the continuous_line
                    guess_height = prev_line[1]
                    guess_center = prev_line[2]
                    guess_width = prev_line[3]
                    guess_bg = prev_line[4]
                    init_guess = [guess_height, guess_center, guess_width, guess_bg]
                    lower_bounds = [guess_height * 1.0 / (1.0 + max_frac_intensity_per_pix), guess_center - max_sep_per_pix, 1.0, -np.inf]
                    upper_bounds = [guess_height * (1.0 + max_frac_intensity_per_pix), guess_center + max_sep_per_pix, line_bg_width, np.inf]
                    #lower_bounds = [0.0, guess_center - max_sep_per_pix, 1.0, -np.inf]
                    #upper_bounds = [np.inf, guess_center + max_sep_per_pix, line_bg_width, np.inf]

                    fit_xs = list(range(max(int(guess_center) - int(line_bg_width / 2), 0), min(int(guess_center) + int(line_bg_width / 2 + 0.5), len(spec_slice))) )
                    fit_ys = spec_slice[fit_xs[0]:fit_xs[-1]+1]
                    fit_funct = lambda xs, A, mu, sig, shift: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift
                    possible_line = fitSingleLine(fit_xs, fit_ys, init_guess, bounds = (lower_bounds, upper_bounds), show_fit =  0)
                    if j == 2:
                        print ('[pix, lower_bounds[0], upper_bounds[0], init_guess[0], possible_line[0]] = ' + str([pix, lower_bounds[0], upper_bounds[0], init_guess[0], possible_line[0]]))
                    if possible_line[0] > 0.0:
                        if not( (possible_line[0] >= lower_bounds[0] + (upper_bounds[0] - lower_bounds[0]) * bound_tolerance)
                                and (possible_line[0] <= upper_bounds[0] - (upper_bounds[0] - lower_bounds[0]) * bound_tolerance) ):
                            n_extreme_amps[j] = n_extreme_amps[j] + 1
                            print ('Hit extreme amplitude for ' + str(j) + 'th line at pixel ' + str(pix) + '.  This is the ' + str(n_extreme_amps[j]) + 'th in a row.')
                        else:
                            n_extreme_amps[j] = 0

                        line_extensions[j][i] = [pix] + possible_line
                        n_failures[j] = 0
                    else:
                        print ('Failed to extend ' + str(j) + 'th line to pixel ' + str(pix) )
                        n_failures[j] = n_failures[j] + 1
        print ('line_extensions[2] = ' + str(line_extensions[2]))
        print ('n_extreme_amps = ' + str(n_extreme_amps))
        line_extensions = [[detection for detection in line if not(detection is None) ] for line in line_extensions]
        #Cut off the final portions of the line extensions if they were considered extreme
        for j in range(len(line_extensions)):
            if n_extreme_amps[j] > 0:
                line_extensions[j] = line_extensions[j][0:-n_extreme_amps[j]]
        print ('line_extensions[2] = ' + str(line_extensions[2]))
        return line_extensions

    #Identify a line is a series of count peaks along many adjacent lines
    # Hopefully, by requiring degree of continuity, we'll be able to reject things like cosmic rays.
    def identifyContinuousLines(self, pix_vals, lines_by_slice, data_to_search,
                                max_sep_per_pix = 5.0, max_frac_intensity_per_pix = 0.1, min_detections = 10,
                                fit_line_center_index = 1, image_range_to_search = None,
                                binning_search_width = 1, line_bg_width = 10, show_start_line_fits = 1):
        continuous_lines = []
        #First, trace out the lines only where they were detected
        #print ('lines_by_slice = ' + str(lines_by_slice))
        for i in range(len(lines_by_slice)):

            pix_val = pix_vals[i]
            lines_in_slice = lines_by_slice[i]
            for line in lines_in_slice:
                #line = line.tolist()
                if line[fit_line_center_index] >= 0:
                    matched_to_line = 0
                    for j in range(len(continuous_lines)):
                        continuous_line = continuous_lines[j]
                        if abs(continuous_line[-1][fit_line_center_index+1] - line[fit_line_center_index]) < max_sep_per_pix:
                           #print ('here1')
                            continuous_lines[j] = continuous_line + [[pix_val] + line]
                            matched_to_line = 1
                            break
                    if not(matched_to_line):
                        #print ('here2')
                        continuous_lines = continuous_lines + [[[pix_val] + line]]
        #print ('continuous_lines[0] = ' + str(continuous_lines[0]))

        if not(image_range_to_search is None) :
            range_below = list(range(image_range_to_search[0], min(image_range_to_search[1], pix_vals[0])))
            range_below.reverse()
            range_above = list(range(max(image_range_to_search[0], pix_vals[-1] + 1), image_range_to_search[1]))
            line_extensions_below = [[None for pix in range_below] for line in continuous_lines]
            line_extensions_above = [[None for pix in range_above] for line in continuous_lines]

            line_extensions_below = extendLinesIntoImage(range_below, line_extensions_below, data_to_search, [continuous_line[0] for continuous_line in continuous_lines],
                                                         binning_search_width = binning_search_width,
                                                         max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix, line_bg_width = line_bg_width )

            line_extensions_above = extendLinesIntoImage(range_above, line_extensions_above, data_to_search, [continuous_line[-1] for continuous_line in continuous_lines],
                                                         binning_search_width = binning_search_width,
                                                         max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix, line_bg_width = line_bg_width  )
            continuous_lines = [  can.niceReverse(line_extensions_below[i]) + continuous_lines[i] + line_extensions_above[i] for i in range(len(continuous_lines)) ]

        if show_start_line_fits:
            for continuous_line in continuous_lines:
                plt.plot([point[2] for point in continuous_line], [point[0] for point in continuous_line])
            plt.show()

        continuous_lines = [ line for line in continuous_lines if len(line) >= min_detections ]

        return continuous_lines


    def consolidateLines(self, line_indeces):
        min_n_sig_sep = self.min_sig_sep_for_distinct_lines
        line_widths = can.niceReverse(sorted(line_indeces.keys()))
        #print ('pre consolidation line_indeces: ' + str(line_indeces))
        #first identify lines of same widths:
        merged_lines = []
        for width in line_widths:
            all_ided_lines = line_indeces[width]
            lines_to_merge = all_ided_lines[:]
            #print ('lines_to_merge = ' + str(lines_to_merge))
            if len(lines_to_merge) > 1:
                still_lines_to_merge = 1
            else:
                still_lines_to_merge = 0
            prev_line_merged = 0
            while still_lines_to_merge:
                merged_lines = []
                still_lines_to_merge = 0
                for i in range(len(lines_to_merge) - 1):
                    #print('[lines_to_merge[i], lines_to_merge[i+1]] = ' + str([lines_to_merge[i], lines_to_merge[i+1]] ))
                    lines_too_close = (lines_to_merge[i+1][0] - lines_to_merge[i][0]) / width < min_n_sig_sep
                    if prev_line_merged:
                        prev_line_merged = 0
                    elif lines_too_close:
                        #print ('for line ' + str(i) + ' width ' + str(width) + ', need to merge lines ' + str([lines_to_merge[i],  lines_to_merge[i+1] ]))
                        #merged_lines = merged_lines + [(lines_to_merge[i][0] * width + lines_to_merge[i+1][0] * width) / (width + width)]
                        merged_lines = merged_lines + [[(lines_to_merge[i+1][0] if lines_to_merge[i+1][1] > lines_to_merge[i+1][0] else lines_to_merge[i][0]), (lines_to_merge[i+1][1] if lines_to_merge[i+1][1] > lines_to_merge[i+1][0] else lines_to_merge[i][1])]]
                        prev_line_merged = 1
                        still_lines_to_merge = 1
                    else:
                        merged_lines = merged_lines + [ lines_to_merge[i] ]
                if not(prev_line_merged):
                    merged_lines = merged_lines + [lines_to_merge[-1]]
                lines_to_merge = merged_lines[:]
                if len(lines_to_merge) == 1:
                    still_lines_to_merge = 0
                prev_line_merged = 0
            line_indeces[width] = merged_lines
        #print ('post consolidation line_indeces: ' + str(line_indeces))

        #Now merge similar lines and go down the list
        full_merged_lines = []
        for width in line_widths:
            all_ided_lines = line_indeces[width]
            lines_to_merge = all_ided_lines[:]
            still_lines_to_merge = 1
            prev_line_merged = 0
            for line_to_merge in lines_to_merge:
                line_too_close_to_existing_line = [abs(line_to_merge[0] - line[0]) / (np.sqrt(line[2] ** 2.0 + width ** 2.0) / 2.0) < min_n_sig_sep for line in full_merged_lines]
                #print('line_too_close_to_existing_line = ' + str(line_too_close_to_existing_line))
                if np.any(line_too_close_to_existing_line):
                    #print ('np.where(line_too_close_to_existing_line) = ' + str(np.where(line_too_close_to_existing_line)))
                    #print('np.where(line_too_close_to_existing_line).tolist() = ' + str(np.where(line_too_close_to_existing_line).tolist() ))
                    #[i for i in range(len(line_too_close_to_existing_line))]
                    #print ('line_to_merge, ' + str(line_to_merge) + ', too close to lines: ' + str( [i for i in range(len(line_too_close_to_existing_line))] ))
                    if line_to_merge[1] > full_merged_lines[ [i for i in range(len(line_too_close_to_existing_line))][0] ][1]:
                         #print ('[line_too_close_to_existing_line, line_to_merge, width] = ' + str([line_too_close_to_existing_line, line_to_merge, width]))
                         line_too_close_to_existing_line_indeces = [i for i in range(len(line_too_close_to_existing_line)) if line_too_close_to_existing_line[i]]
                         #print('np.where(line_too_close_to_existing_line) = ' + str(np.where(line_too_close_to_existing_line) ))
                         #print('line_too_close_to_existing_line_indeces = ' + str(line_too_close_to_existing_line_indeces))
                         #print('line_too_close_to_existing_line_indeces[0] = ' + str(line_too_close_to_existing_line_indeces[0]))
                         full_merged_lines[int(line_too_close_to_existing_line_indeces[0])] = line_to_merge + [width]
                    #else:
                    #    print ('Throwing out line ' + str(line_to_merge))
                else:
                    full_merged_lines = full_merged_lines + [line_to_merge + [width]]
        #print('full_merged_lines = ' + str(full_merged_lines))

        return full_merged_lines



    def detectLinesCentersInOneD(self, pixel_vals, spec_slice, stat_slice,
                                 spec_smoothing = 20, spec_grad_rounding = 5, n_std_for_line = 5,
                                 show = 0, ):
        spec_derivs = np.gradient(spec_slice)
        spec_derivs = np.around(spec_derivs, spec_grad_rounding)

        stat_derivs = np.gradient(stat_slice)
        stat_derivs = np.around(stat_derivs, spec_grad_rounding)
        deriv_median = np.median(stat_derivs)
        deriv_std = np.std(stat_derivs)

        #deriv_median = np.median(spec_derivs)
        #deriv_std = np.std(spec_derivs)
        print ('[len(pixel_vals), len(spec_slice), len(stat_slice), len(spec_derivs), len(stat_derivs)] = ' + str([len(pixel_vals), len(spec_slice), len(stat_slice), len(spec_derivs), len(stat_derivs)]) )
        deriv_emission_crossing_indeces = [i for i in range(len(pixel_vals[1:-1])) if (spec_derivs[i] >= 0.0 and spec_derivs[i + 1 ] < 0.0 )  ]
        deriv_absorbtion_crossing_indeces = [i for i in range(len(pixel_vals[1:-1])) if (spec_derivs[i] <= 0.0 and spec_derivs[i + 1 ] > 0.0 ) ]
        emission_deriv_turns = [[-1, -1] for cross in deriv_emission_crossing_indeces]
        absorbtion_deriv_turns = [[-1, -1] for cross in deriv_absorbtion_crossing_indeces]
        for j in range(len(deriv_emission_crossing_indeces)):
            cross = deriv_emission_crossing_indeces[j]
            left_peak = cross
            right_peak = cross+1
            while (left_peak > 0 and spec_derivs[left_peak] >= spec_derivs[left_peak+1]):
                #print ('left_peak = ' + str(left_peak))
                left_peak = left_peak - 1
            left_peak = left_peak + 1
            while (right_peak < len(spec_slice) and spec_derivs[right_peak] < spec_derivs[right_peak-1]):
                #print ('right_peak = ' + str(right_peak))
                right_peak = right_peak + 1
            right_peak = right_peak - 1
            emission_deriv_turns[j] = [left_peak, right_peak]

        for j in range(len(deriv_absorbtion_crossing_indeces)):
            cross = deriv_absorbtion_crossing_indeces[j]
            left_peak = cross
            right_peak = cross+1
            while (left_peak > 0 and spec_derivs[left_peak] <= spec_derivs[left_peak+1]):
                left_peak = left_peak - 1
            left_peak = left_peak + 1
            while (right_peak < len(spec_slice) and spec_derivs[right_peak] > spec_derivs[right_peak-1]):
                right_peak = right_peak + 1
            right_peak = right_peak - 1
            absorbtion_deriv_turns[j] = [left_peak, right_peak]

        absorbtion_indeces = [(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2 for j in range(len(deriv_absorbtion_crossing_indeces))
                                     if ( abs(spec_derivs[absorbtion_deriv_turns[j][0]] - deriv_median) / deriv_std >= n_std_for_line
                                          and abs(spec_derivs[absorbtion_deriv_turns[j][1]] - deriv_median) / deriv_std >= n_std_for_line )
                                    ]
        #absorbtion_indeces = [(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2 for j in range(len(deriv_absorbtion_crossings))
        #                             if ( abs(spec_derivs[absorbtion_deriv_turns[j][0]] - deriv_median) / deriv_std
        #                                  + abs(spec_derivs[absorbtion_deriv_turns[j][1]] - deriv_median) / deriv_std ) >= n_std_for_line * 2
        #                            ]
        emission_indeces = [(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2 for j in range(len(deriv_emission_crossing_indeces))
                                   if ( abs(spec_derivs[emission_deriv_turns[j][0]] - deriv_median) / deriv_std >= n_std_for_line
                                        and abs(spec_derivs[emission_deriv_turns[j][1]] - deriv_median) / deriv_std >= n_std_for_line )
                                  ]
        #emission_indeces = [(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2 for j in range(len(deriv_emission_crossings))
        #                           if ( abs(spec_derivs[emission_deriv_turns[j][0]] - deriv_median) / deriv_std
        #                               +  abs(spec_derivs[emission_deriv_turns[j][1]] - deriv_median) / deriv_std) >= n_std_for_line * 2
        #                          ]

        ref_spec = np.convolve(spec_slice, np.ones(spec_smoothing) / spec_smoothing, mode = 'same')
        gaussian_deriv = lambda xs, A, mu, sig: A * np.exp(-(xs - mu) ** 2.0 / (2.0 * sig ** 2.0)) * -2.0 * (xs - mu) / (2.0 * sig ** 2.0)
        #ref_spec_deriv0p5 = np.convolve(spec_slice, gaussian_deriv(np.linspace(-5.0, 5.0, 11), 1.0, 0.0, 0.5), mode = 'same')
        #ref_spec_deriv1 = np.convolve(spec_slice, gaussian_deriv(np.linspace(-5.0, 5.0, 11), 1.0, 0.0, 1.0), mode = 'same')
        #ref_spec_deriv1p5 = np.convolve(spec_slice, gaussian_deriv(np.linspace(-5.0, 5.0, 11), 1.0, 0.0, 1.5), mode = 'same')
        #ref_spec_deriv2 = np.convolve(spec_slice, gaussian_deriv(np.linspace(-5.0, 5.0, 11), 1.0, 0.0, 2.0), mode = 'same')
        convolve_widths = np.arange(1.0, 3.0, 1.0)
        emission_line_indeces = {width:[] for width in convolve_widths}
        absorbtion_line_indeces = {width:[] for width in convolve_widths}
        if show:
            plt.show()
            f, axarr = plt.subplots(len(convolve_widths)+ 1, 1, figsize = self.default_figsize)
            spec_deriv_line = axarr[0].plot(pixel_vals, spec_derivs, c = 'g')[0]
            bg_deriv_line = axarr[0].plot(pixel_vals, stat_derivs, c = 'r')[0]
            axarr[0].legend([spec_deriv_line, bg_deriv_line], ['Spec deriv, unsmoothed', 'bkg deriv, unsmoothed'])
            axarr[0].set_ylabel('Deriv spec (ADU / pix)')
        for i in range(len(convolve_widths)):
            width = convolve_widths[i]
            ref_spec_deriv = [np.sum(spec_derivs * gaussian_deriv(np.array(range(len(spec_slice))), 1.0, mu, width)) for mu in range(len(spec_slice))]
            bg_spec_deriv = [np.sum(stat_derivs * gaussian_deriv(np.array(range(len(spec_slice))), 1.0, mu, width)) for mu in range(len(spec_slice))]
            bg_spec_deriv_std = np.std(bg_spec_deriv[self.bg_std_buffer:-self.bg_std_buffer])
            bg_spec_deriv_std = can.sigClipStd(bg_spec_deriv[self.bg_std_buffer:-self.bg_std_buffer], sig_clip = self.bg_sig_clip)
            emission_line_indeces[width] = can.consolidateList([i  for i in range(len(ref_spec_deriv)) if ref_spec_deriv[i] / bg_spec_deriv_std >  n_std_for_line])
            emission_line_indeces[width] = [[i + pixel_vals[0], ref_spec_deriv[i] / bg_spec_deriv_std] for i in emission_line_indeces[width]]
            absorbtion_line_indeces[width] = can.consolidateList([i for i in range(len(ref_spec_deriv)) if ref_spec_deriv[i] / bg_spec_deriv_std < -n_std_for_line ])
            absorbtion_line_indeces[width] = [[i + pixel_vals[0], ref_spec_deriv[i] / bg_spec_deriv_std] for i in absorbtion_line_indeces[width]]
            if show:
                spec_deriv_line = axarr[i+1].plot(pixel_vals, ref_spec_deriv, c = 'g')[0]
                bg_deriv_line = axarr[i+1].plot(pixel_vals, bg_spec_deriv, c = 'r')[0]
                sigma_bounds = axarr[i+1].plot(pixel_vals, 0.0 * np.array(range(len(spec_slice))) + n_std_for_line * bg_spec_deriv_std, c = 'k')[0]
                axarr[i+1].plot(pixel_vals, 0.0 * np.array(range(len(spec_slice))) - n_std_for_line * bg_spec_deriv_std, c = 'k')[0]
                emission_plotted_lines = [ axarr[i+1].axvline(index[0], c = 'cyan') for index in emission_line_indeces[width] ]
                axarr[i+1].legend([spec_deriv_line, bg_deriv_line, sigma_bounds, emission_plotted_lines], ['Spec deriv, smoothed by ' + str(can.round_to_n(width, 2)), 'bkg deriv, smoothed by ' + str(can.round_to_n(width, 2)), str(self.bg_sig_clip) + r'$\sigma$-clipped bkg ' + str(can.round_to_n(n_std_for_line, 2)) + r'$\sigma$ bounds', 'Above threshold emission'])
                axarr[i+1].set_ylabel('Spec Deriv (ADU / pix)')
                if i == len(convolve_widths) - 1:
                    axarr[i+1].set_xlabel('Column number (pix)')
        if show:
            plt.show()
        merged_emission_line_indeces = self.consolidateLines(emission_line_indeces)
        merged_absorbtion_line_indeces = self.consolidateLines(absorbtion_line_indeces)
        #We have an issue where peaks and troughs can be
        ##########
        #FILL ME IN
        ##########
        """
        ref_spec_deriv1_mine = [np.sum(spec_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 1.0)) for mu in range(len(spec_slice))]
        bg_spec_deriv1_mine = [np.sum(stat_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 1.0)) for mu in range(len(spec_slice))]
        ref_spec_deriv3_mine = [np.sum(spec_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 3.0)) for mu in range(len(spec_slice))]
        bg_spec_deriv3_mine = [np.sum(stat_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 3.0)) for mu in range(len(spec_slice))]
        ref_spec_deriv5_mine = [np.sum(spec_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 5.0)) for mu in range(len(spec_slice))]
        bg_spec_deriv5_mine = [np.sum(stat_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 5.0)) for mu in range(len(spec_slice))]
        ref_spec_deriv20_mine = [np.sum(spec_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 20.0)) for mu in range(len(spec_slice))]
        bg_spec_deriv20_mine = [np.sum(stat_derivs * gaussian_deriv(np.array(pixel_vals), 1.0, mu, 20.0)) for mu in range(len(spec_slice))]
        """
        #ref_spec_deriv3 = np.convolve(spec_slice, gaussian_deriv(np.linspace(-5.0, 5.0, 11), 1.0, 0.0, 3.0), mode = 'same')
        """
        f, axarr = plt.subplots(5,1, figsize = self.default_figsize)
        axarr[0].plot(range(len(spec_slice)), spec_derivs, c = 'g')
        axarr[0].plot(range(len(spec_slice)), stat_derivs, c = 'r')
        axarr[0].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) + 20 * np.std(stat_derivs), c = 'k')
        axarr[0].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) - 20 * np.std(stat_derivs), c = 'k')
        #axarr[1].plot(range(len(spec_slice)), ref_spec_deriv0p5, c = 'purple')
        axarr[1].plot(range(len(spec_slice)), ref_spec_deriv1_mine, c = 'g')
        axarr[1].plot(range(len(spec_slice)), bg_spec_deriv1_mine, c = 'r')
        axarr[1].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) + 20 * np.std(bg_spec_deriv1_mine), c = 'k')
        axarr[1].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) - 20 * np.std(bg_spec_deriv1_mine), c = 'k')
        #axarr[1].plot(range(len(spec_slice)), ref_spec_deriv1p5, c = 'orange')
        axarr[2].plot(range(len(spec_slice)), ref_spec_deriv3_mine, c = 'g')
        axarr[2].plot(range(len(spec_slice)), bg_spec_deriv3_mine, c = 'r')
        axarr[2].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) + 20 * np.std(bg_spec_deriv3_mine), c = 'k')
        axarr[2].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) - 20 * np.std(bg_spec_deriv3_mine), c = 'k')
        axarr[3].plot(range(len(spec_slice)), ref_spec_deriv5_mine, c = 'g')
        axarr[3].plot(range(len(spec_slice)), bg_spec_deriv5_mine, c = 'r')
        axarr[3].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) + 20 * np.std(bg_spec_deriv5_mine), c = 'k')
        axarr[3].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) - 20 * np.std(bg_spec_deriv5_mine), c = 'k')
        axarr[4].plot(range(len(spec_slice)), ref_spec_deriv20_mine, c = 'g')
        axarr[4].plot(range(len(spec_slice)), bg_spec_deriv20_mine, c = 'r')
        axarr[4].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) + 20 * np.std(bg_spec_deriv20_mine), c = 'k')
        axarr[4].plot(range(len(spec_slice)), 0.0 * np.array(range(len(spec_slice))) - 20 * np.std(bg_spec_deriv20_mine), c = 'k')
        #axarr[1].plot(range(len(spec_slice)), ref_spec_deriv3, c = 'blue')
        #plt.plot(range(len(spec_slice)), spec_slice - ref_spec, c = 'c')
        plt.show()
        """

        if show:
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
            spec = axarr[0].plot(pixel_vals, spec_slice, c = 'blue')[0]
            axarr[0].set_ylim(np.min(spec_slice) * 0.9, np.max(spec_slice) * 1.1)
            bg = axarr[0].plot(pixel_vals, stat_slice, c = 'red')[0]
            em_line = None
            ab_line = None
            #for line in emission_indeces: em_line = axarr[0].axvline(line, color = 'green')
            #for line in can.flattenListOfLists([emission_line_indeces[width] for width in emission_line_indeces.keys()]): em_line = axarr[0].axvline(line[0], color = 'cyan', linestyle = 'dotted')
            #for line in absorbtion_indeces: ab_line = axarr[0].axvline(line, color = 'orange')
            for line in merged_absorbtion_line_indeces: ab_line = axarr[0].axvline(line[0], color = 'orange', linestyle = ':')
            for line in merged_emission_line_indeces: em_line = axarr[0].axvline(line[0], color = 'green', linestyle = ':')
            #for line in can.flattenListOfLists([absorbtion_line_indeces[width] for width in absorbtion_line_indeces.keys()]): ab_line = axarr[0].axvline(line[0], color = 'magenta', linestyle = 'dotted')
            #axarr[0].legend([spec, bg, em_line, ab_line], ['Spectrum binned by col', 'Background','Emission lines', 'absorbtion_lines'])
            if em_line is None:
                if ab_line is None:
                    axarr[0].legend([spec, bg,], ['Spectrum from binned columns', 'Background'])
                else:
                    axarr[0].legend([spec, bg, ab_line], ['Spectrum from binned columns', 'Background','Absorbtion lines'])
            else:
                if ab_line is None:
                    axarr[0].legend([spec, bg, em_line], ['Spectrum from binned columns', 'Background','Emission lines'])
                else:
                    axarr[0].legend([spec, bg, em_line, ab_line], ['Spectrum from binned columns', 'Background','Emission lines','Absorbtion lines'])

            axarr[0].legend([spec, bg, em_line, ab_line], ['Spectrum from binned columns', 'Background','Emission lines','Absorbtion lines'])
            spec_deriv = axarr[1].plot(pixel_vals, spec_derivs, c = 'blue')[0]
            bg_deriv = axarr[1].plot(pixel_vals, stat_derivs, c = 'red')[0]
            em_line = None
            ab_line = None
            for line in merged_emission_line_indeces: em_line = axarr[1].axvline(line[0], color = 'green', linestyle = ':')
            for line in merged_absorbtion_line_indeces: ab_line = axarr[1].axvline(line[0], color = 'orange', linestyle = ':')
            #axarr[1].legend([spec_deriv, bg_deriv, em_line, ab_line], ['Pixel deriv of binned spectrum', 'Pixel deriv of background','Emission lines', 'Absorbtion lines'])
            axarr[1].legend([spec_deriv, bg_deriv, em_line, ab_line], ['Pixel deriv of binned spectrum', 'Pixel deriv of background','Emission lines', 'Absorbtion lines'])
            axarr[0].set_title('Identified Lines in Binned Spectrum')
            axarr[0].set_xlabel('Column number (pix)')
            axarr[0].set_ylabel('Counts in column (ADU)')
            axarr[1].set_xlabel('Column number (pix)')
            axarr[1].set_ylabel('Pixel deriv of counts in column (ADU/pix)')
            #plt.tight_layout()
            plt.show()

        return sorted([line[0] for line in merged_absorbtion_line_indeces]), sorted([line[0] for line in merged_emission_line_indeces])
        #return sorted([line[0] + pixel_vals[0] for line in merged_absorbtion_line_indeces]), sorted([line[0] + pixel_vals[0] for line in merged_emission_line_indeces])

    def traceLinesOverRange(self, image, search_range,
                            n_std_for_line = 10, line_fit_width = None,
                            coarse_search_binning = 3, fit_binning = 3,
                            max_line_fit_width = 20, parallel_smoothing = 1,
                            n_pix_above_thresh = 1, width_guess = 3,
                            show_process = 0, spec_grad_rounding = 5,
                            draw_stats_below = 1, stat_region_buffer = 10,
                            fit_lines_with_pedestal = 1, fit_line_width_with_seeing_funct = 0 ):

        print ('[search_range, coarse_search_binning] = ' + str([search_range, coarse_search_binning]))
        coarse_pix_vals = list(range(int(search_range[0]), int(search_range[1]) , coarse_search_binning))
        coarse_fit_grid = {pix_val:[] for pix_val in coarse_pix_vals}
        if draw_stats_below:
            stat_region = [max(0, search_range[0] - stat_region_buffer - coarse_search_binning), search_range[0] - stat_region_buffer ]
        else:
            stat_region = [search_range[1] + stat_region_buffer,
                           min(np.shape(image)[self.spec_axis], search_range[1] + stat_region_buffer + coarse_search_binning) ]
        if self.spec_axis == 0:
            stat_slice = image[stat_region[0]:stat_region[1], :]
        else:
            stat_slice = image[:, stat_region[0]:stat_region[1]]
        if coarse_search_binning > 1:
            stat_slice = np.sum(stat_slice, axis = self.spec_axis)

        for i in range(len(coarse_pix_vals)):
            pix_val = coarse_pix_vals[i]
            print ('Identifying lines for orthogonal pixel values from ' + str(pix_val) + ' to ' + str(pix_val + coarse_search_binning))
            if self.spec_axis == 0:
                spec_slice = image[pix_val:pix_val + coarse_search_binning, :]
            else:
                spec_slice = image[:, pix_val:pix_val + coarse_search_binning]
            if coarse_search_binning > 1:
                spec_slice = np.sum(spec_slice, axis = self.spec_axis)

            #spec_slice = np.array(can.smoothList(spec_slice.tolist(), smooth_type = 'boxcar', params = [parallel_smoothing]))
            self.col_binned_spectrum = spec_slice
            self.col_binned_background = stat_slice
            strong_absorbtion_indeces, strong_emission_indeces = self.detectLinesCentersInOneD(list(range(len(spec_slice))) ,spec_slice, stat_slice,
                                                                                          spec_grad_rounding = spec_grad_rounding, n_std_for_line = n_std_for_line, show = show_process,
                                                                                            )
            coarse_fit_grid[pix_val] = strong_emission_indeces

        print ('coarse_fit_grid = ' + str(coarse_fit_grid))
        coarse_pixels = list(coarse_fit_grid.keys())
        pix_vals = list(range(int(search_range[0]), int(search_range[1] - fit_binning) + 1))
        all_slices = []
        for i in range(len(pix_vals)):
            pix_val = pix_vals[i]
            closest_coarse_pix = coarse_pixels[np.argmin([abs(coarse_pix - pix_val) for coarse_pix in coarse_pixels])]
            guess_line_centers = coarse_fit_grid[closest_coarse_pix]
            if self.spec_axis == 0:
                spec_slice = image[pix_val:pix_val + fit_binning, :]
            else:
                spec_slice = image[:, pix_val:pix_val + fit_binning]
            if fit_binning > 1:
                spec_slice = np.median(spec_slice, axis = self.spec_axis)

            line_fits = self.identifyLinesOnSlice(range(len(spec_slice)), spec_slice,
                                             max_line_fit_width = line_fit_width, peak_guesses = guess_line_centers,
                                             show_spec = (i % 20 == 0) * show_process, verbose =  (i % 20 == 0),
                                             fit_lines_with_pedestal = fit_lines_with_pedestal, fit_line_width_with_seeing_funct = fit_line_width_with_seeing_funct)

            print ('Found following line centers for pixel values from ' + str(pix_val) + ' to ' + str(pix_val + fit_binning) + ': ' + str((np.around([line_fit[1] for line_fit in line_fits],3) ).tolist()) )
            #line_fits = identifyLinesOnSlice(range(len(spec_slice)), spec_slice,
            #                                 std_thresh = n_std_for_line, max_line_fit_width = max_line_fit_width,
            #                                 n_pix_above_thresh = n_pix_above_thresh, init_fit_width_guess = width_guess,
            #                                 show_spec = (i % 50 == 51), verbose =  (i % 50 == 0))
            #if show_process:
            #    fit_funct = lambda xs, A, mu, sig, shift: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift
            #    for line_fit in line_fits:
            #        plt.plot(range(len(spec_slice)), fit_funct(range(len(spec_slice)), *(line_fit[0:-2])), c = 'r')
            #    #fit_spec = np.sum([fit_funct(range(len(spec_slice)), *(line_fit[0:-2])) for line_fit in line_fits], axis = 0)
            #    #plt.plot(range(len(spec_slice)), fit_spec)
            #    plt.show()
            all_slices = all_slices + [line_fits]

        return pix_vals, all_slices

    def polyFitVar(self, ind_vals, dep_vals, fit_order, n_std_for_rejection_in_fit):
        med = np.median(dep_vals)
        std = np.std(dep_vals)
        ind_vals_for_fit = [ind_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
        ind_vals_for_fit_range = [min(ind_vals_for_fit), max(ind_vals_for_fit)]
        dep_vals_to_fit = [dep_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
        var_poly_fit = np.polyfit(ind_vals_for_fit, dep_vals_to_fit, fit_order)
        #var_funct = lambda val: np.poly1d(var_poly_fit)(val) * (val >= ind_vals_for_fit_range[0]) * (val <= ind_vals_for_fit_range[1])
        var_funct = lambda val: np.poly1d(var_poly_fit)(val)

        return var_funct


    def getLineFunction(self, line,
                        n_std_for_rejection_in_fit = 3,
                        position_order = 2, A_of_x_order = 2, sig_of_x_order = 2,
                        n_hist_bins = 21 ):

        if self.spec_axis == 0:
            ys = [line_part[0] for line_part in line]
            xs = [line_part[2] for line_part in line]
            ind_var = ys
            dep_var = xs
        else:
            ys = [line_part[2] for line_part in line]
            xs = [line_part[0] for line_part in line]
            ind_var = xs
            dep_var = ys

        As = [line_part[1] for line_part in line]
        sigs = [line_part[3] for line_part in line]

        position_funct = self.polyFitVar(ind_var, dep_var, position_order, n_std_for_rejection_in_fit)
        A_funct = self.polyFitVar(ind_var, As, position_order, n_std_for_rejection_in_fit)
        sig_funct = self.polyFitVar(ind_var, sigs, position_order, n_std_for_rejection_in_fit)

        return [ind_var, position_funct, A_funct, sig_funct]

    def fixContinuousLines(self, lines, line_profiles, n_bins = 21, n_empty_bins_to_drop = 2, n_hist_bins = 21, show_line_matches = 1):
        new_lines = [[] for line in lines]
        for i in range(len(lines)):
            line = lines[i]
            line_profile = line_profiles[i]
            #First, determine if a line is getting smeared by having neighbors, and fix it if so
            pixels = [line_slice[0] for line_slice in line]
            centers = [line_slice[2] for line_slice in line]
            fit_centers = [line_profile[1](pix) for pix in pixels]
            center_offsets = [fit_centers[j] - centers[j] for j in range(len(centers))]
            #axarr[0,0].plot(strong_line_profile[0], [strong_line_profile[1](x) for x in strong_line_profile[0]])
            if show_line_matches:
                f, axarr = plt.subplots(3,1, figsize = self.default_figsize)
                bin_peaks, bin_edges, blanck = axarr[0].hist(center_offsets, bins = n_hist_bins)
                plt.close('all')
            bin_centers = [ (bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(bin_edges) - 1) ]
            peak_index = np.argmax(bin_peaks)
            center_bounds = [bin_edges[0], bin_edges[-1]]
            for j in can.niceReverse(list(range(0, peak_index))):
                if bin_peaks[j] == 0:
                    center_bounds[0] = bin_edges[j]
                    break
            for j in list(range(peak_index, len(bin_peaks))):
                if bin_peaks[j] == 0:
                    center_bounds[1] = bin_edges[j+1]
                    break
            new_line = [ line_slice for line_slice in line if (line_profile[1](line_slice[0]) - line_slice[2] >= center_bounds[0] and line_profile[1](line_slice[0]) - line_slice[2] <= center_bounds[1]) ]
            new_lines[i] = new_line

        return new_lines


    def detectLinesInImage(self, current_image, spec_range,
                           n_std_for_lines = 10.0, line_fit_width = None,
                           search_binning = 'full', fit_binning = 10,
                           max_line_fit_width = 20, parallel_smoothing = 1,
                           width_guess = 1, show_process = 0,
                           max_sep_per_pix = 5.0, max_frac_intensity_per_pix = 0.1,
                           min_detections = 10, fit_line_center_index = 1,
                           search_for_extensions = None, bg_fit_width = 10,
                           draw_stats_below = 1, buffer_for_line_background_stats = 10):
        if search_binning in ['full','FULL','Full']:
            search_binning = spec_range[1] - spec_range[0]

        pix_slices, lines_by_slice = self.traceLinesOverRange(current_image, spec_range,
                                                         n_std_for_line = n_std_for_lines, line_fit_width = line_fit_width,
                                                         coarse_search_binning = search_binning, fit_binning = fit_binning,
                                                         max_line_fit_width = max_line_fit_width, parallel_smoothing = parallel_smoothing,
                                                         n_pix_above_thresh = 1, width_guess = width_guess,
                                                         show_process = show_process,
                                                         draw_stats_below = draw_stats_below, stat_region_buffer = buffer_for_line_background_stats)

        lines = self.identifyContinuousLines(pix_slices, lines_by_slice, current_image,
                                        max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix,
                                        min_detections = min_detections, fit_line_center_index = fit_line_center_index,
                                        image_range_to_search = search_for_extensions, binning_search_width = search_binning,
                                        line_bg_width = bg_fit_width, show_start_line_fits = show_process)

        return lines

    def readInDataTextFile(self, ref_spec_file,
                           spec_file_dir = '', n_ignore = 0,
                           throughput_file = 'default_throughput.txt'):
        ref_spec = can.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore, convert_to_float = 1)
        #plt.plot(*ref_spec)
        #plt.show()

        return ref_spec

    #We want to return the integrated distance (in pixel space) between detected lines
    # and their wavelengths that they might correspond to, given the wavelength solution.
    # We cannot have a single wavelength match two detections (or vice versa) so we perform the
    # matching one line at a time, and remove the match from consideration after each match is done.
    def lineMatchingFunction(self, line_pixels, line_wavelengths, n_matches, wavelength_solution, verbose = 0):
        #We assume that line_wavelengths are sorted in the order in which they should be matched.
        # We don't perform the matching here, to minimize computation time
        measured_waves = [wavelength_solution(pix) for pix in line_pixels]
        if len(measured_waves) < len(line_wavelengths):
            fixed_waves = measured_waves
            sliding_waves = line_wavelengths
        else:
            fixed_waves = line_wavelengths
            sliding_waves = measured_waves
        #if verbose: print ('[fixed_waves, sliding_waves] = ' + str([fixed_waves, sliding_waves]))
        remaining_sliding_matches = sliding_waves.copy()
        total_sep = 0.0
        #remaining_pixels = line_pixels.copy()
        for i in range(n_matches):
            ref_fixed = fixed_waves[i]
            seps_sqr = (remaining_sliding_matches - ref_fixed) ** 2.0
            min_index = np.argmin(seps_sqr)
            min_sep = seps_sqr[min_index]
            total_sep = total_sep + min_sep
            remaining_sliding_matches[i] = np.inf
            if verbose:
                print ('total_sep = ' + str(total_sep))
        if verbose: print ('total_sep = ' + str(total_sep))
        return np.log10(total_sep)

    def matchLinesRecursive(self, line_pixels_to_be_matched, line_wavelengths_to_match, mu_of_wavelength_funct,
                           max_sep_pix = np.inf, best_matches = [], print_params = None):
        #print ('[line_pixels_to_be_matched, line_wavelengths_to_match, best_matches] = ' + str([line_pixels_to_be_matched, line_wavelengths_to_match, best_matches] ))
        if len(line_pixels_to_be_matched) == 0 or len(line_wavelengths_to_match) == 0:
            res = np.sum([match[-1] for match in best_matches])
            if not(print_params is None):
                print('[params, res] = ' + str([print_params, res] ))
            return best_matches
        #print ('[line_pixels_to_be_matched, line_wavelengths_to_match] = ' + str([line_pixels_to_be_matched, line_wavelengths_to_match]))
        #print ('[line_pixels_to_be_matched.tolist(), line_wavelengths_to_match.tolist()] = ' + str([line_pixels_to_be_matched.tolist(), line_wavelengths_to_match.tolist()]))
        line_matches = []
        line_pixel_match_indeces = []
        line_wavelength_match_indeces = []
        min_seps = []
        min_sep_indeces = []
        #First, for each line identify the closest matches...
        for i in range(len(line_pixels_to_be_matched)):
            line_pixel_to_be_matched = line_pixels_to_be_matched[i]
            line_pixels_to_match = mu_of_wavelength_funct(line_wavelengths_to_match)
            line_seps = [(line_pixel_to_match - line_pixel_to_be_matched) ** 2.0 for line_pixel_to_match in line_pixels_to_match]
            #print ('line_seps = ' + str(line_seps))
            min_sep_index = np.argmin(line_seps)
            min_sep = line_seps[min_sep_index]
            min_seps = min_seps + [min_sep]
            min_sep_indeces = min_sep_indeces + [min_sep_index]
        #print ('[min_seps, min_sep_indeces] = ' + str([min_seps, min_sep_indeces] ))
        #Then, keep only the closest N lines, where N is either the number of reference lines or the number of lines identified in the spectrum, whichever is smaller
        best_match_index = np.argmin(min_seps)
        best_matches = best_matches + [[line_pixels_to_be_matched[best_match_index], line_wavelengths_to_match[min_sep_indeces[best_match_index]], min_seps[best_match_index]]]
        #return matchLines(np.array(can.removeListElement(line_pixels_to_be_matched.tolist(), best_match_index)), np.array(can.removeListElement(line_wavelengths_to_match.tolist(), min_sep_indeces[best_match_index])), mu_of_wavelength_funct, max_sep_pix = np.inf, best_matches = best_matches)
        return self.matchLinesRecursive(line_pixels_to_be_matched, np.array(can.removeListElement(line_wavelengths_to_match.tolist(), min_sep_indeces[best_match_index])), mu_of_wavelength_funct, max_sep_pix = np.inf, best_matches = best_matches, print_params = print_params)



    #Match, lines by line, recursively
    def matchLines(self, line_pixels_to_be_matched, line_wavelengths_to_match, mu_of_wavelength_funct,
                   max_sep_pix = np.inf):
        print ('[line_pixels_to_be_matched, line_wavelengths_to_match] = ' + str([line_pixels_to_be_matched, line_wavelengths_to_match]))
        #print ('[line_pixels_to_be_matched.tolist(), line_wavelengths_to_match.tolist()] = ' + str([line_pixels_to_be_matched.tolist(), line_wavelengths_to_match.tolist()]))
        line_matches = []
        line_pixel_match_indeces = []
        line_wavelength_match_indeces = []
        #First, for each line identify the closest matches...
        for i in range(len(line_pixels_to_be_matched)):
            line_pixel_to_be_matched = line_pixels_to_be_matched[i]
            line_pixels_to_match = mu_of_wavelength_funct(line_wavelengths_to_match)
            line_seps = [abs(line_pixel_to_match - line_pixel_to_be_matched) for line_pixel_to_match in line_pixels_to_match]
            min_sep_index = np.argmin(line_seps)
            min_sep = line_seps[min_sep_index]
            if min_sep <= max_sep_pix:
                line_matches = line_matches + [[line_pixel_to_be_matched, line_wavelengths_to_match[min_sep_index]]]
                line_pixel_match_indeces = line_pixel_match_indeces + [i]
                line_wavelength_match_indeces = line_wavelength_match_indeces + [min_sep_index]
        #Then, keep only the closest N lines, where N is either the number of reference lines or the number of lines identified in the spectrum, whichever is smaller
        n_lines_to_keep = min(len(line_pixels_to_be_matched), len(line_wavelengths_to_match,))
        line_seps = [(line_match[0] - mu_of_wavelength_funct(line_match[1])) ** 2.0 for line_match in line_matches]
        line_seps, line_pixel_match_indeces, line_wavelength_match_indeces, line_matches = can.safeSortOneListByAnother(line_seps, [line_seps, line_pixel_match_indeces, line_wavelength_match_indeces, line_matches])
        line_sep = np.sum(sorted(line_seps)[0:n_lines_to_keep])
        line_pixel_match_indeces = line_pixel_match_indeces[0:n_lines_to_keep]
        line_wavelength_match_indeces = line_wavelength_match_indeces[0:n_lines_to_keep]
        line_matches = line_matches[0:n_lines_to_keep]
        print ('line_matches = ' + str(line_matches))
        print ('line_sep = ' + str(line_sep))
        return [line_pixel_match_indeces, line_wavelength_match_indeces, line_matches, line_sep]

    def readFileIntoInterp(self, target_file, target_dir, n_ignore, convert_to_float = 1):
        cols = can.readInColumnsToList(target_file, file_dir = target_dir, n_ignore = n_ignore, convert_to_float = convert_to_float)
        interp = can.safeInterp1d(*cols)
        return interp

    def computeGoodnessOfFitSpectrumModel(self, fit_params, extra_args = [0, None] ):
        show_fit , gauss_smooth_pix = extra_args
        fitted_pixels = np.poly1d(fit_params[1:])(self.ref_spec_lines[0])
        fitted_pixel_interp = interpolate.interp1d(fitted_pixels, self.ref_spec_lines[1], bounds_error = False, fill_value = 0.0)

        expected_spectrum = np.sum([fitted_pixel_interp(pix_part) * 1.0 / np.sqrt(2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0) * np.exp(-(np.array(self.parallel_spec_pixels) - pix_part) ** 2.0 / (2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0)) for pix_part in np.arange(self.parallel_spec_pixels[0], self.parallel_spec_pixels[-1], 0.1)], axis = 0) #]
        if gauss_smooth_pix != None:
            ref_spec = scipy.ndimage.filters.gaussian_filter1d(self.full_pix_spectrum, gauss_smooth_pix )
            expected_spectrum = scipy.ndimage.filters.gaussian_filter1d(expected_spectrum, gauss_smooth_pix, )
        else:
            ref_spec = self.full_pix_spectrum[:]
        max_unscaled_spectrum = np.max(expected_spectrum)
        max_ref_spec = np.max(ref_spec)
        if abs(max_unscaled_spectrum) > 0.0:
            expected_spectrum = expected_spectrum / np.max(expected_spectrum) * max_ref_spec  * fit_params[0]

        if show_fit:
            f, axarr = plt.subplots(3,1, figsize = self.default_figsize)
            axarr[0].plot(self.parallel_spec_pixels, fitted_pixel_interp(self.parallel_spec_pixels) )
            axarr[1].plot(self.parallel_spec_pixels, expected_spectrum)
            axarr[2].plot(self.parallel_spec_pixels, self.full_pix_spectrum)
            plt.show()
        diffs = np.sqrt((expected_spectrum - ref_spec) ** 2.0)
        if abs(max_unscaled_spectrum) == 0.0:
            diff = np.inf
        else:
            diff = np.sum(diffs)
        print ('[fit_params, diff] = ' + str([fit_params, diff]))
        f, axarr = plt.subplots(3,1, figsize = self.default_figsize)
        axarr[0].plot(self.parallel_spec_pixels, expected_spectrum, c = 'blue')
        axarr[1].plot(self.parallel_spec_pixels, ref_spec, c = 'red')
        axarr[2].plot(self.parallel_spec_pixels, diffs, c = 'purple')
        plt.draw()
        plt.pause(0.1)
        plt.close('all')
        return diff


    #Curve fitting doesn't work very well.  Is there a way that we could just detect where lines are and determine where they are supposed to be?
    def determineWavelengthSolution(self, line_solutions, spec_range, ref_spec_file, ref_lines_file, #line_median_areas,
                                    spec_file_dir = '', throughput_file = 'default_throughput.txt',
                                    n_ignore_spec = 0, n_ignore_lines = 0, n_ignore_throughput = 0,
                                    wavelength_solution_drift_order = 2,
                                    coarse_search_param_range= [[-1000.0, -100.0], [1.0, 1.5]], coarse_search_param_step = [51, 51],
                                    solution_save_dir = '', save_solution_image = 1,
                                    save_solution = 1, show_solution = 1, init_guess = [1.0, 0.0, 1.5, -500.0] ):
        coarse_search_param_range = self.ref_param_holder.getCoarseSearchParamRange()
        coarse_search_param_step = self.ref_param_holder.getCoarseSearchNParamStep()
        print('[coarse_search_param_range, coarse_search_param_step] = ' + str([coarse_search_param_range, coarse_search_param_step] ))
        print('[ref_spec_file, ref_lines_file] = ' + str([ref_spec_file, ref_lines_file]))
        self.ref_spec_lines = can.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore_spec, convert_to_float = 1, delimiter = ',')
        self.ref_spec_just_lines = can.readInColumnsToList(ref_lines_file, file_dir = spec_file_dir, n_ignore = n_ignore_lines, convert_to_float = 1, delimiter = ',')

        throughput_interp = self.readFileIntoInterp(throughput_file, spec_file_dir, n_ignore_throughput, convert_to_float = 1)
        self.ref_spec_lines[1] = [self.ref_spec_lines[1][0] / (((self.ref_spec_lines[0][1] + self.ref_spec_lines[0][0]) / 2.0 - self.ref_spec_lines[0][0]) * 2.0)] + [self.ref_spec_lines[1][i] / (-(self.ref_spec_lines[0][i] + self.ref_spec_lines[0][i-1]) / 2.0 + (self.ref_spec_lines[0][i+1] + self.ref_spec_lines[0][i]) / 2.0) for i in range(1, len(self.ref_spec_lines[0])-1)] + [self.ref_spec_lines[1][-1] / (((self.ref_spec_lines[0][-1] + self.ref_spec_lines[0][-2]) / 2.0 - self.ref_spec_lines[0][-2]) * 2.0)]
        self.ref_spec_lines[1] = (np.array(self.ref_spec_lines[1]) * throughput_interp(self.ref_spec_lines[0])).tolist()

        #f, axarr = plt.subplots(2,1)
        #axarr[0].plot(self.ref_spec_lines[0], self.ref_spec_lines[1])
        #axarr[0].set_xlabel('wavelength (nm)')
        #axarr[1].plot(self.parallel_spec_pixels, self.full_pix_spectrum)
        #axarr[1].set_xlabel('pix num')

        #init_guess = [1.0, self.approx_pix_scale, -500.0]
        #rough_wavelength_fit = optimize.minimize(self.computeGoodnessOfFitSpectrumModel, init_guess, args = [0, self.crude_fit_gauss] )
        #print ('rough_wavelength_fit = ' + str(rough_wavelength_fit))
        #init_guess = [rough_wavelength_fit['x'].tolist()[0]] + [0.0] + rough_wavelength_fit['x'].tolist()[1:]
        #wavelength_fit = optimize.minimize(self.computeGoodnessOfFitSpectrumModel, init_guess, args = [0, None] )
        #print ('wavelength_fit = ' + str(wavelength_fit))
        slice_pixels = range(*spec_range)
        fitted_spectra = [[0.0 for pix in slice_pixels] for guess in range(self.wavelength_solution_order + 1)]
        best_match_params1 = [0.0, 0.0]
        best_match_params = [0.0, 0.0]
        best_matches = []
        best_match_sep = np.inf
        best_match_pixel_indeces = []
        best_match_wavelength_indeces = []

        wavelength_of_mu = lambda mu, lam0, lam1, lam2: lam0 + lam1 * mu + lam2 * mu ** 2.0
        wavelength_of_mu = lambda mu, lam0, lam1: lam0 + lam1 * mu

        #median_line_centers = [np.median([line_solution[1](pix) for pix in slice_pixels]) for line_solution in line_solutions]
        median_line_centers = [line_solution[1] for line_solution in line_solutions]
        print ('median_line_centers = ' + str(median_line_centers))

        #there is one light that I expect to be the brightest.  We use that to anchor the fit.
        #peak_area_index = np.argmax(line_median_areas)
        #peak_line_center = median_line_centers[peak_area_index]
        #peak_wavelength = 882.97
        #print ('peak_line_center = ' + str(peak_line_center))
        #print ('peak_wavelength = ' + str(peak_wavelength))
        #print('len(line_solutions) = ' + str(len(line_solutions)))
        #print('len(line_median_areas) = ' + str(len(line_median_areas)))
        #mu_of_wavelength = lambda lam, mu0, mu1: mu1 * (lam - peak_wavelength) + peak_line_center
        #for test_mu1 in np.linspace(*(coarse_search_param_range[1]), coarse_search_param_step[1]):
        #    matched_line_pixel_indeces, matched_line_wavelength_indeces, matched_lines, matched_lines_sep = matchLines(np.array(median_line_centers), np.array(ref_spec_lines[0]), lambda waves: mu_of_wavelength(waves, 0.0, test_mu1) )
        #    if (len(matched_lines) > len(best_matches)) or (len(matched_lines) == len(best_matches) and matched_lines_sep < best_match_sep):
        #        best_match_params1 = [0.0, test_mu1]
        #        best_matches = matched_lines
        #        best_match_sep = matched_lines_sep
        #        best_match_pixel_indeces = matched_line_pixel_indeces
        #        best_match_wavelength_indeces = matched_line_wavelength_indeces
        #print ('best_match_params1 = ' + str(best_match_params1))
        #plt.scatter(np.array(ref_spec_lines[0]), [mu_of_wavelength(line, *best_match_params1) for line in ref_spec_lines[0]])
        #plt.show()
        #axarr[1].scatter(ref_spec_lines[0], [0 for line in ref_spec_lines[0]], c = 'b')
        #axarr[1].scatter([ref_spec_lines[0][index] for index in best_match_wavelength_indeces], [0 for index in best_match_wavelength_indeces], c = 'c')
        #[ axarr[1].annotate(i, (ref_spec_lines[0][best_match_wavelength_indeces[i]],0) ) for i in range(len(best_match_wavelength_indeces)) ]

        wavelength_of_mu = lambda mu, lam0, lam1, lam2: lam0 / self.wavelength_scaling[0] + lam1 / self.wavelength_scaling[1] * mu + lam2 / self.wavelength_scaling[2] * mu ** 2.0
        mu_of_wavelength = lambda lam, mu0, mu1, mu2: mu0 / self.wavelength_scaling[0] + mu1 / self.wavelength_scaling[1] * lam + mu2 / self.wavelength_scaling[2] * lam ** 2.0
        #wavelength_of_mu = lambda mu, lam0, lam1: lam0 / self.wavelength_scaling[0] + lam1 / self.wavelength_scaling[1] * mu
        #mu_of_wavelength = lambda lam, mu0, mu1: mu0 / self.wavelength_scaling[0] + mu1 / self.wavelength_scaling[1] * lam
        print ('[coarse_search_param_range, coarse_search_param_step] = ' + str([coarse_search_param_range, coarse_search_param_step]))
        best_match_res = np.inf
        best_matches = []
        for test_mu0 in np.linspace(*(coarse_search_param_range[0]), coarse_search_param_step[0]):
            for test_mu1 in np.linspace(*(coarse_search_param_range[1]), coarse_search_param_step[1]):
                #print ('[test_lam0, test_lam1] = ' + str([test_lam0, test_lam1]))
                #matched_line_pixel_indeces, matched_line_wavelength_indeces, matched_lines, matched_lines_sep = self.matchLines(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, test_mu0, test_mu1, 0.0) )
                matched_lines = self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, test_mu0, test_mu1, 0.0), print_params = None)
                match_res = np.sum([match[-1] for match in matched_lines])
                if match_res < best_match_res:
                    best_match_res = match_res
                    best_matches = matched_lines
                    best_match_params = [test_mu0, test_mu1]
                #if (len(matched_lines) > len(best_matches)) or (len(matched_lines) == len(best_matches) and matched_lines_sep < best_match_sep):
                #    best_match_params = [test_mu0, test_mu1]
                #    best_matches = matched_lines
                #    best_match_sep = matched_lines_sep
                #    best_match_pixel_indeces = matched_line_pixel_indeces
                #    best_match_wavelength_indeces = matched_line_wavelength_indeces

        #print ('[best_match_pixel_indeces, best_match_wavelength_indeces, best_match_params, best_matches, best_match_sep] = ' + str([best_match_pixel_indeces, best_match_wavelength_indeces, best_match_params, best_matches, best_match_sep]))
        #print('[median_line_centers, self.ref_spec_just_lines[0]] = ' + str([median_line_centers, self.ref_spec_just_lines[0]]))
        #best_match_params = [best_match_params[0] + 1.0, best_match_params[1] + 1.0]
        #best_match_params = [-569.3521584 ,    1.21126767]
        print ('[np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), best_match_params] = ' + str([np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), best_match_params]))
        best_match_linear = optimize.minimize(lambda test_params: np.sum([match[-1] for match in self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, *(test_params.tolist() + [0.0])), print_params = None )]), best_match_params  , method = 'Nelder-Mead')

        print ('linear best match params: ' + str(best_match_params))
        print ('quadratic best match params: ' + str(best_match_linear))
        best_match_quadratic = optimize.minimize(lambda test_params: np.sum([match[-1] for match in self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, *test_params), print_params = None )]), best_match_linear['x'].tolist() + [0.0]  , method = 'Nelder-Mead')
        best_match_quadratic['x'] = np.array([ best_match_quadratic['x'][i] / self.wavelength_scaling[i] for i in range(len(best_match_quadratic['x'])) ])
        print ('quadratic best match params: ' + str(best_match_quadratic))
        best_match = best_match_linear['x'].tolist() + [0.0]
        print ('best_match = ' + str(best_match))
        best_fit_terms = can.niceReverse(best_match[:])
        print ('best_match = ' + str(best_match))
        best_match_lines = self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, *best_match), print_params = best_match )
        print ('best_match_lines = ' + str(best_match_lines))
        print ('len(best_match_lines) = ' + str(len(best_match_lines)))
        best_match_pixels = [match[0] for match in best_match_lines]
        best_match_wavelengths = [match[1] for match in best_match_lines]
        print ('[best_match_pixels, best_match_wavelengths] = ' + str([best_match_pixels, best_match_wavelengths]))
        #spec_pixels = np.arange(*line_range)
        #for i in range(spec_range[1] - spec_range[0]):
        #    pix = slice_pixels[i]
        #    #line_centers = [line_solutions[index][1](pix) for index in best_match_pixel_indeces]
        #    line_centers = [line_solution[1] for line_solution in line_solutions]
        #    #print ('matched_lines = ' + str(matched_lines))

        #    wavelength_solution = np.polyfit([match[1] for match in best_matches], [line_centers[index] for index in best_match_pixel_indeces], wavelength_solution_order)
        #    for pix_order in range(wavelength_solution_order + 1):
        #        fitted_spectra[pix_order][i] = wavelength_solution[pix_order]
        #    wavelength_funct = np.poly1d(wavelength_solution)

        #print ('wavelength_solution = ' + str(wavelength_solution))
        #wavelength_solution_functions = [[fitted_spectra[i][0]]]quad_fit_terms
        mu_of_wavelength_solution, wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(best_fit_terms)
        print ('[show_solution, save_solution_image] = ' + str([show_solution, save_solution_image] ))
        if show_solution or save_solution_image:
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
            fitted_pixels = np.poly1d(best_fit_terms)(self.ref_spec_lines[0])
            fitted_pixel_interp = interpolate.interp1d(fitted_pixels, self.ref_spec_lines[1], bounds_error = False, fill_value = 0.0)
            expected_spectrum = np.sum([np.array(fitted_pixel_interp(pix_part)) * 1.0 / np.sqrt(2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0) * np.exp(-(np.array(self.parallel_ref_spec_pixels) - pix_part) ** 2.0 / (2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0)) for pix_part in np.arange(self.parallel_ref_spec_pixels[0], self.parallel_ref_spec_pixels[-1], 0.1)], axis = 0) #]
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
            #axarr[0].plot(self.parallel_spec_pixels, fitted_pixel_interp(self.parallel_spec_pixels) )
            parallel_ref_spec_wavelengths = [wavelength_of_mu_solution(pix) for pix in self.parallel_ref_spec_pixels]
            print ('[len(self.parallel_ref_spec_pixels), len(parallel_ref_spec_wavelengths), len(expected_spectrum)] = ' + str([len(self.parallel_ref_spec_pixels),  len(parallel_ref_spec_wavelengths), len(expected_spectrum)]))
            spec_plot = axarr[0].plot(parallel_ref_spec_wavelengths, np.array(expected_spectrum) / np.max(expected_spectrum))[0]
            print ('[best_match_pixel_indeces, median_line_centers] = ' + str([best_match_pixel_indeces, median_line_centers]))
            print ('[median_line_centers[index] for index in best_match_pixel_indeces] = ' + str([median_line_centers[index] for index in best_match_pixel_indeces]))
            print ('[wavelength_of_mu_solution(median_line_centers[index]) for index in best_match_pixel_indeces] = ' + str([wavelength_of_mu_solution(median_line_centers[index]) for index in best_match_pixel_indeces]))
            #ref_emission_vlines = [axarr[0].axvline(line, c = 'orange') for line in self.ref_spec_just_lines[0]][0]
            ref_emission_vlines = [axarr[0].axvline(wave, c = 'red') for wave in best_match_wavelengths][0]
            matched_emission_vlines = [axarr[0].axvline(wavelength_of_mu_solution(pix), c = 'cyan', linestyle = 'dotted') for pix in best_match_pixels][0]

            axarr[0].legend([spec_plot, ref_emission_vlines, matched_emission_vlines], ['Reference spectrum convolved with seeing', 'Reference lines', 'Matched spectrum lines'])
            axarr[0].set_ylabel('Relative expected line intensity')
            spec_plot = axarr[1].plot(parallel_ref_spec_wavelengths, self.full_ref_pix_spectrum)[0]
            all_emission_vlines = [axarr[1].axvline(wavelength_of_mu_solution(strong_centroid), c = 'orange') for strong_centroid in median_line_centers][0]
            #print ('median_line_centers = ' + str(median_line_centers))
            matched_emission_vlines = [axarr[1].axvline(wavelength_of_mu_solution(pix), c = 'cyan') for pix in best_match_pixels][0]
            ref_emission_vlines = [axarr[1].axvline(wave, c = 'red', linestyle = 'dotted') for wave in best_match_wavelengths][0]
            axarr[1].legend([spec_plot, matched_emission_vlines, all_emission_vlines, ref_emission_vlines ], ['Full measured spectrum', 'IDed and matched emission lines', 'IDed and unmatched emission lines', 'Matched reference lines'])
            axarr[1].set_xlabel('Calibration wavelength (nm)')
            axarr[1].set_ylabel('Total counts in spec (ADU)')
            plt.tight_layout()
        if save_solution_image:
            print ('Saving line matching image to file ' + self.target_dir + 'SpecSolution.pdf')
            plt.savefig(self.target_dir + 'SpecSolution.pdf')
            if not(show_solution):
                plt.close('all')
        if show_solution:
            plt.show()
        #if show_solution:
        #    f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
        #    axarr[0].scatter(slice_pixels, fitted_spectra[0])
        #    axarr[1].scatter(slice_pixels, fitted_spectra[1])
        #    axarr[0].set_title('Wavelength solution')
        #    plt.show()

        #if save_solution:
        #    np.save(self.archival_data_dir + wavelength_solution_file,  wavelength_solution)
        if save_solution:
            print ('best_fit_terms = ' + str(best_fit_terms))
            can.saveListToFile([self.anchor_parallel_pix] + best_fit_terms, self.target_dir + self.ref_spec_solution_file)

        self.wavelength_poly_terms = best_fit_terms[:]
        mu_of_wavelength_solution, wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(best_fit_terms)

        return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    def create2DWavelengthSolutionCallableFunctions(self, wavelength_polyfit):

        #fitted_wavelength_solution = [np.poly1d(fit_to_solution_term) for fit_to_solution_term in wavelength_polyfit]
        mu_of_wavelength_solution = lambda lam, y: np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution])(lam)
        wavelength_of_mu_solution = lambda mu, y: ((np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution]) - mu).roots)[0]

        return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    #def loadWavelengthSolution(solution_file, load_dir = ''):
    #    fitted_wavelength_solution = np.load(archival_data_dir + solution_file)
    #    mu_of_wavelength_solution = lambda lam, y: np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution])(lam)
    #    wavelength_of_mu_solution = lambda mu, y: ((np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution]) - mu).roots)[0]
    #
    #    return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    def createWavelengthSolutionCallableFunctions(self, wavelength_poly_terms):
        print ('wavelength_poly_terms = ' + str(wavelength_poly_terms))
        mu_of_wavelength_solution = lambda lam: np.poly1d(wavelength_poly_terms)(lam)
        if self.wavelength_solution_order > 1:
            wavelength_of_mu_solution = lambda mu: ((np.poly1d(wavelength_poly_terms) - mu).roots)[-1]
        else:
            wavelength_of_mu_solution = lambda mu: ((np.poly1d(wavelength_poly_terms) - mu).roots)
        return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    def determineMacroLineCurvatureDict(self, spec_range, strong_line_profiles, pix_index = 0, mu_index = 1, curvature_fit_order = 2, anchor_parallel_pix = None):
        if anchor_parallel_pix == None:
            anchor_parallel_pix = self.anchor_parallel_pix
        print ('anchor_parallel_pix = ' + str(anchor_parallel_pix))
        anchor_line_perp_pixels = [strong_line_profile[mu_index](anchor_parallel_pix) for strong_line_profile in strong_line_profiles]
        line_anchor_funct_dict = {}
        for spec_parallel_pix in range(spec_range[0], spec_range[1]):
            line_perp_pixels = [strong_line_profile[mu_index](spec_parallel_pix) for strong_line_profile in strong_line_profiles]
            undo_curve_fit = np.polyfit(line_perp_pixels, anchor_line_perp_pixels, curvature_fit_order)
            undo_curve_funct = np.poly1d(undo_curve_fit)
            line_anchor_funct_dict[spec_parallel_pix] = undo_curve_funct

        return line_anchor_funct_dict


    #line_profile_dict takes in a pixel y and then a pixel x to determine the pixel x position where a line at that xy would trace to at y = 0 (center of the spectrum)
    def measureFullPixelSpectrum(self, current_image, spec_range, undo_line_curvature_dict,
                                 mu_index = 1, width_index = 3, width_fit_order = 2 ):
        spec_pixels = list(range(*spec_range))
        intensities = [[[], []] for i in range(len(spec_pixels))]
        spec_slice_pixels = list(range(np.shape(current_image)[(self.spec_axis + 1) % 2]))

        for i in range(len(spec_pixels)):
            pix = spec_pixels[i]
            #print ('Computing intensity for pix = ' + str(pix))
            sys.stdout.write("\r{0}".format('Computing intensity for pixel ' + str(pix) + ' (' + str(int (i / len(spec_pixels) * 100)) + '% done)...' ))
            sys.stdout.flush()
            if self.spec_axis == 0:
                spec_slice = current_image[pix, :]
            else:
                spec_slice = current_image[:, pix]

            #intensities[i] = [[wavelength_of_mu_solution(spec_slice_pixels[j], pix) for j in range(len(spec_slice_pixels))],
            #                  [spec_slice[j] / throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) if throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) > 0.0 else 0.0 for j in range(len(spec_slice_pixels))] ]
            #Determine the pixel that a line passing through this pixel on this row should trace to, according to the line curvature solutions that we have already found
            intensities[i] = [[undo_line_curvature_dict[pix](spec_slice_pixel) for spec_slice_pixel in spec_slice_pixels ], [spec_slice_val for spec_slice_val in spec_slice]]

        print ('Intensities computed.')
        #print ('intensities[0] = ' + str(intensities[0]))
        #print ('intensities[-1] = ' + str(intensities[-1]))
        intensity_interps = [can.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
        #full_spec_interp = can.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
        full_spec_interp = can.safeInterp1d( spec_slice_pixels, [np.sum([interp(slice_pixel) for interp in intensity_interps]) for slice_pixel in spec_slice_pixels] )
        #Don't do the curvature correcting
        #full_spec_interp = can.safeInterp1d( spec_slice_pixels, np.sum(current_image[spec_range[0]:spec_range[1], :], axis = 0))

        return spec_slice_pixels, full_spec_interp



    def deconvolveSpectrum(self, current_image, spec_range, mu_of_wavelength_solution, strong_line_profiles, throughput_file, throughput_dir, n_ignore_throughput,
                           mu_index = 1, width_index = 3, width_fit_order = 2, wavelength_range = [300.0, 1100.0], wavelength_step = 0.5 ):
        throughput_interp = readFileIntoInterp(throughput_file, throughput_dir, n_ignore_throughput, convert_to_float = 1) # Throughput(wavelength)
        spec_pixels = list(range(*spec_range))
        intensities = [[[], []] for i in range(len(spec_pixels))]
        #for i in range(len(spec_pixels)):
        for i in range(len(spec_pixels)):
            pix = spec_pixels[i]
            #print ('Computing intensity for pix = ' + str(pix))
            sys.stdout.write("\r{0}".format('Computing intensity for pixel ' + str(pix) + ' (' + str(int (i / len(spec_pixels) * 100)) + '% done)...' ))
            sys.stdout.flush()
            if self.spec_axis == 0:
                spec_slice = current_image[pix, :]
            else:
                spec_slice = current_image[:, pix]
            width_funct = np.poly1d(np.polyfit([strong_line[mu_index](pix) for strong_line in strong_line_profiles], [strong_line[width_index](pix) for strong_line in strong_line_profiles], width_fit_order))
            #plt.scatter([strong_line[mu_index](pix) for strong_line in strong_line_profiles], [strong_line[width_index](pix) for strong_line in strong_line_profiles])
            #plt.plot(np.arange(min([strong_line[mu_index](pix) for strong_line in strong_line_profiles]), max([strong_line[mu_index](pix) for strong_line in strong_line_profiles])),
            #         width_funct(np.arange(min([strong_line[mu_index](pix) for strong_line in strong_line_profiles]), max([strong_line[mu_index](pix) for strong_line in strong_line_profiles]))))
            #plt.show()
            spec_slice_pixels = list(range(len(spec_slice)))
            widths = width_funct(spec_slice_pixels)

            unity_interp = lambda wave: 1.0
            throughput_interp = unity_interp
            intensities[i] = [[wavelength_of_mu_solution(spec_slice_pixels[j], pix) for j in range(len(spec_slice_pixels))],
                              [spec_slice[j] / throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) if throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) > 0.0 else 0.0 for j in range(len(spec_slice_pixels))] ]
            #f, axarr = plt.subplots(2,1)
            #axarr[0].plot(spec_slice_pixels, spec_slice)
            #axarr[1].plot(spec_slice_pixels, np.exp((-np.array(spec_slice_pixels) ** 2.0 ) / (2.0 * widths ** 2.0)))
            #axarr[1].plot(spec_slice_pixels, intensities[i][1])
            #plt.show()
            #plt.plot(spec_slice_pixels, [throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) for j in spec_slice_pixels])
            #plt.plot(spec_slice_pixels, [np.sqrt(2.0 * np.pi * widths[j] ** 2.0) for j in spec_slice_pixels])
            #plt.show()
        print ('Intensities computed.')
        intensity_interps = [can.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
        full_spec_interp = can.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
        doconvolved_lines = 1
        f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
        axarr[0].plot(np.arange(*wavelength_range, wavelength_step), intensity_interps[len(spec_pixels) // 2](np.arange(*wavelength_range, wavelength_step)))
        axarr[1].plot(np.arange(*wavelength_range, wavelength_step), full_spec_interp(np.arange(*wavelength_range, wavelength_step)))
        plt.show()
        return 1

    def stackImage(self, current_images, current_headers, combine = 'median'):
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

    def correctBackground(self, image, spec_range, background_buffer, background_size, background_low = 1, ):
        background_cut_pixels = [int(self.mu_of_wavelength_solution(self.background_cut_wavelengths[0])), int(self.mu_of_wavelength_solution(self.background_cut_wavelengths[1]))]

        image_shape = np.shape(image)
        print ('[spec_range, background_cut_pixels, self.bg_fit_range] = ' + str([spec_range, background_cut_pixels, self.bg_fit_range] ))
        #masked_image[spec_range[0]:spec_range[1], background_cut_pixels[0]:background_cut_pixels[1] ] = 0.0
        clip_section = [[max(spec_range[0] - self.bg_fit_range[0][0], 0), min(spec_range[1] + self.bg_fit_range[0][0], image_shape[0])],
                               [max(background_cut_pixels[0] - self.bg_fit_range[1][0], 0), min(background_cut_pixels[1] + self.bg_fit_range[1][1], image_shape[1])]]
        print ('clip_section = ' + str(clip_section))
        clipped_image = image[clip_section[0][0]:clip_section[0][1], clip_section[1][0]:clip_section[1][1]]
        masked_image = clipped_image.copy()
        can.saveDataToFitsFile(np.transpose(clipped_image), 'ClippedImage.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        clipped_image_shape = np.shape(clipped_image)
        fit_x_lims = [-clipped_image_shape[1] // 2, clipped_image_shape[1] // 2 ]
        fit_y_lims = [-clipped_image_shape[0] // 2, clipped_image_shape[0] // 2]
        spec_box_section = [[background_cut_pixels[0] - clip_section[1][0] - clipped_image_shape[1] // 2, background_cut_pixels[1] - clip_section[1][0] - clipped_image_shape[1] // 2],
                            [spec_range[0] - clip_section[0][0] - clipped_image_shape[0] // 2, spec_range[1] - clip_section[0][0] - clipped_image_shape[0] // 2] ]

        masked_image[spec_range[0] - clip_section[0][0]:spec_range[1] - clip_section[0][0],
                     background_cut_pixels[0] - clip_section[1][0]:background_cut_pixels[1] - clip_section[1][0] ] = 0.0
        print ('[fit_x_lims, fit_y_lims, spec_box_section] = ' + str([fit_x_lims, fit_y_lims, spec_box_section]))
        can.saveDataToFitsFile(np.transpose(masked_image), 'ImageWithSpecMask.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        can.saveDataToFitsFile(np.transpose(image), 'ImageWithBackgound.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        background_fit, background_fit_funct, clipped_background = can.fitMaskedImage(clipped_image, mask_region = spec_box_section, fit_funct = 'poly3', verbose = 0, param_scalings = [1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0], x_lims = fit_x_lims, y_lims = fit_y_lims, )
        x_mesh, y_mesh = np.meshgrid(range(0, image_shape[1]) , range(0, image_shape[0]))
        fit_x_mesh = x_mesh - clip_section[1][0] - clipped_image_shape[1] // 2
        fit_y_mesh = y_mesh - clip_section[0][0] - clipped_image_shape[0] // 2
        background = background_fit_funct(fit_x_mesh, fit_y_mesh)

        can.saveDataToFitsFile(np.transpose(background), 'BackgroundFit.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        can.saveDataToFitsFile(np.transpose(clipped_image - clipped_background), 'ClippedImageWithoutBackground.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        can.saveDataToFitsFile(np.transpose(clipped_background), 'ClippedBackgroundFit.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)



        print ('background_fit = ' + str(background_fit))
        """
        image_shape = np.shape(image)
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, image_shape[0]), np.linspace(0, image_shape[1]))
        if background_low:
            background_range = [max(0, spec_range[0] - (background_buffer + background_size)),
                                min(np.shape(image)[(self.spec_axis + 1)%2], spec_range[0] - background_buffer )]
        else:
            background_range = [max(0, spec_range[1] + background_buffer ),
                                min(np.shape(image)[(self.spec_axis + 1)%2], spec_range[1] + background_size + background_buffer )]
        if self.spec_axis == 0:
            background = image[background_range[0]:background_range[1], :]
        else:
            background = image[:, background_range[0]:background_range[1]]
        if background_size > 1:
            background = np.median(background, axis = self.spec_axis)

        for pix in range(*spec_range):
            if self.spec_axis == 0:
                image[pix, :] = image[pix, :] - background
            else:
                image[:, pix] = image[:, pix] - background
        """
        image = image - background
        can.saveDataToFitsFile(np.transpose(image), 'BackgroundSubtractedImage.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        return image


    def processImages(self, spec_files_to_reduce = None, do_bias = None, do_dark = None, crc_correct = None, cosmic_prefix = None, save_stacked_image = None, save_image_name = None ):
        if do_bias is None:
            do_bias = self.do_bias
        if do_dark is None:
            do_dark = self.do_dark
        if crc_correct is None:
            crc_correct = self.crc_correct
        if cosmic_prefix is None:
            cosmic_prefix = self.cosmic_prefix
        if spec_files_to_reduce == None:
            spec_files_to_reduce = self.spec_files
        if save_stacked_image == None:
            save_stacked_image = self.save_stacked_image


        self.current_images = [[] for spec_file in spec_files_to_reduce]
        self.current_headers = [[] for spec_file in spec_files_to_reduce]
        if crc_correct:
            self.CleanCosmics(self.target_dir, spec_files_to_reduce, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = cosmic_prefix)
            spec_files_to_reduce = [cosmic_prefix + spec_file for spec_file in spec_files_to_reduce]
        for i in range(len(spec_files_to_reduce)):
            print ('Reading in raw spectrum from ' + self.target_dir + spec_files_to_reduce[i])
            self.current_images[i], self.current_headers[i] = self.readInRawSpect(spec_files_to_reduce[i], self.target_dir) #Read in raw spectrum

        #Overscan correct (?) ## Not currently set up to do this

        #[OPTIONAL] Make master bias
        if do_bias or do_dark:
            master_bias_exists = os.path.isfile(self.target_dir + self.master_bias_file)
            if not(master_bias_exists):
                print ('Making master bias file.  Will be saved to ' + self.target_dir + self.master_bias_file)
                master_bias_exists = self.makeMasterBias(self.master_bias_file, self.target_dir)
            if not(master_bias_exists):
                print ('Unable to find master bias file, ' + self.target_dir + self.master_bias_file + ', and also could not make it.  Returning without processing.')
                #sys.exit()

        #[OPTIONAL] Make master dark
        if do_dark :
            master_dark_exists = os.path.isfile(target_dir + master_dark_file)
            if not(master_dark_exists):
                print ('Making master dark file.  Will be saved to ' + self.target_dir + self.master_dark_file)
                master_dark_exists = self.makeMasterDark(self.master_dark_file, self.target_dir, self.master_bias_file)
            if not(master_dark_exists):
                print ('Unable to find master dark file, ' + self.target_dir + self.master_dark_file + ', and also could not make it.  Returning without processing.')
                #sys.exit()

        #Bias Subtract
        for i in range(len(spec_files_to_reduce)):
            if do_bias or do_dark:
                self.current_images[i], self.current_headers[i] = self.biasSubtract(self.current_images[i], self.current_headers[i], self.master_bias_file)
            if do_dark:
                self.current_images[i], self.current_headers[i] = self.darkSubtract(self.current_images[i], self.current_headers[i], self.master_dark_file)

        self.current_image, self.current_header = self.stackImage(self.current_images, self.current_headers, combine = 'mean' )
        #self.current_image = np.median(self.current_images, axis = 0)
        self.current_header = self.current_headers[0]
        if save_stacked_image and save_image_name != None:
            can.saveDataToFitsFile(np.transpose(self.current_image), save_image_name , self.target_dir, header = self.current_header)

        print ('spec_files_to_reduce = '  + str(spec_files_to_reduce ))
        print ('self.target_dir = ' + str(self.target_dir))
        if self.remove_intermed_images and crc_correct:
            [os.remove(self.target_dir + file) for file in spec_files_to_reduce]
        return self.current_image, self.current_header

    def loadWavelengthSolution(self, solution_file = None, load_dir = ''):
        if solution_file == None:
            solution_file = self.ref_spec_solution_file
        #wavelength_poly_terms = np.load(archival_data_dir + solution_file)
        loaded_solution = can.readInColumnsToList(solution_file, file_dir = self.target_dir, n_ignore = 0, convert_to_float = 1)[0]
        self.anchor_parallel_pix, self.wavelength_poly_terms = [int(loaded_solution[0]), loaded_solution[1:]]
        print ('wavelength_poly_terms = ' + str(self.wavelength_poly_terms))
        #mu_of_wavelength_solution, wavelength_of_mu_solution = create2DWavelengthSolutionCallableFunctions(wavelength_poly_terms)
        self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(self.wavelength_poly_terms)
        return self.mu_of_wavelength_solution, self.wavelength_of_mu_solution


    def reduceImagesTo1dSpectrum(self, images_to_reduce, n_std_for_strong_line, save_intermediate_images = 0, define_new_anchor_pix = 0, assign_new_master_val = 0, save_image_name = None, redetermine_spec_range = 1, crc_correct = None, reference_image = 0, bin_along_curve = 1, normalize = 1):
        #Makes self.current_images point to an array of the coadded, reduced images_to_reduce
        current_image, current_header = self.processImages(spec_files_to_reduce = images_to_reduce, save_image_name = save_image_name, crc_correct = crc_correct)
        self.image_roots = [spec_file[0:-len(self.data_image_suffix)] for spec_file in images_to_reduce]
        self.processed_file = self.image_roots[0] + self.processed_file_suffix + self.data_image_suffix
        self.processed_spectrum = self.image_roots[0] + self.processed_spectra_image_suffix + self.figure_suffix
        self.perp_spec_image_name = self.image_roots[0] + self.perp_spec_image_suffix + self.figure_suffix
        print ('[self.spec_range == None, redetermine_spec_range] = ' + str([self.spec_range == None, redetermine_spec_range]))
        if self.spec_range == None or redetermine_spec_range:
            self.spec_range = self.determineSpecRowRanges(self.current_image, showIDedLines = self.show_fits, save_perp_spec_image = self.save_perp_spec_image, perp_spec_image_name = self.target_dir + self.perp_spec_image_name)
        else:
            print ('Using reference spectral range... ')
        if define_new_anchor_pix:
            self.anchor_parallel_pix = ( self.spec_range[1] + self.spec_range[0] ) // 2

        print ('self.spec_range = ' + str(self.spec_range))
        self.current_image = self.correctBackground(self.current_image, self.spec_range, self.background_buffer, self.background_size, background_low = self.background_low, )

        if save_intermediate_images:
            can.saveDataToFitsFile(np.transpose(self.current_image), self.processed_file, self.target_dir, header = self.current_header, overwrite = True, n_mosaic_extensions = 0)
            print ('Just saved processed file to: ' + self.target_dir + self.processed_file)

        if bin_along_curve :
            print ('self.n_std_for_strong_line = ' + str(self.n_std_for_strong_line ))
            if reference_image:
                self.strong_lines = self.detectLinesInImage(self.current_image, self.spec_range,
                                                  n_std_for_lines = n_std_for_strong_line, line_fit_width = self.width_pix_sample_to_fit_ref_line,
                                                  search_binning = self.strong_line_search_binning, fit_binning = self.strong_line_fit_binning,
                                                  max_line_fit_width = self.max_line_fit_width, parallel_smoothing = self.parallel_smoothing,
                                                  width_guess = self.line_width_guess, show_process = self.show_fits,
                                                  max_sep_per_pix = self.max_sep_per_pixel_for_line_trace, min_detections = self.min_detections_for_ident_as_line ,
                                                  draw_stats_below = self.background_low, buffer_for_line_background_stats = self.background_buffer,
                                                  )
            else:
                self.strong_lines = self.detectLinesInImage(self.current_image, self.spec_range,
                                                  n_std_for_lines = n_std_for_strong_line, line_fit_width = self.width_pix_sample_to_fit_line,
                                                  search_binning = self.strong_line_search_binning, fit_binning = self.strong_line_fit_binning,
                                                  max_line_fit_width = self.max_line_fit_width, parallel_smoothing = self.parallel_smoothing,
                                                  width_guess = self.line_width_guess, show_process = self.show_fits,
                                                  max_sep_per_pix = self.max_sep_per_pixel_for_line_trace, min_detections = self.min_detections_for_ident_as_line ,
                                                  draw_stats_below = self.background_low, buffer_for_line_background_stats = self.background_buffer,
                                                  )
            if len(self.strong_lines) < 1:
                print ('Failed to identify any statistically significant lines in images stacked from ' + str(images_to_reduce))
                return [[], [], [], []]

            #print ('strong_lines = ' + str(strong_lines))
            line_pix_vals = can.union([[line_slice[0] for line_slice in line] for line in self.strong_lines])
            line_range = [min(line_pix_vals), max(line_pix_vals)]
            for i in range(len(self.strong_lines)):
                line = self.strong_lines[i]

            self.strong_line_profiles = [self.getLineFunction(line,
                                                    position_order = self.line_mean_fit_order,
                                                    A_of_x_order = self.line_amplitude_fit_order, sig_of_x_order = self.line_width_fit_order)
                                    for line in self.strong_lines]
            self.strong_lines = self.fixContinuousLines(self.strong_lines, self.strong_line_profiles )
            self.strong_line_profiles = [self.getLineFunction(line,
                                                    position_order = self.line_mean_fit_order,
                                                    A_of_x_order = self.line_amplitude_fit_order, sig_of_x_order = self.line_width_fit_order)
                                    for line in self.strong_lines]
            if self.show_fits:
                for strong_line in self.strong_lines:
                    plt.plot([line_slice[2] for line_slice in strong_line], [line_slice[0] for line_slice in strong_line])
                plt.xlabel('Column number (pix)')
                plt.ylabel('Line number (pix)')
                plt.title('Profiles of identified lines')
                plt.show()

            line_fit_mean_areas = [integrate.quad(lambda x: strong_line_profile[2](x) * strong_line_profile[3](x), strong_line_profile[0][0], strong_line_profile[0][-1])[0] / (strong_line_profile[0][-1] - strong_line_profile[0][0])
                             for strong_line_profile in self.strong_line_profiles ]
            if self.show_fits:
                f, axarr = plt.subplots(2,2, figsize = self.default_figsize)
                for i in range(len(self.strong_line_profiles)):
                    strong_line_profile = self.strong_line_profiles[i]
                    axarr[0,0].plot(strong_line_profile[0], [strong_line_profile[1](x) for x in strong_line_profile[0]])
                    axarr[0,0].set_xlabel('Row number (pix)')
                    axarr[0,0].set_ylabel('Center of line')
                    axarr[0,1].plot(strong_line_profile[0], [strong_line_profile[2](x) for x in strong_line_profile[0]])
                    axarr[0,1].set_xlabel('Row number (pix)')
                    axarr[0,1].set_ylabel('Height of line')
                    axarr[1,0].plot(strong_line_profile[0], [strong_line_profile[3](x) for x in strong_line_profile[0]])
                    axarr[1,0].set_xlabel('Row number (pix)')
                    axarr[1,0].set_ylabel('Width of line')
                    axarr[1,1].plot(strong_line_profile[0], [2.0 * np.pi * strong_line_profile[2](x) * strong_line_profile[3](x) for x in strong_line_profile[0] ] )
                    axarr[1,1].set_xlabel('Row number (pix)')
                    axarr[1,1].set_ylabel('Area of line')
                    axarr[0,0].text(strong_line_profile[0][0], strong_line_profile[1](strong_line_profile[0][0]), str(i) )
                    axarr[0,1].text(strong_line_profile[0][0], strong_line_profile[2](strong_line_profile[0][0]), str(i) )
                    axarr[1,0].text(strong_line_profile[0][0], strong_line_profile[3](strong_line_profile[0][0]), str(i) )
                    axarr[1,1].text(strong_line_profile[0][0], 2.0 * np.pi * strong_line_profile[2](strong_line_profile[0][0]) * strong_line_profile[3](strong_line_profile[0][0]), str(i) )
                    plt.tight_layout()
                plt.show()

            print ('line_fit_mean_areas = ' + str(line_fit_mean_areas))

            #We now sum the spectrum along the interpolated lines of curvature, as determined from the strong lines.
            undo_line_curvature_dict = self.determineMacroLineCurvatureDict (self.spec_range, self.strong_line_profiles)
            #print ('undo_line_curvature_dict = ' + str(undo_line_curvature_dict))
            print ('Binning spectrum, along curvature of lines...')
            parallel_spec_pixels, self.full_pix_interp = self.measureFullPixelSpectrum(self.current_image, self.spec_range, undo_line_curvature_dict)
            #We also do the same thing on a region above or below to determine the background statistics for our binned spectrum
            if self.background_low:
                stat_region = [max(0, self.spec_range[0] - self.background_buffer - (self.spec_range[1] - self.spec_range[0]) ), self.spec_range[0] - self.background_buffer ]
            else:
                stat_region = [self.spec_range[1] + self.background_buffer,
                               min(np.shape(self.current_image)[self.spec_axis], search_range[1] + self.background_buffer + (self.spec_range[1] - self.spec_range[0])) ]
            #print ('stat_region = ' + str(stat_region))
            print ('Binning background in the same way as the spectrum...')
            background_inserted_spectrum = np.copy(self.current_image)
            print ('[self.spec_range, self.stat_region] = ' + str([self.spec_range, stat_region] ))
            if self.spec_axis == 0:
                if self.spec_range[1] -self.spec_range[0] <= stat_region[1] - stat_region[0]:
                    background_inserted_spectrum[self.spec_range[0]:self.spec_range[1], :] = self.current_image[stat_region[0]:stat_region[1], :]
                else:
                    stat_region_len = (stat_region[1] - stat_region[0])
                    if self.background_low:
                        background_inserted_spectrum[self.spec_range[1] - stat_region_len:self.spec_range[1], :] = self.current_image[stat_region[0]:stat_region[1], :]
                        background_inserted_spectrum[self.spec_range[0]:self.spec_range[1] - stat_region_len, :] = np.median(self.current_image[stat_region[0]:stat_region[1], :], axis = 0)
                    else:
                        background_inserted_spectrum[self.spec_range[0]:self.spec_range[0] + stat_region_len, :] = self.current_image[stat_region[0]:stat_region[1], :]
                        background_inserted_spectrum[self.spec_range[0] + stat_region_len:self.spec_range[1], :] = np.median(self.current_image[stat_region[0]:stat_region[1], :], axis = 0)

            else:
                background_inserted_spectrum[:, self.spec_range[0]:self.spec_range[1]] =  self.current_image[:, stat_region[0]:stat_region[1]]

            can.saveDataToFitsFile(background_inserted_spectrum, 'ArrayFromWhicBackgroundStatsAreComputed.fits', self.target_dir)
            self.stat_slice_pixels, self.stat_slice_interp = self.measureFullPixelSpectrum(background_inserted_spectrum, self.spec_range, undo_line_curvature_dict)

            parallel_spec_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in parallel_spec_pixels]
            full_pix_spectrum = self.full_pix_interp(parallel_spec_pixels)
            full_pix_background_stats = self.stat_slice_interp(self.stat_slice_pixels)

            if self.show_fits:
                max_full = np.max(full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
                max_ref = max(full_pix_background_stats[self.spec_range[0]:self.spec_range[1]])
                spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum / max_full , c = 'g')[0]
                background = plt.plot(parallel_spec_wavelengths, full_pix_background_stats / max_full, c = 'r')[0]

            full_pix_spectrum = full_pix_spectrum / self.throughput_interp(parallel_spec_wavelengths)
            full_pix_background_stats = full_pix_background_stats / self.throughput_interp(parallel_spec_wavelengths)
            max_full = np.max(full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
            max_ref = max(full_pix_background_stats[self.spec_range[0]:self.spec_range[1]])
            if self.show_fits:
                throughput_plot = plt.plot(parallel_spec_wavelengths, self.throughput_interp(parallel_spec_wavelengths), c = 'k')[0]
                norm_spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum / max_full , c = 'blue', )[0]
                norm_background = plt.plot(parallel_spec_wavelengths, full_pix_background_stats / max_full, c = 'orange')[0]
                full_pix_background_stats = full_pix_background_stats / self.throughput_interp(parallel_spec_wavelengths)
                plt.title('Re-measured spectrum, binned according to strong line fits')
                plt.xlabel('Start of bin line (pix)')
                plt.ylabel('Total count binned along fit (ADU)')
                plt.ylim(-0.05, 1.05)
                plt.legend([spec, background, throughput_plot, norm_spec, norm_background], ['Spec. - bg', 'bg', 'Throughput', 'Throughput corr. Spec. - bg', 'Throughput corr. bg'])
                plt.show()
            if normalize:
                full_pix_spectrum = full_pix_spectrum / max_full
                full_pix_background_stats  = full_pix_background_stats / max_full
        else:
            full_pix_spectrum = np.sum(self.current_image[self.spec_range[0]:self.spec_range[1], :], axis = 0).tolist()
            parallel_spec_pixels = list(range(len(full_pix_spectrum)))
            print ('[binned_spec, parallel_spec_pixels] = ' + str([full_pix_spectrum, parallel_spec_pixels]))
            print ('[len(binned_spec), len(parallel_spec_pixels)] = ' + str([len(full_pix_spectrum), len(parallel_spec_pixels)]))
            full_pix_background_stats = []


        return parallel_spec_pixels, full_pix_spectrum , full_pix_background_stats, self.current_header

    def measureSystemThroughtput(self, throughput_files_list, target_dir = None, show_fits = 1):
        if target_dir == None:
            target_dir = self.target_dir
        if show_fits != None:
            self.show_fits = show_fits
        if self.wavelength_of_mu_solution == None:
            self.getWavelengthSolution()
        throughput_files =  can.readInColumnsToList(throughput_files_list, file_dir = target_dir, n_ignore = 0, )[0]
        self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(throughput_files, self.n_std_for_strong_ref_line, define_new_anchor_pix = 1, save_intermediate_images = 1, crc_correct = 0, reference_image = 0, bin_along_curve = 0)
        parallel_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_ref_spec_pixels]
        max_throughput_ADU = np.max(self.full_ref_pix_spectrum)
        throughput_source = can.readInColumnsToList(self.ref_throughput_file, file_dir = self.archival_data_dir, n_ignore = 1, delimiter = ',')
        ref_throughput_interp = scipy.interpolate.interp1d([float(elem) for elem in throughput_source[0]], np.array([float(elem) for elem in throughput_source[1]]) * np.array([float(elem) for elem in throughput_source[0]]) * self.energy_to_photon_scaling, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
        interped_throughput_ref = ref_throughput_interp(parallel_wavelengths).tolist()
        throughput = np.array(self.full_ref_pix_spectrum) / np.array(interped_throughput_ref)
        max_throughput = np.max(throughput)
        rel_throughput = throughput / max_throughput
        max_interped_throughput_ref = max(interped_throughput_ref )
        if show_fits:
            fig = plt.figure(figsize=(9, 3))
            measured = plt.plot(parallel_wavelengths, self.full_ref_pix_spectrum / max_throughput_ADU, c = 'g')[0]
            ref = plt.plot(parallel_wavelengths, np.array(interped_throughput_ref) / max_interped_throughput_ref, c = 'r')[0]
            throughput = plt.plot(parallel_wavelengths, rel_throughput, c = 'b')[0 ]
            plt.xlabel('Wavelength (nm)', fontsize= 14)
            plt.ylabel('Normalized intensity', fontsize= 14)
            plt.legend([measured, ref, throughput], ['DH2000 on OSELOTS', 'DH2000 Reported Intensity', 'Inferred Throughput'], fontsize = 10)
            plt.tight_layout()
            plt.show()
        new_thruoghput = can.saveListsToColumns([parallel_wavelengths, rel_throughput], self.throughput_file, self.archival_data_dir, sep = ',', header = 'Wavelengths(nm), Throughput')

        return 1


    def computeReferenceSpectrum(self, ref_spec_images_list = None, save_new_reference_spectrum = 1, data_dir = None, spec_save_dir = None, ref_spec = None ):
        if ref_spec_images_list == None:
            ref_spec_images_list = self.ref_spec_images_list
        if data_dir == None:
            data_dir = self.target_dir
        if spec_save_dir == None:
            spec_save_dir = self.target_dir
        if ref_spec == None:
            ref_spec = self.ref_spec
        ref_spec_file = self.spec_archival_info[ref_spec]['spec_file']
        n_ignore_spec = self.spec_archival_info[ref_spec]['n_spec_lines_to_ignore']
        ref_just_lines_file = self.spec_archival_info[ref_spec]['lines_file']
        n_ignore_lines = self.spec_archival_info[ref_spec]['n_lines_lines_to_ignore']
        throughput_file = self.spec_archival_info['throughput']['spec_file']
        n_ignore_throughput = self.spec_archival_info['throughput']['n_lines_to_ignore']

        ref_spec_image_files = can.readInColumnsToList(ref_spec_images_list, file_dir = data_dir, n_ignore = 0, )[0]
        self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(ref_spec_image_files, self.n_std_for_strong_ref_line, define_new_anchor_pix = 1, save_intermediate_images = 1, crc_correct = 0, reference_image = 1)
        if len(self.full_ref_pix_spectrum) < 1:
            print ('Failed to identify lines in reference image list ' + str(ref_spec_images_list))
            false_wavelength_solution = np.zeros(self.wavelength_solution_order)
            false_wavelength_solution[-1] = self.approx_pix_scale
            mu_of_wavelength_solution, wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(false_wavelength_solution)
            return [mu_of_wavelength_solution, wavelength_of_mu_solution]
        #print ('[full_pix_spectrum, full_pix_background_stats] = ' + str([full_pix_spectrum, full_pix_background_stats]))

        self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings = self.detectLinesCentersInOneD(self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_ref_line, show = self.show_fits)
        print ('[self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings] = ' + str([self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings]))
        self.full_ref_emission_crossings = [crossing for crossing in self.full_ref_emission_crossings if (crossing > 0 and crossing < self.parallel_ref_spec_pixels[-1])]

        self.full_ref_emission_fits = self.identifyLinesOnSlice(self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum,
                                 max_line_fit_width = self.width_pix_sample_to_fit_ref_line, peak_guesses = self.full_ref_emission_crossings, show_spec = self.show_fits, verbose = 1,
                                 fit_lines_with_pedestal = 1, fit_line_width_with_seeing_funct = 0)

        print ('self.full_ref_emission_fits = ' + str(self.full_ref_emission_fits))
        #self.full_emission_centroids = [fit[1] for fit in self.full_emission_fits]
        full_ref_emission_pixels_vs_widths = [[fit[1], fit[2]] for fit in self.full_ref_emission_fits]
        #print('self.full_emission_pixels_vs_widths = ' + str(self.full_emission_pixels_vs_widths))
        self.seeing_ref_fit_funct = np.poly1d(can.polyFitNSigClipping( [emit[0] for emit in full_ref_emission_pixels_vs_widths], [emit[1] for emit in full_ref_emission_pixels_vs_widths], 2, self.sig_clip_for_line_width) [3])

        mu_of_wavelength_solution, wavelength_of_mu_solution = self.determineWavelengthSolution(self.full_ref_emission_fits, self.spec_range, ref_spec_file, ref_just_lines_file,
                                                                                       spec_file_dir = self.archival_data_dir, throughput_file = throughput_file,
                                                                                       n_ignore_spec = n_ignore_spec, n_ignore_lines = n_ignore_lines, n_ignore_throughput = n_ignore_throughput,
                                                                                       show_solution = self.show_fits,
                                                                                       solution_save_dir = spec_save_dir, save_solution = save_new_reference_spectrum )

        return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    def getWavelengthSolution(self, ref_spec_images_list = None, ref_spec_solution_file = None, save_new_reference_spectrum = 1, ref_spec = None, show_fits = None):
        if show_fits != None:
            self.show_fits = show_fits
        if ref_spec_images_list != None:
            if ref_spec_images_list != None:
                self.ref_spec_images_list = ref_spec_images_list
            self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.computeReferenceSpectrum(save_new_reference_spectrum = save_new_reference_spectrum, ref_spec = ref_spec)
            if save_new_reference_spectrum:
                if ref_spec_solution_file == None:
                    self.ref_spec_solution_file = ref_spec_solution_file
        else:
            if ref_spec_solution_file != None:
                self.ref_spec_solution_file = ref_spec_solution_file
            self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.loadWavelengthSolution(solution_file = self.ref_spec_solution_file, load_dir = self.archival_data_dir)

        return 1

    def subtractContinuum(self, pixels, spec_of_pixels, used_continuum_seeds = None, continuum_smoothing = None, n_continuum_fit_seeds = None, min_line_vs_seed_sep = None, show_fits = None):
        if used_continuum_seeds == None:
            used_continuum_seeds = can.readInColumnsToList(self.continuum_seeds_file, self.archival_data_dir, n_ignore = self.n_ignore_for_continuum_seeds, convert_to_float = 1) [0]
        used_continuum_seed_indeces = [int(self.mu_of_wavelength_solution(seed)) for seed in used_continuum_seeds]
        used_continuum_seedds = [self.wavelength_of_mu_solution(pix) for pix in used_continuum_seed_indeces]
        if show_fits == None:
            show_fits = self.show_fits
        if continuum_smoothing == None:
            continuum_smoothing = self.continuum_smoothing
        if n_continuum_fit_seeds == None:
            n_continuum_fit_seeds = self.n_continuum_fit_seeds
        if min_line_vs_seed_sep == None:
            min_line_vs_seed_sep = self.min_line_vs_seed_sep
        print ('continuum_smoothing = ' +  str(continuum_smoothing))
        print('min_line_vs_seed_sep = ' + str(min_line_vs_seed_sep))
        print ('n_continuum_fit_seeds = ' + str(n_continuum_fit_seeds))
        initial_continuum_seed_indeces = np.linspace(0, len(pixels)-1, n_continuum_fit_seeds + 1)[1:-1]
        initial_continuum_seed_indeces = [int(index) for index in initial_continuum_seed_indeces]
        #print('[np.min([abs(pixels[index] - line_center) for line_center in line_centers]) for index in initial_continuum_seed_indeces ] = ' + str([np.min([abs(pixels[index] - line_center) for line_center in line_centers]) for index in initial_continuum_seed_indeces ] ))
        #used_continuum_seed_indeces = [index for index in initial_continuum_seed_indeces if np.min([abs(pixels[index] - line_center) for line_center in line_centers]) >= min_line_vs_seed_sep ]
        print ('used_continuum_seeds = ' + str(used_continuum_seeds))
        print ('used_continuum_seed_indeces = ' + str(used_continuum_seed_indeces ) )
        #used_continuum_seeds = [pixels[index] for index in used_continuum_seed_indeces]
        smoothed_spec = scipy.ndimage.gaussian_filter1d(spec_of_pixels, continuum_smoothing)
        sampled_continuum = [smoothed_spec[index] for index in used_continuum_seed_indeces]
        if show_fits:
            plt.scatter(used_continuum_seeds, sampled_continuum)
            plt.show()
        continuum_interp = scipy.interpolate.interp1d(used_continuum_seeds, sampled_continuum, kind = 'cubic', bounds_error = False, fill_value = 0.0)

        if show_fits:
            plt.plot(self.parallel_spec_wavelengths, spec_of_pixels, c = 'blue')
            plt.plot(self.parallel_spec_wavelengths, smoothed_spec, c = 'r')
            plt.plot(self.parallel_spec_wavelengths, continuum_interp(self.parallel_spec_wavelengths), c = 'g')
            plt.scatter(used_continuum_seeds, sampled_continuum, marker = 'o', c = 'k')
            plt.show()

        subtracted_spec = np.array(spec_of_pixels) - continuum_interp([self.wavelength_of_mu_solution(pix) for pix in pixels])
        return continuum_interp, subtracted_spec, [used_continuum_seeds, sampled_continuum]


    def correctThroughput(self, spec_wavelengths, spec_to_correct, throughput_file = None, throughput_data = None, throughput_file_delimiter = ' ', n_lines_to_ignore = 0, throughput_file_dir = None):
        print ('self.throughput_interp = ' + str(self.throughput_interp))
        if throughput_file != None or throughput_data != None or self.throughput_interp == None:
            self.throughput_interp = self.importSystemThroughput(throughput_file = throughput_file, throughput_data = throughput_data, throughput_file_delimiter = throughput_file_delimiter, n_lines_to_ignore = n_lines_to_ignore, throughput_file_dir = throughput_file_dir)
        throughput = np.array([self.throughput_interp(wave)[0] for wave in spec_wavelengths])
        corrected_spec = spec_to_correct / throughput
        print('[throughput, spec_to_correct, corrected_spec] = ' + str([throughput, spec_to_correct, corrected_spec]))
        return corrected_spec


    def measureStrengthOfLinesInImage(self, image_to_measure, show_fits = None, line_dict_id = None, redetermine_spec_range = 0):
        if show_fits != None:
            self.show_fits = show_fits
        if self.wavelength_of_mu_solution == None:
            self.getWavelengthSolution()
        if line_dict_id == None:
            line_dict_id = image_to_measure

        parallel_spec_pixels, full_pix_spectrum, full_pix_background_stats, image_header = self.reduceImagesTo1dSpectrum([image_to_measure], self.n_std_for_strong_line, save_image_name = image_to_measure[:-len(self.data_image_suffix)] + self.processed_file_suffix + self.data_image_suffix, redetermine_spec_range = redetermine_spec_range)
        if len(full_pix_spectrum) < 1:
            print ('Failed to identify lines in image ' + str(image_to_measure))
            return 0
        parallel_spec_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in parallel_spec_pixels]
        no_background_sub_full_pix_spectrum = full_pix_spectrum[:]
        continuum_interp, full_pix_spectrum, continuum_fit_points = self.subtractContinuum(parallel_spec_pixels, full_pix_spectrum, show_fits = show_fits)

        pix_target_range = [int(self.mu_of_wavelength_solution(self.wavelength_target_range[0])), int(self.mu_of_wavelength_solution(self.wavelength_target_range[1]))]
        full_absorbtion_crossings, full_emission_crossings = self.detectLinesCentersInOneD(parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_line, show = self.show_fits)

        init_full_emission_fits = self.identifyLinesOnSlice(range(len(full_pix_spectrum )), full_pix_spectrum,
                                 max_line_fit_width = self.width_pix_sample_to_fit_line, peak_guesses = self.identified_full_emission_line_centers, show_spec = show_fits, verbose = 1,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1)

        full_emission_centroids = [fit[1] for fit in init_full_emission_fits]
        seeing_fit_params = self.simulFitLineWidths(np.array(range(len(full_pix_spectrum ))), full_pix_spectrum, init_full_emission_fits)
        #print('seeing_fit_params = ' + str(seeing_fit_params ))
        seeing_fit_funct = np.poly1d(seeing_fit_params)
        #print ('seeing_fit_funct = ' + str(seeing_fit_funct))
        full_emission_fits = self.identifyLinesOnSlice(range(len(full_pix_spectrum )), full_pix_spectrum,
                                 max_line_fit_width = self.refined_width_pix_sample_to_fit_line,  peak_guesses = self.identified_full_emission_line_centers, show_spec = self.show_fits, verbose = 1,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 0, seeing_fit_funct = seeing_fit_funct )
        full_emission_fits = [fit for fit in full_emission_fits if fit[1] > parallel_spec_pixels[1] and fit[1] < parallel_spec_pixels[-2] ]
        #print ('full_emission_fits = ' + str(full_emission_fits))
        self.identified_lines_dict[line_dict_id] = { }
        self.identified_lines_dict[line_dict_id][self.lines_in_dict_keyword] = {i:full_emission_fits[i] for i in range(len(self.full_emission_fits))}
        self.identified_lines_dict[line_dict_id][self.obs_time_keyword] = image_header[self.obs_time_keyword]
        identified_full_emission_line_centers = [fit[1] for fit in full_emission_fits]
        identified_full_emission_line_heights = [fit[0] for fit in full_emission_fits]

        if self.show_fits or self.save_final_plot:
            self.plotFullLineImage(parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, full_pix_background_stats, continuum_interp, full_pix_spectrum,
                                   identified_full_emission_line_centers, identified_full_emission_line_heights, continuum_fit_points, self.wavelength_of_mu_solution,
                                   self.show_fits, self.save_final_plot, plot_title = 'Spectrum of image: ' + str(line_dict_id),
                                   save_image_name = image_to_measure[:-len(self.data_image_suffix)] + self.processed_spectra_image_suffix + self.figure_suffix)
        return 1


    def plotLineProfilesInTime(self, n_subplots = 8, n_cols = 2, figsize = (8,16), line_variation_image_name = 'skyLineChangesOverTime.pdf', n_legend_col = 2, legend_text_size = 8, n_ticks = 11, xlabel = r'$\Delta t$ since first exp. (min)', ylabel ='Fitted height of line (ADU)' , y_lims = [-0.05, 1.05]):
        single_lines = [[] for i in range(n_subplots)]
        if n_subplots == 1:
            f, axarr = plt.subplots(1,1)
        else:
            f, axarr = plt.subplots(n_subplots // n_cols, n_cols, figsize = figsize, sharex = True, sharey = True)
        #plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        spectra_ids = list(self.identified_lines_dict.keys())
        line_numbers = [key for key in self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword] ]
        n_lines_per_plot = len(line_numbers) // n_subplots
        n_lines_per_plot = [n_lines_per_plot + (1 if (len(line_numbers) % n_lines_per_plot - i) > 0 else 0)  for i in range(n_subplots)]
        line_numbers_by_plot = [0] + [np.sum(n_lines_per_plot[0:i]) for i in range(1, n_subplots)]  + [len(line_numbers)]
        n_extra_lines = len(line_numbers) % n_subplots
        timestamps = [can.convertDateTimeStrToSeconds(self.identified_lines_dict[spectrum_id][self.obs_time_keyword], self.date_format_str) for spectrum_id in spectra_ids]
        print ('timestamps = ' + str(timestamps))
        min_time = np.min(timestamps)
        delta_ts = [(time - min_time) / 60 for time in timestamps]

        for i in range(len(line_numbers)):
             plot_number = [j for j in range(len(n_lines_per_plot)) if i < line_numbers_by_plot[1:][j]] [0]
             line_number = line_numbers[i]
             line_heights = [self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][0] for key in self.identified_lines_dict.keys()]
             #y_lims = [min([y_lims[0]] + line_heights), max([y_lims[1]] + line_heights) ]
             image_numbers = [key for key in self.identified_lines_dict.keys()]
             #single_lines[plot_number] = single_lines[plot_number] + [axarr[plot_number // 2, plot_number % 2].plot(image_numbers[1:], line_heights[1:])[0] ]
             single_lines[plot_number] = single_lines[plot_number] + [axarr[plot_number // 2, plot_number % 2].plot(delta_ts[1:], line_heights[1:])[0] ]

        for i in range(n_subplots):
             print ('Making subplot ' + str(i + 1) + ' of ' + str(n_subplots))
             line_identifiers = ['L' + str(line_number) + ': ' + str(can.round_to_n(self.wavelength_of_mu_solution(self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword][line_number][1]), 3)) + 'nm' for line_number in line_numbers[line_numbers_by_plot[i]:line_numbers_by_plot[i+1]] ]
             axarr[ i // 2, i % 2].legend(single_lines[i], line_identifiers, ncol = n_legend_col, prop={'size':legend_text_size} )
             axarr[ i // 2, i % 2].set_ylim([*y_lims])
             if i // 2 == n_subplots // 2 -1 :
                  axarr[ i // 2, i % 2].set_xlabel(xlabel )
             if i % 2 == 0:
                  axarr[ i // 2, i % 2].set_ylabel(ylabel )
             axarr[ i // 2, i % 2 ].set_xticks
        print ('Now saving plot... ')
        plt.savefig(self.target_dir + line_variation_image_name )
        return 1


    def plotFullLineImage(self, parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, full_pix_background_stats, continuum_interp, full_pix_spectrum,
                                identified_full_emission_line_centers, full_emission_heights, continuum_fit_points, wavelength_of_mu_solution,
                                show_fits, save_final_plot, xlabel = 'Sky wavelength (nm)', ylabel = 'Strength of line (ADU)', plot_title = None, xticks = np.arange(400, 1301, 100),
                                xlims = None, ylims = None, save_image_name = 'NO_NAME_SAVED_FIGURE_OF_OSELOTS_LINES.txt', legend_pos = [[0.1, 0.75], 'center left']):

        if ylims == None:
            ylims = self.ylims
        if xlims == None:
            xlims = self.wavelength_target_range
        fig = plt.figure(figsize = [self.default_figsize[0], self.default_figsize[1] / 2])
        full_spec = plt.plot(parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, c = 'blue', zorder = -1)[0]
        #background_spec = plt.plot(parallel_spec_wavelengths, full_pix_background_stats, c = 'red', zorder = -2)[0]
        pre_throughput_spec = plt.plot(parallel_spec_wavelengths, no_background_sub_full_pix_spectrum * self.throughput_interp(parallel_spec_wavelengths) / np.max(no_background_sub_full_pix_spectrum * self.throughput_interp(parallel_spec_wavelengths) ), c = 'purple', zorder = -2)[0]
        continuum_estimate = plt.plot(parallel_spec_wavelengths, continuum_interp(parallel_spec_wavelengths), c = 'k', zorder = 0, linestyle = '--')[0]
        background_sub_spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum, c = 'green', zorder = 2)[0]
        identified_lines = plt.scatter([wavelength_of_mu_solution(center) for center in identified_full_emission_line_centers], full_emission_heights, marker = 'x', c = 'orange', zorder = 3)
        continuum_sampling_points = plt.scatter(continuum_fit_points[0], continuum_fit_points[1], marker = 'o', color = 'k', zorder = 1)
        identified_line_centers = [plt.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'orange', linewidth = 0.75) for center in identified_full_emission_line_centers]
        orig_line_centers = [plt.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'cyan', linewidth = 0.75) for center in self.identified_full_emission_line_centers]
        throughput = plt.plot(parallel_spec_wavelengths, self.throughput_interp(parallel_spec_wavelengths), c = 'purple', zorder = -4, alpha = 0.5 )[0]
        plt.plot(xlims, [0.0, 0.0], c = 'k', alpha = 0.5, zorder = -5)
        for i in range(len(full_emission_heights)):
            plt.text(self.wavelength_of_mu_solution(identified_full_emission_line_centers[i]), -500 + (i % 3) * 100, str(i), color = 'k', zorder = 4, fontsize = 8, horizontalalignment = 'center', )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print ('plot_title = ' + str(plot_title))
        plt.title(plot_title)
        plt.legend([identified_lines, full_spec, pre_throughput_spec, continuum_estimate, continuum_sampling_points, background_sub_spec, throughput], ['Identified lines', 'Full spectrum', 'Spectrum before throughput', 'Continuum estimate', 'Continuum interpolation points', 'Spec - background', 'Throughput'],bbox_to_anchor=legend_pos[0], loc = legend_pos[1], )
        #plt.scatter(ref_line_wavelengths, ref_line_heights , marker = 'x')
        plt.xticks(xticks)
        plt.tight_layout()
        plt.ylim(*ylims)
        plt.xlim(*xlims)
        if save_final_plot:
            print ('!!!! Saving final plot to figure named ' + save_image_name + '!!!!!')
            plt.savefig(self.target_dir + save_image_name )
            if not (show_fits):
                plt.close('all')
        if show_fits:
            plt.show()
        plt.close('all')
        return 1


    def pullCombinedSpectrumFromImages(self, spec_images_to_stack, show_fits = None, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked Spectrum'):
        if show_fits != None:
            self.show_fits = show_fits
        if self.wavelength_of_mu_solution == None:
            self.getWavelengthSolution()
        if line_dict_id == None:
            line_dict_id = self.stacked_image_line_dict_id
        self.parallel_spec_pixels, self.full_pix_spectrum, self.full_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(spec_images_to_stack, self.n_std_for_strong_line, save_image_name = self.stacked_image_name + self.data_image_suffix)

        print ('self.parallel_spec_pixels = ' + str(self.parallel_spec_pixels))
        self.parallel_spec_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_spec_pixels]
        #plt.plot(self.parallel_spec_wavelengths, self.full_pix_spectrum, c = 'green')

        self.no_background_sub_full_pix_spectrum = self.full_pix_spectrum[:]
        self.continuum_interp, self.full_pix_spectrum, self.continuum_fit_points = self.subtractContinuum(self.parallel_spec_pixels, self.full_pix_spectrum, show_fits = show_fits)
        self.continuum = self.continuum_interp(self.parallel_spec_wavelengths)
        full_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_spec_pixels]

        #self.strong_absorbtion_crossings, self.strong_emission_crossings = self.detectLinesCentersInOneD(self.full_pix_spectrum, self.full_pix_background_stats, spec_grad_rounding = 5, n_std_for_line = self.n_std_for_strong_line, show = self.show_fits)
        #self.strong_emission_fits = self.identifyLinesOnSlice(range(len(self.full_pix_spectrum )), self.full_pix_spectrum,
        #                         peak_guesses = self.strong_emission_crossings, std_thresh = 3, max_line_fit_width = 20,
        #                         n_pix_above_thresh = 1, init_fit_width_guess = 2.0, background_bin_width = 10,
        #                         search_width = 3.0, show_spec = self.show_fits, verbose = 1,
        #                         fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 0)
        #print ('self.strong_emission_fits = ' + str(self.strong_emission_fits))
        #strong_emission_line_centers = [fit[1] for fit in self.strong_emission_fits]
        #strong_emission_pixels_vs_widths = [[fit[1], fit[2]] for fit in self.strong_emission_fits]
        #print('self.full_emission_pixels_vs_widths = ' + str(self.full_emission_pixels_vs_widths))
        #self.seeing_fit_funct = np.poly1d(can.polyFitNSigClipping( [emit[0] for emit in strong_emission_pixels_vs_widths], [emit[1] for emit in strong_emission_pixels_vs_widths], 2, self.sig_clip_for_line_width) [3])

        pix_target_range = [int(self.mu_of_wavelength_solution(self.wavelength_target_range[0])), int(self.mu_of_wavelength_solution(self.wavelength_target_range[1]))]
        self.full_absorbtion_crossings, self.full_emission_crossings = self.detectLinesCentersInOneD(self.parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_line, show = self.show_fits)
        #print ('[self.full_absorbtion_crossings, self.full_emission_crossings] = ' + str([self.full_absorbtion_crossings, self.full_emission_crossings]))

        #self.full_absorbtion_crossings = [cross + pix_target_range[0] for cross in self.full_absorbtion_crossings]
        #self.full_emission_crossings = [cross + pix_target_range[0] for cross in self.full_emission_crossings]
        print('self.full_emission_crossings = ' + str(self.full_emission_crossings))
        self.full_emission_fits = self.identifyLinesOnSlice(self.parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                 max_line_fit_width = self.width_pix_sample_to_fit_line, peak_guesses = self.full_emission_crossings, show_spec = self.show_fits, verbose = 1,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1)
        full_emission_centroids = [fit[1] for fit in self.full_emission_fits ]
        self.seeing_fit_params = self.simulFitLineWidths(np.array(self.parallel_spec_pixels) , self.full_pix_spectrum, self.full_emission_fits)
        print('self.seeing_fit_params = ' + str(self.seeing_fit_params ))
        print ('full_emission_centroids = ' + str(full_emission_centroids))
        self.seeing_fit_funct = np.poly1d(self.seeing_fit_params)
        #print ('self.seeing_fit_funct = ' + str(self.seeing_fit_funct))

        self.full_emission_fits = self.identifyLinesOnSlice(range(len(self.parallel_spec_pixels)), self.full_pix_spectrum,
                                 max_line_fit_width = self.refined_width_pix_sample_to_fit_line, peak_guesses = full_emission_centroids, show_spec = self.show_fits, verbose = 1,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1)
        self.full_emission_fits = [fit for fit in self.full_emission_fits if fit[1] > self.parallel_spec_pixels[1] and fit[1] < self.parallel_spec_pixels[-2] ]
        #print ('self.full_emission_fits = ' + str(self.full_emission_fits))
        self.identified_full_emission_line_centers = [fit[1] for fit in self.full_emission_fits]
        full_emission_heights = [fit[0] for fit in self.full_emission_fits]
        full_emission_pixels_vs_widths = [[fit[1], fit[2]] for fit in self.full_emission_fits]
        self.identified_lines_dict[self.stacked_image_keyword] = {}
        self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword] = {i:self.full_emission_fits[i] for i in range(len(self.full_emission_fits))}
        self.identified_lines_dict[self.stacked_image_keyword][self.obs_time_keyword] = self.stacked_header[self.obs_time_keyword]
        #print('self.full_emission_pixels_vs_widths = ' + str(self.full_emission_pixels_vs_widths))
        #self.seeing_fit_funct = np.poly1d(can.polyFitNSigClipping( [emit[0] for emit in full_emission_pixels_vs_widths], [emit[1] for emit in full_emission_pixels_vs_widths], 2, self.sig_clip_for_line_width) [3])
        if self.show_fits or self.save_final_plot:
            self.plotFullLineImage(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.no_background_sub_full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], self.continuum_interp, self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                    self.identified_full_emission_line_centers, full_emission_heights, self.continuum_fit_points, self.wavelength_of_mu_solution,
                                    self.show_fits, self.save_final_plot, plot_title = plot_title,
                                    save_image_name = self.stacked_image_name + self.processed_spectra_image_suffix + self.figure_suffix)
        return 1

    def loadSpecProcessor(self, load_name, load_dir = None):
        """
        Note the order here is important, and needs to be kept consistent with the order
           in which things are written in the saveSpecProcessor function.
        """
        if load_dir == None:
            load_dir = self.target_dir
        load_object = open(load_dir + load_name, "r")
        l_wavelength_poly_terms = load_object.readline().split(' ')
        self.wavelength_poly_terms = [float(term) for term in l_wavelength_poly_terms[1:]]
        self.mu_of_wavelength_solution = None
        self.wavelength_of_mu_solution = None
        l_anchor_pix = load_object.readline().split(' ')[1:]
        l_anchor_pix = l_anchor_pix[-1][:-1]
        self.anchor_parallel_pixel = float(l_anchor_pix[1])
        l_spec_range = load_object.readline().split(' ')
        l_spec_range[-1] = l_spec_range[-1][:-1]
        self.spec_range = [float(term) for term in l_spec_range[1:]]
        l_parallel_spec_pixels = load_object.readline().split(' ')
        l_parallel_spec_pixels [-1] = l_parallel_spec_pixels [-1][0:-1]
        self.parallel_spec_pixels = [float(val) for val in l_parallel_spec_pixels[1:]]
        l_full_pix_spectrum = load_object.readline().split(' ')
        l_full_pix_spectrum[-1] = l_full_pix_spectrum[-1][0:-1]
        self.full_pix_spectrum = [float(val) for val in l_full_pix_spectrum[1:]]
        l_full_pix_background_stats = load_object.readline().split(' ')
        l_full_pix_background_stats[-1] = l_full_pix_background_stats[-1][0:-1]
        self.full_pix_background_stats = [float(val) for val in l_full_pix_background_stats[1:]]
        l_parallel_spec_wavelengths = load_object.readline().split(' ')
        l_parallel_spec_wavelengths[-1] = l_parallel_spec_wavelengths[-1][0:-1]
        self.parallel_spec_wavelengths = [float(val) for val in l_parallel_spec_wavelengths[1:]]
        l_continuum_fit_points = load_object.readline().split(' ')
        #l_continuum_fit_points[-1] = l_continuum_fit_points[-1][0:-1]
        #print ('l_continuum_fit_points = ' + str(l_continuum_fit_points))
        #self.continuum_fit_points = [[val for val in can.recursiveStrToListOfLists(point)] for point in l_continuum_fit_points[1:]]
        l_continuum = load_object.readline().split(' ')
        l_continuum[-1] = l_continuum[-1][0:-1]
        self.continuum =  [float(val) for val in l_continuum[1:]]
        l_full_absorbtion_crossings = load_object.readline().split(' ')
        l_full_absorbtion_crossings[-1] = l_full_absorbtion_crossings[-1][0:-1]
        self.full_absorbtion_crossings = [float(val) for val in l_full_absorbtion_crossings[1:] ]
        l_full_emission_crossings = load_object.readline().split(' ')
        l_full_emission_crossings[-1] = l_full_emission_crossings[-1][0:-1]
        self.full_emission_crossings = [float(val) for val in l_full_emission_crossings[1:] ]
        l_seeing_fit_params = load_object.readline().split(' ')
        l_seeing_fit_params [-1] = l_seeing_fit_params [-1][0:-1]
        self.seeing_fit_params = [float(val) for val in l_seeing_fit_params[1:] ]
        l_identified_full_emission_line_centers = load_object.readline().split(' ')
        l_identified_full_emission_line_centers [-1] = l_identified_full_emission_line_centers [-1][0:-1]
        self.identified_full_emission_line_centers = [float(val) for val in l_identified_full_emission_line_centers[1:] ]
        l_spectrum_ids = load_object.readline().split(' ')[1:]
        l_spectrum_ids [-1] = l_spectrum_ids [-1][0:-1]
        self.identified_lines_dict = { }
        for spec_id in l_spectrum_ids:
            self.identified_lines_dict[spec_id] = {self.lines_in_dict_keyword:{}, self.obs_time_keyword:''}
        l_start_exps = load_object.readline().split(' ')
        print ('l_start_exps = ' + str(l_start_exps))
        l_start_exps [-1] = l_start_exps [-1][0:-1]
        for i in range(len(l_spectrum_ids)):
            spec_id = l_spectrum_ids[i]
            self.identified_lines_dict[spec_id][self.obs_time_keyword] = l_start_exps[i+1]
        next_line = load_object.readline().split(' ')
        while len(next_line) > 1 and len(next_line[0]) > 0:
            l_line_profiles = next_line
            line_id = l_line_profiles[0]
            line_profiles = [[float(prof_elem) for prof_elem in can.recursiveStrToListOfLists(prof)] for prof in l_line_profiles[1:]]
            for i in range(len(l_spectrum_ids)):
                spec_id = l_spectrum_ids[i]
                self.identified_lines_dict[spec_id][self.lines_in_dict_keyword][line_id] = line_profiles[i]
            next_line = load_object.readline().split(' ')

        self.seeing_fit_funct = np.poly1d(self.seeing_fit_params)
        self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(self.wavelength_poly_terms)

        return 1

    def saveSpecProcessor(self, save_name, save_dir = None, ):
        if save_dir == None:
            save_dir = self.target_dir
        #What do we want to save? The wavelength solution, and the identified lines directory.
        save_object = open(save_dir + save_name, "w")
        save_object.write('wavelength_poly_terms' + ' ' + ' '.join([str(can.round_to_n(term, 5)) for term in self.wavelength_poly_terms]))
        save_object.write('\n')
        save_object.write('anchor_pix' + ' ' + str(self.anchor_parallel_pix))
        save_object.write('\n')
        save_object.write('spec_range' + ' ' + ' '.join([str(term) for term in self.spec_range]))
        save_object.write('\n')
        save_object.write('parallel_spec_pixels' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.parallel_spec_pixels]))
        save_object.write('\n')
        save_object.write( 'full_pix_spectrum' + ' '  + ' '.join([str(can.round_to_n(val, 5)) for val in self.full_pix_spectrum]) )
        save_object.write('\n')
        save_object.write( 'full_pix_background_stats' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.full_pix_background_stats]))
        save_object.write('\n')
        save_object.write( 'parallel_spec_wavelengths' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.parallel_spec_wavelengths]))
        save_object.write('\n')
        save_object.write( 'continuum_fit_points' + ' ' + ' '.join([str([can.round_to_n(point[0], 5), can.round_to_n(point[1], 5)]).replace(" ", "") for point in self.continuum_fit_points]))
        save_object.write('\n')
        save_object.write( 'continuum' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.continuum]))
        save_object.write('\n')
        save_object.write( 'full_absorbtion_crossings' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.full_absorbtion_crossings]))
        save_object.write('\n')
        save_object.write( 'full_emission_crossings' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.full_emission_crossings]))
        save_object.write('\n')
        save_object.write( 'seeing_fit_params' + ' ' + ' '.join([str(can.round_to_n(val, 5)) for val in self.seeing_fit_params]))
        save_object.write('\n')
        save_object.write('identified_full_emission_line_centers' + ' ' + ' '.join([str(can.round_to_n(center,5)) for center in self.identified_full_emission_line_centers]))
        save_object.write('\n')

        spec_ids = list(self.identified_lines_dict.keys())
        line_ids = list(self.identified_lines_dict[spec_ids[0]][self.lines_in_dict_keyword].keys())
        print ('spec_ids = ' + str(spec_ids))
        save_object.write( 'spectrum_id' + ' ' + ' '.join([str(elem) for elem in spec_ids]) )
        save_object.write('\n')
        save_object.write(self.obs_time_keyword + ' ' + ' '.join([self.identified_lines_dict[spec_id][self.obs_time_keyword] for spec_id in spec_ids]) )
        for i in range(len(self.identified_lines_dict[spec_ids[0]][self.lines_in_dict_keyword])):
            line_id = line_ids[i]
            save_object.write('\n')
            line_fits = [self.identified_lines_dict[spec_id][self.lines_in_dict_keyword][line_id] for spec_id in spec_ids]
            save_object.write(str(line_id) + ' ' + ' '.join([ '[' + ','.join([str(can.round_to_n(fit_elem, 5)) for fit_elem in line_fit]) + ']' for line_fit in line_fits ]) )

        save_object.close()
        return 1

    def initialize_params_from_ref_params(self):
        self.master_bias_file = self.ref_param_holder.getMasterBiasName()
        self.master_dark_file = self.ref_param_holder.getMasterDarkName()
        self.master_bias_list = self.ref_param_holder.getBiasList()
        self.master_dark_file = self.ref_param_holder.getDarkList()
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
        self.obs_time_keyword = self.ref_param_holder.getStartExposureKeyword()
        self.date_format_str = self.ref_param_holder.getDateFormatString()
        self.stacked_image_keyword = self.ref_param_holder.getStackedKeyword()
        self.lines_in_dict_keyword = self.ref_param_holder.getLinesDictKeyword()
        self.wavelength_target_range = self.ref_param_holder.getWavelengthRangeOfInterest()
        self.energy_to_photon_scaling = self.ref_param_holder.getWavelengthToPhotonScaling()
        self.background_cut_wavelengths = self.ref_param_holder.getBackgroundCutWavelengths()
        self.bg_fit_range = self.ref_param_holder.getBackgroundFitRegion()

        self.throughput_interp = None
        self.mu_of_wavelength_solution = None
        self.wavelength_of_mu_solution = None
        self.spec_range = None

        return 1


    def __init__(self, target_dir, master_bias_prefix = 'BIAS', master_dark_prefix = 'DARK', ref_spec = 'KR1',
                 processed_file_suffix = '_proc', processed_spectra_image_suffix = '_spec', perp_spec_image_suffix = '_perp_spec', processed_prefix = 'proc_',
                 data_image_suffix = '.fits', save_image_suffix = '.pdf', list_suffix = '.list', sig_clip_for_line_width = 3.5, save_stacked_image = 1,
                 crc_correct = 1, do_bias = 1, do_dark = 0, cosmic_prefix = 'crc_', show_fits = 1, save_final_plot = 1, save_perp_spec_image = 1, spec_axis = 0,
                 background_buffer = 10, background_size = 100, background_low = 1, n_std_for_strong_line = 20.0, n_std_for_full_line = 10.0,
                 archival_data_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/',
                 ref_params_file = 'OSELOTSDefaults.txt', ref_params_dir = '/Users/sashabrownsberger/Documents/sashas_python_scripts/skySpectrograph/',
                 remove_intermed_images = 1):

        self.ref_param_holder = ref_param.CommandHolder(spectrograph_file = ref_params_file, defaults_dir = ref_params_dir)
        self.initialize_params_from_ref_params()
        self.spec_archival_info = {'KR1':{'spec_file':'KR1LinesSpec.csv','n_spec_lines_to_ignore':1, 'lines_file':'KR1_lines_all.txt', 'n_lines_lines_to_ignore':1},
                              #'Gemini':{'spec_file':'GeminiSkyLines.txt','n_lines_to_ignore':14},
                             'Gemini':{'spec_file':'GeminiSkyBrightness.txt','n_lines_to_ignore':14},
                              'throughput':{'spec_file':'OSELOT_throughput.txt','n_lines_to_ignore':0} }
        #data_image_suffix = self.ref_param_holder.getImageSuffix()
        #list_suffix = self.ref_param_holder.getImageSuffix()
        #master_bias_prefix = self.ref_param_holder.getMasterBiasPrefix()
        #master_dark_prefix = self.ref_param_holder.getMasterDarkPrefix()
        #self.spec_files = spec_files
        self.do_bias = do_bias
        self.do_dark = do_dark
        self.ref_spec = ref_spec
        self.show_fits = show_fits
        self.crc_correct = crc_correct
        self.save_perp_spec_image = save_perp_spec_image
        self.save_stacked_image = save_stacked_image
        self.save_final_plot = save_final_plot
        self.ref_sky_lines_data = can.readInColumnsToList(self.ref_sky_lines_file, self.archival_data_dir, n_ignore = self.ref_sky_lines_file_n_ignore, convert_to_float = 1)
        self.identified_lines_dict = {}
        self.xlims = [450, 1350]
        self.ylims = [-0.2, 1.2]
        self.remove_intermed_images = remove_intermed_images
        self.throughput_interp = self.importSystemThroughput()
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        #spec_files = spec_file.replace('[','')
        #spec_files = spec_files.replace(']','')
        #spec_files = spec_files.split(',')
        self.target_dir = target_dir
        self.seeing_fit_params = np.zeros(self.seeing_fit_order + 1)
        self.seeing_fit_params[-1] = self.init_seeing_guess
        self.seeing_fit_funct = np.poly1d(self.seeing_fit_params)
