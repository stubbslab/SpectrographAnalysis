import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import cantrips as c
import time
from datetime import datetime
import scipy.optimize as optimize
import scipy.special as special
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import cosmics_py3 as cosmics
import sys

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

def CleanCosmics(image_dir, image_names, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = 'crc_'):
    for image_name in image_names:
        #print ('Beginning cosmic ray cleaning for image ' + image_dir + image_name)
        image_array, image_header = cosmics.fromfits(image_dir + image_name)
        c = cosmics.cosmicsimage(image_array, readnoise = readnoise, sigclip = sigclip, sigfrac = sigfrac, objlim = objlim, verbose = False)
        c.run(maxiter = maxiter)
        image_header['CRCLEANED'] = 'Cosmic rays removed by cosmics.py'
        cosmics.tofits(image_dir + new_image_prefix + image_name, c.cleanarray, image_header)
    return 1

def readInRawSpect(target_file, target_dir):
    return c.readInDataFromFitsFile(target_file, target_dir)

def makeMasterBias(master_bias_file, target_dir,
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
    med_bias = c.smartMedianFitsFiles(bias_images, target_dir, bias_x_partitions, bias_y_partitions)[0]
    m_bias_header = c.readInDataFromFitsFile(bias_images[-1], target_dir)[1]
    utc_time = datetime.utcnow()
    m_bias_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
    m_bias_header['NCOMBINE'] = (str(len(bias_images)), 'Number of raw biases stacked.')
    m_bias_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')

    print ('med_bias = ' + str(med_bias))
    #print('med_bias.data = ' + str(med_bias.data))
    #print ('med_bias_header = ' + str(m_bias_header))
    c.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = m_bias_header, overwrite = True, n_mosaic_extensions = 0)
    #c.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = 'default', overwrite = True, n_mosaic_extensions = 0)
    print ('Master bias file created ' + target_dir + master_bias_file)
    return 1

def biasSubtract(image_data, image_header, master_bias_file):
    bias_data, bias_header = c.readInDataFromFitsFile(master_bias_file, target_dir)
    image_data = image_data - bias_data
    image_header['BIASSUB'] = (str(datetime.utcnow() ), 'UTC of Bias subtraction')
    image_header['MBIAS'] = (master_bias_file, 'Name of Subtracted Master Bias File')

    return image_data, image_header

def darkSubtract(image_data, image_header, master_dark_file, exp_time_keyword = 'EXPTIME'):
    dark_data, dark_header = c.readInDataFromFitsFile(master_dark_file, target_dir)
    exp_time = float(image_header[exp_time_keyword])
    image_data = image_data - dark_data * exp_time
    image_header['DARKSUB'] = (str(datetime.utcnow() ), 'UTC of Dark subtraction')
    image_header['MDARK'] = (master_bias_file, 'Name of Subtracted Master Bias File')

    return image_data, image_header

def makeMasterDark(master_dark_file, target_dir, master_bias_file,
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
        single_dark_data, single_dark_header = c.readInDataFromFitsFile(dark_file, target_dir)
        single_dark_data, single_dark_header = biasSubtract(single_dark_data, single_dark_header, master_bias_file)
        exp_times[i] = float(single_dark_header['EXPTIME'])
        c.saveDataToFitsFile(np.transpose(single_dark_data), bias_sub_prefix + dark_file, target_dir, header = single_dark_header, overwrite = True, n_mosaic_extensions = 0)
        m_dark_header = single_dark_header

    #median combine the dark images
    med_dark = c.smartMedianFitsFiles([bias_sub_prefix + dark_image for dark_image in dark_images], target_dir, dark_x_partitions, dark_y_partitions, scalings = [1.0 / time for time in exp_times] )[0]
    if remove_intermediate_files:
        [os.remove(target_dir + bias_sub_prefix + dark_image) for dark_image in dark_images ]
    m_dark_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
    m_dark_header['NCOMBINE'] = (str(len(dark_images)), 'Number of bias-sub darks stacked.')
    m_dark_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')

    c.saveDataToFitsFile(np.transpose(med_dark), master_dark_file, target_dir, header = m_dark_header, overwrite = True, n_mosaic_extensions = 0)
    print ('Master dark file created ' + target_dir + master_dark_file)
    return 1

def determineSpecRowRanges(current_image,
                           spec_axis = 0, sum_method = 'sum', n_sig_deriv_spikes_for_spec = 3.0, n_sig_gauss_to_define_spec_width = 2.0 , sig_step = 0.5, showIDedLines = 1, figsize = [10, 6], save_perp_spec_image = 0, perp_spec_image_name = 'IdentifiedSpecRange.png'):
    perp_spec_axis = (spec_axis + 1) % 2
    perp_spec = np.sum(current_image, axis = perp_spec_axis)
    perp_spec_smooth = c.smoothList(perp_spec, smooth_type = 'boxcar', averaging = 'median', params = [50])
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
        f, axarr = plt.subplots(2,1, figsize = figsize )
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

    return [perp_line_step_up, perp_line_step_down]

def fitSingleLine(fit_xs, fit_ys, init_guess, bounds = ([0.0, 0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]), show_fit = 0):

    fit_funct = lambda xs, A, mu, sig, shift, lin_scale: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift + lin_scale * (np.array(xs) - np.mean(xs))
    if show_fit:
        plt.plot(fit_xs, fit_funct(fit_xs, *init_guess), c = 'red')
    try:
        fit_res = optimize.curve_fit(fit_funct, fit_xs, fit_ys, p0 = init_guess, bounds = bounds)[0].tolist()
        if show_fit:
            plt.plot(fit_xs, fit_funct(fit_xs, *fit_res), c = 'green')
        fit_sum_of_sqrs = np.sum((np.array(fit_ys) - np.array(fit_funct(fit_xs, *fit_res))) ** 2.0)
        mean_sum_of_sqrs = np.sum((np.array(fit_ys) - np.mean(fit_ys)) ** 2.0)
        fit_res = fit_res + [fit_sum_of_sqrs, mean_sum_of_sqrs]
        if fit_res[0] <0.0: print ('[bounds, fit_res] = ' + str([bounds, fit_res]))
    except (RuntimeError, TypeError) :
        #print ('Failed to fit one possible line.')
        return [-1, -1, -1, -1, -1, -1]
    return fit_res


def identifyLinesOnSlice(xs, ys,
                         peak_guesses = [], std_thresh = 3, max_line_fit_width = 20,
                         n_pix_above_thresh = 1, init_fit_width_guess = 2.0, background_bin_width = 20,
                         search_width = 3.0, show_spec = 0, verbose = 1):

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
        print ('n_peak_guesses = ' + str(n_peak_guesses))
    line_fits = [0 for peak in peak_guesses]
    if show_spec:
        print ('Here 1')
        plt.plot(xs, ys, c = 'blue')
        plt.xlabel('Pixel number (column)')
        plt.ylabel('Binned spectrum (ADU)')
        plt.title('Fits to IDed lines on spectrum slice')
    for j in range(0, n_peak_guesses ):
        peak_guess = peak_guesses[j]
        if n_peak_guesses == 1:
            fit_xs = list( range(max(int(peak_guess - max_line_fit_width), xs[0]),
                                 min(int(peak_guess + max_line_fit_width), n_pix)) )
        elif j == 0:
            fit_xs = list( range(max(int(peak_guess - max_line_fit_width), 0),
                                 min(int(peak_guess + max_line_fit_width), int(peak_guess + peak_guesses[j+1]) // 2, xs[-1])) )
        elif j == n_peak_guesses - 1:
            fit_xs = list( range(max(int(peak_guess - max_line_fit_width), int(peak_guesses[j-1] + peak_guesses[j]) // 2, xs[0]),
                             min(int(peak_guess + max_line_fit_width), xs[-1])) )
        else:
            fit_xs = list( range(max(int(peak_guess - max_line_fit_width), int(peak_guesses[j-1] + peak_guess) // 2, xs[0]),
                             min(int(peak_guess + max_line_fit_width), int(peak_guess + peak_guesses[j+1]) // 2, xs[-1])) )
        fit_ys = ys[fit_xs[0]:fit_xs[-1] + 1]
        #print ('[fit_xs, fit_ys] = ' + str([fit_xs, fit_ys]))
        init_guess = [max(fit_ys), peak_guess, init_fit_width_guess, 0.0, 0.0 ]
        lower_bounds = [0.0, init_guess[1] - search_width, 0.2, -np.inf, -np.inf ]
        lower_bounds = [min(lower_bounds[i], init_guess[i]) for i in range(len(init_guess))]
        upper_bounds = [np.inf, init_guess[1] + search_width, xs[-1] - xs[0], init_guess[1] + init_guess[3], np.inf]
        upper_bounds = [max(upper_bounds[i], init_guess[i]) for i in range(len(init_guess))]
        line_fits[j] = fitSingleLine(fit_xs, fit_ys, init_guess, bounds = (lower_bounds, upper_bounds), show_fit = show_spec)

    if show_spec:
        plt.show()
    #print ('line_fits = ' + str(line_fits))
    return line_fits

def extendLinesIntoImage(range_to_extend, line_extensions, data_to_search, ref_line_ends,
                         binning_search_width = 3, spec_axis = 0, max_sep_per_pix = 5,
                         max_frac_intensity_per_pix = 0.1, line_bg_width = 10,
                         max_seq_fit_failures = 3, max_seq_extreme_amps = 5,
                         bound_tolerance = 0.01):

    max_frac_intensity_per_pix = 0.2
    n_failures = [0 for line in line_extensions]
    n_extreme_amps = [0 for line in line_extensions]

    for i in range(len(range_to_extend)):
        pix = range_to_extend[i]
        if spec_axis == 0:
            spec_slice = np.median(data_to_search[max(0, pix - int(binning_search_width/2)):pix + int(binning_search_width/2 + 0.5), :], axis = spec_axis)
        else:
            spec_slice = np.median(data_to_search[:, max(0, pix - int(binning_search_width/2)):pix + int(binning_search_width/2 + 0.5)], axis = spec_axis)

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
def identifyContinuousLines(pix_vals, lines_by_slice, data_to_search,
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
                                                     binning_search_width = binning_search_width, spec_axis = spec_axis,
                                                     max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix, line_bg_width = line_bg_width )

        line_extensions_above = extendLinesIntoImage(range_above, line_extensions_above, data_to_search, [continuous_line[-1] for continuous_line in continuous_lines],
                                                     binning_search_width = binning_search_width, spec_axis = spec_axis,
                                                     max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix, line_bg_width = line_bg_width  )
        continuous_lines = [  c.niceReverse(line_extensions_below[i]) + continuous_lines[i] + line_extensions_above[i] for i in range(len(continuous_lines)) ]

    if show_start_line_fits:
        for continuous_line in continuous_lines:
            plt.plot([point[2] for point in continuous_line], [point[0] for point in continuous_line])
        plt.show()

    continuous_lines = [ line for line in continuous_lines if len(line) >= min_detections ]

    return continuous_lines

def detectLinesCentersInOneD(spec_slice, stat_slice,
                             spec_grad_rounding = 5, n_std_for_line = 5,
                             show = 0, figsize = [14, 8]):
    spec_derivs = np.gradient(spec_slice)
    spec_derivs = np.around(spec_derivs, spec_grad_rounding)

    stat_derivs = np.gradient(stat_slice)
    stat_derivs = np.around(stat_derivs, spec_grad_rounding)
    deriv_median = np.median(stat_derivs)
    deriv_std = np.std(stat_derivs)

    #deriv_median = np.median(spec_derivs)
    #deriv_std = np.std(spec_derivs)
    deriv_emission_crossings = [pix for pix in range(1, len(spec_derivs) - 2) if (spec_derivs[pix] >= 0.0 and spec_derivs[pix + 1 ] < 0.0 )  ]
    deriv_absorbtion_crossings = [pix for pix in range(1, len(spec_derivs) - 2) if (spec_derivs[pix] <= 0.0 and spec_derivs[pix + 1 ] > 0.0 ) ]
    emission_deriv_turns = [[-1, -1] for cross in deriv_emission_crossings]
    absorbtion_deriv_turns = [[-1, -1] for cross in deriv_absorbtion_crossings]
    for j in range(len(deriv_emission_crossings)):
        cross = deriv_emission_crossings[j]
        left_peak = cross
        right_peak = cross+1
        while (left_peak > 0 and spec_derivs[left_peak] >= spec_derivs[left_peak+1]):
            left_peak = left_peak - 1
        left_peak = left_peak + 1
        while (right_peak < len(spec_slice) and spec_derivs[right_peak] < spec_derivs[right_peak-1]):
            right_peak = right_peak + 1
        right_peak = right_peak - 1
        emission_deriv_turns[j] = [left_peak, right_peak]

    for j in range(len(deriv_absorbtion_crossings)):
        cross = deriv_absorbtion_crossings[j]
        left_peak = cross
        right_peak = cross+1
        while (left_peak > 0 and spec_derivs[left_peak] <= spec_derivs[left_peak+1]):
            left_peak = left_peak - 1
        left_peak = left_peak + 1
        while (right_peak < len(spec_slice) and spec_derivs[right_peak] > spec_derivs[right_peak-1]):
            right_peak = right_peak + 1
        right_peak = right_peak - 1
        absorbtion_deriv_turns[j] = [left_peak, right_peak]

    absorbtion_indeces = [(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2 for j in range(len(deriv_absorbtion_crossings))
                                 if ( abs(spec_derivs[absorbtion_deriv_turns[j][0]] - deriv_median) / deriv_std >= n_std_for_line
                                      and abs(spec_derivs[absorbtion_deriv_turns[j][1]] - deriv_median) / deriv_std >= n_std_for_line )
                                ]
    emission_indeces = [(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2 for j in range(len(deriv_emission_crossings))
                               if ( abs(spec_derivs[emission_deriv_turns[j][0]] - deriv_median) / deriv_std >= n_std_for_line
                                    and abs(spec_derivs[emission_deriv_turns[j][1]] - deriv_median) / deriv_std >= n_std_for_line )
                              ]

    if show:
        f, axarr = plt.subplots(2,1, figsize = figsize)
        spec = axarr[0].plot(range(len(spec_slice)), spec_slice, c = 'blue')[0]
        axarr[0].set_ylim(np.min(spec_slice) * 0.9, np.max(spec_slice) * 1.1)
        bg = axarr[0].plot(range(len(spec_slice)), stat_slice, c = 'red')[0]
        em_line = None
        ab_line = None
        for line in emission_indeces: em_line = axarr[0].axvline(line, color = 'green')
        for line in absorbtion_indeces: ab_line = axarr[0].axvline(line, color = 'orange')
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
        spec_deriv = axarr[1].plot(range(len(spec_slice)), spec_derivs, c = 'blue')[0]
        bg_deriv = axarr[1].plot(range(len(stat_slice)), stat_derivs, c = 'red')[0]
        em_line = None
        ab_line = None
        for line in emission_indeces: em_line = axarr[1].axvline(line, color = 'green')
        for line in absorbtion_indeces: ab_line = axarr[1].axvline(line, color = 'orange')
        #axarr[1].legend([spec_deriv, bg_deriv, em_line, ab_line], ['Pixel deriv of binned spectrum', 'Pixel deriv of background','Emission lines', 'Absorbtion lines'])
        axarr[1].legend([spec_deriv, bg_deriv, em_line, ab_line], ['Pixel deriv of binned spectrum', 'Pixel deriv of background','Emission lines', 'Absorbtion lines'])
        axarr[0].set_title('Identified Lines in Binned Spectrum')
        axarr[0].set_xlabel('Column number (pix)')
        axarr[0].set_ylabel('Counts in column (ADU)')
        axarr[1].set_xlabel('Column number (pix)')
        axarr[1].set_ylabel('Pixel deriv of counts in column (ADU/pix)')
        #plt.tight_layout()
        plt.show()

    return absorbtion_indeces, emission_indeces

def traceLinesOverRange(image, search_range,
                        spec_axis = 0, n_std_for_line = 10,
                        coarse_search_binning = 3, fit_binning = 3,
                        max_line_fit_width = 20, parallel_smoothing = 1,
                        n_pix_above_thresh = 1, width_guess = 3,
                        show_process = 0, spec_grad_rounding = 5,
                        draw_stats_below = 1, stat_region_buffer = 10):

    print ('[search_range, coarse_search_binning] = ' + str([search_range, coarse_search_binning]))
    coarse_pix_vals = list(range(int(search_range[0]), int(search_range[1]) , coarse_search_binning))
    coarse_fit_grid = {pix_val:[] for pix_val in coarse_pix_vals}
    if draw_stats_below:
        stat_region = [max(0, search_range[0] - stat_region_buffer - coarse_search_binning), search_range[0] - stat_region_buffer ]
    else:
        stat_region = [search_range[1] + stat_region_buffer,
                       min(np.shape(image)[spec_axis], search_range[1] + stat_region_buffer + coarse_search_binning) ]
    if spec_axis == 0:
        stat_slice = image[stat_region[0]:stat_region[1], :]
    else:
        stat_slice = image[:, stat_region[0]:stat_region[1]]
    if coarse_search_binning > 1:
        stat_slice = np.sum(stat_slice, axis = spec_axis)

    for i in range(len(coarse_pix_vals)):
        pix_val = coarse_pix_vals[i]
        print ('Identifying lines for orthogonal pixel values from ' + str(pix_val) + ' to ' + str(pix_val + coarse_search_binning))
        if spec_axis == 0:
            spec_slice = image[pix_val:pix_val + coarse_search_binning, :]
        else:
            spec_slice = image[:, pix_val:pix_val + coarse_search_binning]
        if coarse_search_binning > 1:
            spec_slice = np.sum(spec_slice, axis = spec_axis)

        spec_slice = np.array(c.smoothList(spec_slice.tolist(), smooth_type = 'boxcar', params = [parallel_smoothing]))
        strong_absorbtion_indeces, strong_emission_indeces = detectLinesCentersInOneD(spec_slice, stat_slice,
                                                                                      spec_grad_rounding = spec_grad_rounding, n_std_for_line = n_std_for_line,
                                                                                      show = show_process)

        coarse_fit_grid[pix_val] = strong_emission_indeces


    print ('coarse_fit_grid = ' + str(coarse_fit_grid))
    coarse_pixels = list(coarse_fit_grid.keys())
    pix_vals = list(range(int(search_range[0]), int(search_range[1] - fit_binning) + 1))
    all_slices = []
    for i in range(len(pix_vals)):
        pix_val = pix_vals[i]
        closest_coarse_pix = coarse_pixels[np.argmin([abs(coarse_pix - pix_val) for coarse_pix in coarse_pixels])]
        guess_line_centers = coarse_fit_grid[closest_coarse_pix]
        if spec_axis == 0:
            spec_slice = image[pix_val:pix_val + fit_binning, :]
        else:
            spec_slice = image[:, pix_val:pix_val + fit_binning]
        if fit_binning > 1:
            spec_slice = np.median(spec_slice, axis = spec_axis)
        #Stuff
        line_fits = identifyLinesOnSlice(range(len(spec_slice)), spec_slice,
                                         peak_guesses = guess_line_centers, std_thresh = n_std_for_line, max_line_fit_width = max_line_fit_width,
                                         n_pix_above_thresh = n_pix_above_thresh, init_fit_width_guess = width_guess,
                                         show_spec = (i % 50 == 0) * show_process, verbose =  (i % 50 == 0))
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

def polyFitVar(ind_vals, dep_vals, fit_order, n_std_for_rejection_in_fit):
    med = np.median(dep_vals)
    std = np.std(dep_vals)
    ind_vals_for_fit = [ind_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
    ind_vals_for_fit_range = [min(ind_vals_for_fit), max(ind_vals_for_fit)]
    dep_vals_to_fit = [dep_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
    var_poly_fit = np.polyfit(ind_vals_for_fit, dep_vals_to_fit, fit_order)
    #var_funct = lambda val: np.poly1d(var_poly_fit)(val) * (val >= ind_vals_for_fit_range[0]) * (val <= ind_vals_for_fit_range[1])
    var_funct = lambda val: np.poly1d(var_poly_fit)(val)

    return var_funct


def getLineFunction(line,
                    spec_axis = 0, n_std_for_rejection_in_fit = 3,
                    position_order = 2, A_of_x_order = 2, sig_of_x_order = 2,
                    n_hist_bins = 21 ):

    if spec_axis == 0:
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

    position_funct = polyFitVar(ind_var, dep_var, position_order, n_std_for_rejection_in_fit)
    A_funct = polyFitVar(ind_var, As, position_order, n_std_for_rejection_in_fit)
    sig_funct = polyFitVar(ind_var, sigs, position_order, n_std_for_rejection_in_fit)

    return [ind_var, position_funct, A_funct, sig_funct]

def fixContinuousLines(lines, line_profiles, n_bins = 21, n_empty_bins_to_drop = 2, n_hist_bins = 21, show_line_matches = 1):
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
            f, axarr = plt.subplots(3,1)
            bin_peaks, bin_edges, blanck = axarr[0].hist(center_offsets, bins = n_hist_bins)
            plt.close('all')
        bin_centers = [ (bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(bin_edges) - 1) ]
        peak_index = np.argmax(bin_peaks)
        center_bounds = [bin_edges[0], bin_edges[-1]]
        for j in c.niceReverse(list(range(0, peak_index))):
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


def detectLinesInImage(current_image, spec_range,
                       spec_axis = 0, n_std_for_lines = 3.0,
                       search_binning = 'full', fit_binning = 10,
                       max_line_fit_width = 20, parallel_smoothing = 1,
                       width_guess = 1, show_process = 0,
                       max_sep_per_pix = 5.0, max_frac_intensity_per_pix = 0.1,
                       min_detections = 10, fit_line_center_index = 1,
                       search_for_extensions = None, bg_fit_width = 10,
                       draw_stats_below = 1, buffer_for_line_background_stats = 10):
    if search_binning in ['full','FULL','Full']:
        search_binning = spec_range[1] - spec_range[0]

    pix_slices, lines_by_slice = traceLinesOverRange(current_image, spec_range,
                                                     spec_axis = spec_axis, n_std_for_line = n_std_for_lines,
                                                     coarse_search_binning = search_binning, fit_binning = fit_binning,
                                                     max_line_fit_width = max_line_fit_width, parallel_smoothing = parallel_smoothing,
                                                     n_pix_above_thresh = 1, width_guess = width_guess,
                                                     show_process = show_process,
                                                     draw_stats_below = draw_stats_below, stat_region_buffer = buffer_for_line_background_stats)

    lines = identifyContinuousLines(pix_slices, lines_by_slice, current_image,
                                    max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix,
                                    min_detections = min_detections, fit_line_center_index = fit_line_center_index,
                                    image_range_to_search = search_for_extensions, binning_search_width = search_binning,
                                    line_bg_width = bg_fit_width, show_start_line_fits = show_process)

    return lines

def readInDataTextFile(ref_spec_file,
                       spec_file_dir = '', n_ignore = 0,
                       throughput_file = 'default_throughput.txt'):
    ref_spec = c.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore, convert_to_float = 1)
    #plt.plot(*ref_spec)
    #plt.show()

    return ref_spec

#We want to return the integrated distance (in pixel space) between detected lines
# and their wavelengths that they might correspond to, given the wavelength solution.
# We cannot have a single wavelength match two detections (or vice versa) so we perform the
# matching one line at a time, and remove the match from consideration after each match is done.
def lineMatchingFunction(line_pixels, line_wavelengths, n_matches, wavelength_solution, verbose = 0):
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


#Match, lines by line
def matchLines(line_pixels_to_be_matched, line_wavelengths_to_match, mu_of_wavelength_funct,
               max_sep_pix = 5.0):
    line_matches = []
    line_pixel_match_indeces = []
    line_wavelength_match_indeces = []
    for i in range(len(line_pixels_to_be_matched)):
        line_pixel_to_be_matched = line_pixels_to_be_matched[i]
        line_pixels_to_match = mu_of_wavelength_funct(line_wavelengths_to_match)
        line_to_be_matched = mu_of_wavelength_funct(line_pixel_to_be_matched)
        line_seps = [abs(line_pixel_to_match - line_pixel_to_be_matched) for line_pixel_to_match in line_pixels_to_match]
        min_sep_index = np.argmin(line_seps)
        min_sep = line_seps[min_sep_index]
        if min_sep <= max_sep_pix:
            line_matches = line_matches + [[line_pixel_to_be_matched, line_wavelengths_to_match[min_sep_index]]]
            line_pixel_match_indeces = line_pixel_match_indeces + [i]
            line_wavelength_match_indeces = line_wavelength_match_indeces + [min_sep_index]
    line_sep = np.sum([np.abs(line_match[0] - line_match[1]) for line_match in line_matches])
    return [line_pixel_match_indeces, line_wavelength_match_indeces, line_matches, line_sep]

def readFileIntoInterp(target_file, target_dir, n_ignore, convert_to_float = 1):
    cols = c.readInColumnsToList(target_file, file_dir = target_dir, n_ignore = n_ignore, convert_to_float = convert_to_float)
    interp = c.safeInterp1d(*cols)
    return interp


#Curve fitting doesn't work very well.  Is there a way that we could just detect where lines are and determine where they are supposed to be?
def determineWavelengthSolution(line_solutions, line_range, spec_range, ref_spec_file, line_median_areas,
                                spec_file_dir = '', throughput_file = 'default_throughput.txt',
                                n_ignore_spec = 0, n_ignore_throughput = 0,
                                wavelength_solution_order = 1, wavelength_solution_drift_order = 2,
                                coarse_search_param_range= [[-400.0, -100.0], [1.0, 1.5]], coarse_search_param_step = [26, 26],
                                wavelength_solution_file = 'MISSING_NAME.fits', solution_save_dir = '',
                                save_solution = 1, show_solution = 1 ):
    ref_spec_lines = c.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore_spec, convert_to_float = 1)

    throughput_interp = readFileIntoInterp(throughput_file, spec_file_dir, n_ignore_throughput, convert_to_float = 1)
    ref_spec_lines[1] = (np.array(ref_spec_lines[1]) * throughput_interp(ref_spec_lines[0])).tolist()

    print ('ref_spec_lines = ' + str(ref_spec_lines))
    slice_pixels = range(*spec_range)
    fitted_spectra = [[0.0 for pix in slice_pixels] for guess in range(wavelength_solution_order + 1)]
    best_match_params1 = [0.0, 0.0]
    best_match_params = [0.0, 0.0]
    best_matches = []
    best_match_sep = np.inf
    best_match_pixel_indeces = []
    best_match_wavelength_indeces = []

    wavelength_of_mu = lambda mu, lam0, lam1, lam2: lam0 + lam1 * mu + lam2 * mu ** 2.0
    wavelength_of_mu = lambda mu, lam0, lam1: lam0 + lam1 * mu

    median_line_centers = [np.median([line_solution[1](pix) for pix in slice_pixels]) for line_solution in line_solutions]

    #there is one light that I expect to be the brightest.  We use that to anchor the fit.
    peak_area_index = np.argmax(line_median_areas)
    peak_line_center = median_line_centers[peak_area_index]
    peak_wavelength = 882.97
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

    wavelength_of_mu = lambda mu, lam0, lam1: lam0 + lam1 * mu
    mu_of_wavelength = lambda lam, mu0, mu1: mu0 + mu1 * lam
    for test_mu0 in np.linspace(*(coarse_search_param_range[0]), coarse_search_param_step[0]):
        for test_mu1 in np.linspace(*(coarse_search_param_range[1]), coarse_search_param_step[1]):
            #print ('[test_lam0, test_lam1] = ' + str([test_lam0, test_lam1]))
            matched_line_pixel_indeces, matched_line_wavelength_indeces, matched_lines, matched_lines_sep = matchLines(np.array(median_line_centers), np.array(ref_spec_lines[0]), lambda pixels: mu_of_wavelength(pixels, test_mu0, test_mu1) )
            if (len(matched_lines) > len(best_matches)) or (len(matched_lines) == len(best_matches) and matched_lines_sep < best_match_sep):
                best_match_params = [test_mu0, test_mu1]
                best_matches = matched_lines
                best_match_sep = matched_lines_sep
                best_match_pixel_indeces = matched_line_pixel_indeces
                best_match_wavelength_indeces = matched_line_wavelength_indeces

    if show_solution:
        f, axarr = plt.subplots(3,1)
        axarr[0].scatter(median_line_centers, [0 for line in median_line_centers], c = 'b')
        axarr[0].scatter([median_line_centers[index] for index in best_match_pixel_indeces], [0 for index in best_match_pixel_indeces], c = 'g')
        [ axarr[0].annotate(i, (median_line_centers[best_match_pixel_indeces[i]],0) ) for i in range(len(best_match_pixel_indeces)) ]
        axarr[2].scatter(ref_spec_lines[0], [0 for line in ref_spec_lines[0]], c = 'b')
        axarr[2].scatter([ref_spec_lines[0][index] for index in best_match_wavelength_indeces], [0 for index in best_match_wavelength_indeces], c = 'c')
        [ axarr[2].annotate(i, (ref_spec_lines[0][best_match_wavelength_indeces[i]],0) ) for i in range(len(best_match_wavelength_indeces)) ]
        plt.show()
    print ('[best_match_pixel_indeces, best_match_wavelength_indeces, best_match_params, best_matches, best_match_sep] = ' + str([best_match_pixel_indeces, best_match_wavelength_indeces, best_match_params, best_matches, best_match_sep]))
    spec_pixels = np.arange(*line_range)
    for i in range(spec_range[1] - spec_range[0]):
        pix = slice_pixels[i]
        line_centers = [line_solutions[index][1](pix) for index in best_match_pixel_indeces]
        #print ('matched_lines = ' + str(matched_lines))

        wavelength_solution = np.polyfit([match[1] for match in best_matches], line_centers, wavelength_solution_order)
        for pix_order in range(wavelength_solution_order + 1):
            fitted_spectra[pix_order][i] = wavelength_solution[pix_order]
        wavelength_funct = np.poly1d(wavelength_solution)

        if pix % 50 == 51:
            print ('[best_match_params, best_matches, best_match_sep, wavelength_solution] = ' + str([best_match_params, best_matches, best_match_sep, wavelength_solution]))
            plt.scatter([match[0] for match in best_matches], [match[1] for match in best_matches])
            plt.plot([match[0] for match in best_matches], np.array([match[0] for match in best_matches]) * best_match_params[1] +  best_match_params[0], c = 'r')
            plt.plot([match[0] for match in best_matches], wavelength_funct([match[0] for match in best_matches]), c = 'g')
            plt.show()
    #wavelength_solution_functions = [[fitted_spectra[i][0]]]
    if show_solution:
        f, axarr = plt.subplots(2,1)
        axarr[0].scatter(slice_pixels, fitted_spectra[0])
        axarr[1].scatter(slice_pixels, fitted_spectra[1])
        axarr[0].set_title('Wavelength solution')
        plt.show()
    wavelength_polyfit = [np.polyfit(slice_pixels, fitted_spectra_term, wavelength_solution_drift_order) for fitted_spectra_term in fitted_spectra]
    print ('wavelength_polyfit = ' + str(wavelength_polyfit))

    if save_solution:
        np.save(archival_data_dir + wavelength_solution_file,  wavelength_solution)

    mu_of_wavelength_solution, wavelength_of_mu_solution = createWavelengthSolutionCallableFunctions(wavelength_solution)
    print ('here1')

    return [mu_of_wavelength_solution, wavelength_of_mu_solution]

def create2DWavelengthSolutionCallableFunctions(wavelength_polyfit):

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

def loadWavelengthSolution(solution_file, load_dir = ''):
    #wavelength_poly_terms = np.load(archival_data_dir + solution_file)
    wavelength_poly_terms = [term_list[0] for term_list in c.readInColumnsToList(solution_file, file_dir = archival_data_dir, n_ignore = 0, convert_to_float = 1)]
    print ('wavelength_poly_terms = ' + str(wavelength_poly_terms))
    #mu_of_wavelength_solution, wavelength_of_mu_solution = create2DWavelengthSolutionCallableFunctions(wavelength_poly_terms)
    mu_of_wavelength_solution, wavelength_of_mu_solution = createWavelengthSolutionCallableFunctions(wavelength_poly_terms)
    return mu_of_wavelength_solution, wavelength_of_mu_solution

def createWavelengthSolutionCallableFunctions(wavelength_poly_terms):
    print ('wavelength_poly_terms = ' + str(wavelength_poly_terms))
    mu_of_wavelength_solution = lambda lam: np.poly1d(wavelength_poly_terms)(lam)
    wavelength_of_mu_solution = lambda mu: ((np.poly1d(wavelength_poly_terms) - mu).roots)[0]
    return [mu_of_wavelength_solution, wavelength_of_mu_solution]


def determineMacroLineCurvatureDict(spec_range, strong_line_profiles, pix_index = 0, mu_index = 1, curvature_fit_order = 2):
    anchor_parallel_pix = ( spec_range[1] + spec_range[0] ) // 2
    anchor_line_perp_pixels = [strong_line_profile[mu_index](anchor_parallel_pix) for strong_line_profile in strong_line_profiles]
    line_anchor_funct_dict = {}
    for spec_parallel_pix in range(spec_range[0], spec_range[1]):
        line_perp_pixels = [strong_line_profile[mu_index](spec_parallel_pix) for strong_line_profile in strong_line_profiles]
        undo_curve_fit = np.polyfit(line_perp_pixels, anchor_line_perp_pixels, curvature_fit_order)
        undo_curve_funct = np.poly1d(undo_curve_fit)
        line_anchor_funct_dict[spec_parallel_pix] = undo_curve_funct

    return line_anchor_funct_dict


#line_profile_dict takes in a pixel y and then a pixel x to determine the pixel x position where a line at that xy would trace to at y = 0 (center of the spectrum)
def measureFullPixelSpectrum(current_image, spec_range, undo_line_curvature_dict,
                             mu_index = 1, width_index = 3, width_fit_order = 2 ):
    spec_pixels = list(range(*spec_range))
    intensities = [[[], []] for i in range(len(spec_pixels))]
    spec_slice_pixels = list(range(np.shape(current_image)[(spec_axis + 1) % 2]))

    for i in range(len(spec_pixels)):
        pix = spec_pixels[i]
        #print ('Computing intensity for pix = ' + str(pix))
        sys.stdout.write("\r{0}".format('Computing intensity for pixel ' + str(pix) + ' (' + str(int (i / len(spec_pixels) * 100)) + '% done)...' ))
        sys.stdout.flush()
        if spec_axis == 0:
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
    intensity_interps = [c.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
    #full_spec_interp = c.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
    full_spec_interp = c.safeInterp1d( spec_slice_pixels, [np.sum([interp(slice_pixel) for interp in intensity_interps]) for slice_pixel in spec_slice_pixels] )

    return spec_slice_pixels, full_spec_interp



def deconvolveSpectrum(current_image, spec_range, mu_of_wavelength_solution, strong_line_profiles, throughput_file, throughput_dir, n_ignore_throughput,
                       mu_index = 1, width_index = 3, width_fit_order = 2, spec_axis = 0, wavelength_range = [300.0, 1100.0], wavelength_step = 0.5 ):
    throughput_interp = readFileIntoInterp(throughput_file, throughput_dir, n_ignore_throughput, convert_to_float = 1) # Throughput(wavelength)

    spec_pixels = list(range(*spec_range))
    intensities = [[[], []] for i in range(len(spec_pixels))]
    #for i in range(len(spec_pixels)):
    for i in range(len(spec_pixels)):
        pix = spec_pixels[i]
        #print ('Computing intensity for pix = ' + str(pix))
        sys.stdout.write("\r{0}".format('Computing intensity for pixel ' + str(pix) + ' (' + str(int (i / len(spec_pixels) * 100)) + '% done)...' ))
        sys.stdout.flush()
        if spec_axis == 0:
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
    intensity_interps = [c.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
    full_spec_interp = c.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
    doconvolved_lines = 1
    f, axarr = plt.subplots(2,1)
    axarr[0].plot(np.arange(*wavelength_range, wavelength_step), intensity_interps[len(spec_pixels) // 2](np.arange(*wavelength_range, wavelength_step)))
    axarr[1].plot(np.arange(*wavelength_range, wavelength_step), full_spec_interp(np.arange(*wavelength_range, wavelength_step)))
    plt.show()
    return 1

def stackImage(current_images, current_headers):
    current_image = np.median(current_images, axis = 0)
    current_header = current_headers[0]
    current_header['STACMETH'] = ( 'MEDIAN', 'Method of stacking individual spectra')
    current_header['NSTACK'] = ( str(len(current_images)), 'Number of individual stacked spectra')
    current_header['EXPTIME'] = (str(sum([float(current_header['EXPTIME']) for current_header in current_headers])), 'Total exposure time of all stacked spectra')
    return current_image, current_header

def correctBackground(image, spec_range, background_buffer, background_size,
                      background_low = 1, spec_axis = 0):
    if background_low:
        background_range = [max(0, spec_range[0] - (background_buffer + background_size)),
                            min(np.shape(current_image)[(spec_axis + 1)%2], spec_range[0] - background_buffer )]
    else:
        background_range = [max(0, spec_range[1] + background_buffer ),
                            min(np.shape(current_image)[(spec_axis + 1)%2], spec_range[1] + background_size + background_buffer )]
    if spec_axis == 0:
        background = image[background_range[0]:background_range[1], :]
    else:
        background = image[:, background_range[0]:background_range[1]]
    if background_size > 1:
        background = np.median(background, axis = spec_axis)

    for pix in range(*spec_range):
        if spec_axis == 0:
            image[pix, :] = image[pix, :] - background
        else:
            image[:, pix] = image[:, pix] - background
    return image


if __name__ == "__main__":

    args = sys.argv[1:]
    crc_correct = 1
    do_bias = 1
    do_dark = 0
    cosmic_prefix = 'crc_'
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    show_fits = 1
    save_final_plot = 1
    save_perp_spec_image = 1

    spec_files, target_dir, master_bias_file, master_dark_file, processed_file, processed_spectra, perp_spec_image_name = args
    spec_files = spec_files.replace('[','')
    spec_files = spec_files.replace(']','')
    spec_files = spec_files.split(',')
    print ('[spec_files, target_dir, master_bias_file, master_dark_file, processed_file, processed_spectra, perp_spec_image_name ] = ' + str([spec_files, target_dir, master_bias_file, master_dark_file, processed_file, processed_spectra,  perp_spec_image_name ]))

    current_images = [[] for spec_file in spec_files]
    current_headers = [[] for spec_file in spec_files]
    if crc_correct:
        CleanCosmics(target_dir, spec_files, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = cosmic_prefix)
        spec_files = [cosmic_prefix + spec_file for spec_file in spec_files]
    for i in range(len(spec_files)):
        print ('Reading in raw spectrum from ' + target_dir + spec_files[i])
        current_images[i], current_headers[i] = readInRawSpect(spec_files[i] , target_dir) #Read in raw spectrum


    processed_prefix = 'proc_'

    #Overscan correct (?) ## Not currently set up to do this

    #[OPTIONAL] Make master bias
    if do_bias or do_dark:
        master_bias_exists = os.path.isfile(target_dir + master_bias_file)
        if not(master_bias_exists):
            print ('Making master bias file.  Will be saved to ' + target_dir + master_bias_file)
            master_bias_exists = makeMasterBias(master_bias_file, target_dir)
        if not(master_bias_exists):
            print ('Unable to find master bias file, ' + target_dir + master_bias_file + ', and also could not make it.  Returning without processing.')
            sys.exit()

    #[OPTIONAL] Make master dark
    if do_dark :
        master_dark_exists = os.path.isfile(target_dir + master_dark_file)
        if not(master_dark_exists):
            print ('Making master dark file.  Will be saved to ' + target_dir + master_dark_file)
            master_dark_exists = makeMasterDark(master_dark_file, target_dir, master_bias_file)
        if not(master_dark_exists):
            print ('Unable to find master dark file, ' + target_dir + master_dark_file + ', and also could not make it.  Returning without processing.')
            sys.exit()

    #Bias Subtract
    for i in range(len(spec_files)):
        if do_bias or do_dark:
            current_images[i], current_headers[i] = biasSubtract(current_images[i], current_headers[i], master_bias_file)
        if do_dark:
            current_images[i], current_headers[i] = darkSubtract(current_images[i], current_headers[i], master_dark_file)

    current_image, current_header = stackImage(current_images, current_headers)
    current_image = np.median(current_images, axis = 0)
    current_header = current_headers[0]


    spec_axis = 0
    spec_range = determineSpecRowRanges(current_image, spec_axis = spec_axis, showIDedLines = show_fits, save_perp_spec_image = save_perp_spec_image, perp_spec_image_name = target_dir + perp_spec_image_name)
    if spec_range[0] < 0:
        print ('Unable to identify spectrum.  Exiting...')
        sys.exit()

    print ('spec_range = ' + str(spec_range))

    background_buffer = 10
    background_size = 100
    background_low = 1
    current_image = correctBackground(current_image, spec_range, background_buffer, background_size,
                                      background_low = background_low, spec_axis = spec_axis)

    c.saveDataToFitsFile(np.transpose(current_image), processed_file, target_dir, header = current_header, overwrite = True, n_mosaic_extensions = 0)
    print ('Just saved processed file to: ' + target_dir + processed_file)

    ghosts_n_sig_width = 10
    background_fit_order = 2
    ghosts_high = 1
    ghosts_right = 0
    min_ghosts = 3
    removeGhostByShiftingSpectrum = 0
    clean_buffer = 5
    n_std_for_most_ghosts = 1.0
    n_std_for_first_ghosts = 4.0
    min_detections_for_ident_as_line = 50
    ghost_search_buffer = [10, 50]
    n_std_for_strong_line = 5.0
    n_std_for_all_lines = 2.5
    strong_line_search_binning = 'full'
    strong_line_fit_binning = 20
    max_line_fit_width = 30.0
    line_width_guess = 1.0
    max_sep_per_pixel_for_line_trace = 5.0
    parallel_smoothing = 3
    show_strong_lines = show_fits
    line_mean_fit_order = 2
    line_amplitude_fit_order = 2
    line_width_fit_order = 2
    draw_background_line_stats_from_below = 1
    buffer_for_line_background_stats = 20
    #wavelength_of_pix_solution_guess = [287, 0.83, 0.0] # based on historic, hand-determine solutions
    wavelength_of_pix_solution_guess = [291.0, 0.83] # based on historic, hand-determine solutions

    strong_lines = detectLinesInImage(current_image, spec_range,
                                      spec_axis = spec_axis, n_std_for_lines = n_std_for_strong_line,
                                      search_binning = strong_line_search_binning, fit_binning = strong_line_fit_binning,
                                      max_line_fit_width = max_line_fit_width, parallel_smoothing = parallel_smoothing,
                                      width_guess = line_width_guess, show_process = show_strong_lines,
                                      max_sep_per_pix = max_sep_per_pixel_for_line_trace, min_detections = min_detections_for_ident_as_line ,
                                      draw_stats_below = draw_background_line_stats_from_below, buffer_for_line_background_stats = buffer_for_line_background_stats )
    #print ('strong_lines = ' + str(strong_lines))
    line_pix_vals = c.union([[line_slice[0] for line_slice in line] for line in strong_lines])
    line_range = [min(line_pix_vals), max(line_pix_vals)]
    for i in range(len(strong_lines)):
        line = strong_lines[i]
        negative_part = 0

    strong_line_profiles = [getLineFunction(line,
                                            spec_axis = spec_axis, position_order = line_mean_fit_order,
                                            A_of_x_order = line_amplitude_fit_order, sig_of_x_order = line_width_fit_order)
                            for line in strong_lines]
    strong_lines = fixContinuousLines(strong_lines, strong_line_profiles )
    strong_line_profiles = [getLineFunction(line,
                                            spec_axis = spec_axis, position_order = line_mean_fit_order,
                                            A_of_x_order = line_amplitude_fit_order, sig_of_x_order = line_width_fit_order)
                            for line in strong_lines]
    if show_fits:
        for strong_line in strong_lines:
            plt.plot([line_slice[2] for line_slice in strong_line], [line_slice[0] for line_slice in strong_line])
        plt.xlabel('Column number (pix)')
        plt.ylabel('Line number (pix)')
        plt.title('Profiles of identified lines')
        plt.show()

    #print ('len(strong_line_profiles) = ' + str(len(strong_line_profiles)))
    #print ('len(line_pix_vals) = ' + str(len(line_pix_vals)))
    #print ('strong_line_profiles = ' + str(strong_line_profiles))
    line_fit_mean_areas = [integrate.quad(lambda x: strong_line_profile[2](x) * strong_line_profile[3](x), strong_line_profile[0][0], strong_line_profile[0][-1])[0] / (strong_line_profile[0][-1] - strong_line_profile[0][0])
                     for strong_line_profile in strong_line_profiles ]
    if show_fits:
        f, axarr = plt.subplots(2,2)
        for i in range(len(strong_line_profiles)):
            strong_line_profile = strong_line_profiles[i]
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
    undo_line_curvature_dict = determineMacroLineCurvatureDict (spec_range, strong_line_profiles)
    #print ('undo_line_curvature_dict = ' + str(undo_line_curvature_dict))
    print ('Binning spectrum, along curvature of lines...')
    parallel_spec_pixels, full_pix_interp = measureFullPixelSpectrum(current_image, spec_range, undo_line_curvature_dict)
    #We also do the same thing on a region above or below to determine the background statistics for our binned spectrum
    if draw_background_line_stats_from_below:
        stat_region = [max(0, spec_range[0] - buffer_for_line_background_stats - (spec_range[1] - spec_range[0]) ), spec_range[0] - buffer_for_line_background_stats ]
    else:
        stat_region = [spec_range[1] + buffer_for_line_background_stats,
                       min(np.shape(image)[spec_axis], search_range[1] + buffer_for_line_background_stats + (spec_range[1] - spec_range[0])) ]
    #print ('stat_region = ' + str(stat_region))
    print ('Binning background in the same way as the spectrum...')
    background_inserted_spectrum = np.copy(current_image)
    if spec_axis == 0:
        background_inserted_spectrum[spec_range[0]:spec_range[1], :] = current_image[stat_region[0]:stat_region[1], :]
    else:
        background_inserted_spectrum[:, spec_range[0]:spec_range[1]] =  current_image[:, stat_region[0]:stat_region[1]]

    stat_slice_pixels, stat_slice_interp = measureFullPixelSpectrum(background_inserted_spectrum, spec_range, undo_line_curvature_dict)

    full_pix_spectrum = full_pix_interp(parallel_spec_pixels)
    full_pix_background_stats = stat_slice_interp(stat_slice_pixels)
    if show_fits:
        spec = plt.plot(parallel_spec_pixels, full_pix_spectrum)[0]
        background = plt.plot(stat_slice_pixels, full_pix_background_stats, c = 'r')[0]
        plt.title('Re-measured spectrum, binned according to strong line fits')
        plt.xlabel('Start of bin line (pix)')
        plt.ylabel('Total count binned along fit (ADU)')
        plt.legend([spec, background], ['Spectrum - bg binned along fit', 'Background binned along fit'])
        plt.show()
    #print ('[full_pix_spectrum, full_pix_background_stats] = ' + str([full_pix_spectrum, full_pix_background_stats]))

    full_absorbtion_crossings, full_emission_crossings = detectLinesCentersInOneD(full_pix_spectrum, full_pix_background_stats, spec_grad_rounding = 5, n_std_for_line = n_std_for_all_lines, show = show_fits)
    print ('[full_absorbtion_crossings, full_emission_crossings] = ' + str([full_absorbtion_crossings, full_emission_crossings]))
    if show_fits:
        f, axarr = plt.subplots(1,1)
        axarr.plot(range(len(full_pix_spectrum )), full_pix_spectrum, c = 'blue')
        axarr.plot(range(len(full_pix_spectrum )), full_pix_background_stats, c = 'red')
        for line in full_emission_crossings: axarr.axvline(line, color = 'green')
        #for line in full_absorbtion_crossings: axarr.axvline(line, color = 'orange')
        #axarr[1].plot(range(len(full_pix_spectrum)), spec_derivs, c = 'blue')
        #for line in strong_emission_indeces: axarr[1].axvline(line, color = 'green')
        #for line in strong_absorbtion_indeces: axarr[1].axvline(line, color = 'orange')
        plt.show()

    #detectLinesCentersInOneD(spec_slice,
    #                         spec_grad_rounding = spec_grad_rounding, )

    #Now that we have our best intensity vs pixel curve, we can more aggresively extract lines for wavelength matching


    coarse_line_match_param_range = [[-600.0, -300.0], [1.0, 1.5]]
    coarse_line_match_param_step = [31, 26]

    archival_data_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/'
    save_wavelength_solution = 1
    spec_archival_info = {'KR1':{'spec_file':'KR1_lines.txt','n_lines_to_ignore':1},
                          'Gemini':{'spec_file':'GeminiSkyLines.txt','n_lines_to_ignore':14},
                          'throughput':{'spec_file':'OSELOT_throughput.txt','n_lines_to_ignore':0} }

    expected_spectrum = 'Gemini' #KR1, Gemini, ...

    throughput_keyword = 'throughput'
    ref_spec_file, n_ignore_spec = [spec_archival_info[expected_spectrum]['spec_file'], spec_archival_info[expected_spectrum]['n_lines_to_ignore']]
    throughput_file, n_ignore_throughput = [spec_archival_info['throughput']['spec_file'], spec_archival_info['throughput']['n_lines_to_ignore']]

    # should be like: peak of line profile, mu, as a function of lambda and y: mu(lam, y)
    wavelength_solution_file = 'bd_crc_sky_2019_06_22_92_1d_byHand_wavelength_solution.txt'
    wavelength_solution_file = None
    if wavelength_solution_file is None:
        mu_of_wavelength_solution, wavelength_of_mu_solution = determineWavelengthSolution(strong_line_profiles, line_range, spec_range, ref_spec_file, line_fit_mean_areas,
                                                                                           spec_file_dir = archival_data_dir, throughput_file = throughput_file,
                                                                                           n_ignore_spec = n_ignore_spec, n_ignore_throughput = n_ignore_throughput,
                                                                                           coarse_search_param_range= coarse_line_match_param_range, coarse_search_param_step =  coarse_line_match_param_step,
                                                                                           show_solution = show_fits,
                                                                                           wavelength_solution_file = c.removeFileTagFromString(processed_file) + '_2d_wavelength_solution.npy', solution_save_dir = archival_data_dir, save_solution = save_wavelength_solution )
    else:
        mu_of_wavelength_solution, wavelength_of_mu_solution = loadWavelengthSolution(wavelength_solution_file, load_dir = archival_data_dir)

    print ('[mu_of_wavelength_solution, wavelength_of_mu_solution] = ' + str([mu_of_wavelength_solution, wavelength_of_mu_solution]))
    #wavelength_of_mu_solution = np.interp1d([], )

    #wavelength_range = [wavelength_of_mu_solution(0, (spec_range[1] - spec_range[0]) // 2), wavelength_of_mu_solution(1023, (spec_range[1] - spec_range[0]) // 2)]
    wavelength_range = [wavelength_of_mu_solution(0), wavelength_of_mu_solution(1023)]
    print('wavelength_range = ' + str(wavelength_range))
    if show_fits or save_final_plot:
        plt.plot([wavelength_of_mu_solution(pix) for pix in parallel_spec_pixels], full_pix_spectrum)
        plt.xlabel(r'Sky Wavelength (nm)')
        plt.ylabel(r'Counts in spectrum (ADU)')
        plt.title('Final spectrum')
    if save_final_plot:
        plt.savefig(target_dir + processed_spectra)
    if show_fits:
        plt.show()
    # deconvolved_spectrum = deconvolveSpectrum(parallel_spec_pixels, full_pix_spectrum, wavelength_of_mu_solutions, strong_line_profiles, throughput_file, archival_data_dir, n_ignore_throughput)

    sys.exit()


    #strong_line_curves =


    #current_image = correctGhosts(current_image, spec_range, ghosts_high = 1, spec_axis = spec_axis)

    c.saveDataToFitsFile(np.transpose(current_image), 'test_ghost_sub' + processed_file, target_dir, header = current_header, overwrite = True, n_mosaic_extensions = 0)

    #subtract shifted spectrum from current spectrum

    #Find the strongest lines that likely correspond to the ghosts
    #raw_linesstrongest_lines = determineStrongestLines(current_image, spec_range, ghost_traces)



    #Determine wavelength solution

    #Correct the ghosts

    #Determine the spectrum row range

    #Bin the spectrum, to get counts per pixel

    #Apply a wavelength solution, to get counts vs wavelength

    #Convert from counts per wavelength to flux per wavelength

    #Save to a text file
