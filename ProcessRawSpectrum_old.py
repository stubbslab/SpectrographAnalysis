import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os 
import cantrips as c
import time
from datetime import datetime
import scipy.optimize as optimize
import scipy.interpolate as interpolate 

def generateGhostProfileByShiftingSpectrum(current_image, spec_range, ghost_range, ghost_fit_funct, axis_shift,
                         ghosts_high = 1, spec_axis = 0):
    #The ghost_fit_funct applies the proper shift along the spectrum axis,
    # as well as the appopriate scaling as a function of distance along the spectrum axis. 
    ghosted_image = ghost_fit_funct(current_image)
    x_mesh, y_mesh = np.meshgrid(range(np.shape(current_image)[1]), range(np.shape(current_image)[0])) 
    if ghosts_high:
        ghost_shift = ghost_range[1] - spec_range[1]
        print ('[ghost_shift, axis_shift, (y_mesh > axis_shift if axis_shift > 0 else y_mesh < np.shape(current_image)[spec_axis] + axis_shift)  ] = ' + str([ghost_shift, axis_shift, (y_mesh > axis_shift if axis_shift > 0 else y_mesh < np.shape(current_image)[spec_axis] + axis_shift)  ]))
        y_mask = y_mesh > ghost_shift
        x_mask = (x_mesh > axis_shift if axis_shift > 0 else x_mesh < np.shape(current_image)[spec_axis] + axis_shift) 
    else:
        ghost_shift = ghost_range[0] - spec_range[0]
        print ('[ghost_shift, axis_shift, (y_mesh > axis_shift if axis_shift > 0) else (x_mesh > axis_shift if axis_shift > 0 else x_mesh < np.shape(current_image)[spec_axis] + axis_shift) ] = ' + str([ghost_shift, axis_shift, (x_mesh > axis_shift if axis_shift > 0 else x_mesh < np.shape(current_image)[spec_axis] + axis_shift)  ]))
        y_mask = y_mesh > ghost_shift
        x_mask = (x_mesh > axis_shift if axis_shift > 0 else x_mesh < np.shape(current_image)[spec_axis] + axis_shift) 

    shift_mask = x_mask * y_mask
    print ('x_mask = ' + str(x_mask))
    print ('y_mask = ' + str(y_mask))
    print ('shift_mask = ' + str(shift_mask)) 
    f, axarr = plt.subplots(2,2)
    #axarr[0,0].imshow(current_image) 
    print ('ghost_shift = ' + str(ghost_shift)) 
    ghosted_image = np.roll(ghosted_image, ghost_shift, axis = 0)# roll and replace with 0

    #axarr[0,1].imshow(ghosted_image)
    #axarr[1,1].imshow(shift_mask) 
    #plt.show()

    ghosted_image = ghosted_image * shift_mask 
    
    return ghosted_image 

def removeCleanSpectrumFromImage(current_image, spec_range, clean_subimage,
                                 spec_axis = 0):
    clean_n_pix = np.shape(clean_subimage)[0]
    print ('clean_n_pix = ' + str(clean_n_pix) ) 
    clean_spec = np.median(clean_subimage, axis = spec_axis)
    spectrumless_image = current_image.copy()
    plt.plot(range(len(clean_spec)), clean_spec)
    plt.show()
    print ('current_image[500, 500] = ' + str(current_image[500, 500])) 
    if spec_axis == 0: 
        spectrumless_image[spec_range[0]:spec_range[1], :] = spectrumless_image[spec_range[0]:spec_range[1], :] - clean_spec
    else:
        spectrumless_image[:, spec_range[0]:spec_range[1]] = spectrumless_image[:, spec_range[0]:spec_range[1]] - clean_spec
    print ('current_image[500, 500] = ' + str(current_image[500, 500]))
    print ('spectrumless_image[500, 500] = ' + str(spectrumless_image[500, 500]))  
    f, axarr = plt.subplots(1,2)     
    axarr[0].imshow(current_image) 
    axarr[1].imshow(spectrumless_image)
    c.saveDataToFitsFile(np.transpose(spectrumless_image), 'test_subtract_clean.fits', target_dir, header = 'default', overwrite = True, n_mosaic_extensions = 0)     
    plt.show() 

    return spectrumless_image 

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
    med_bias = c.smartMedianFitsFiles(bias_images, target_dir, bias_x_partitions, bias_y_partitions)
    m_bias_header = c.readInDataFromFitsFile(bias_images[-1], target_dir)[1]
    utc_time = datetime.utcnow()
    m_bias_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
    m_bias_header['NCOMBINE'] = (str(len(bias_images)), 'Number of raw biases stacked.')
    m_bias_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')

    c.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = m_bias_header, overwrite = True, n_mosaic_extensions = 0)
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
    med_dark = c.smartMedianFitsFiles([bias_sub_prefix + dark_image for dark_image in dark_images], target_dir, dark_x_partitions, dark_y_partitions, scalings = [1.0 / time for time in exp_times] )
    if remove_intermediate_files:
        [os.remove(target_dir + bias_sub_prefix + dark_image) for dark_image in dark_images ] 
    m_dark_header['MKTIME'] = (str(datetime.utcnow() ), 'UTC of master bias creation')
    m_dark_header['NCOMBINE'] = (str(len(dark_images)), 'Number of bias-sub darks stacked.')
    m_dark_header['SUM_TYPE'] = ('MEDIAN','Addition method for stacking biases.')
        
    c.saveDataToFitsFile(np.transpose(med_dark), master_dark_file, target_dir, header = m_dark_header, overwrite = True, n_mosaic_extensions = 0)
    print ('Master dark file created ' + target_dir + master_dark_file) 
    return 1

def determineSmoothedDerives(current_image, smoothing = 1):
    smoothed_spec = [np.median(current_image[i:i+smoothing]) for i in range(0, len(current_image) - smoothing + 1) ]
    #plt.plot(range(len(smoothed_spec)), smoothed_spec)
    #plt.show() 
    pseudo_derivs = [smoothed_spec[i+1] - smoothed_spec[i] for i in range(len(smoothed_spec)-1)]
    #plt.plot(range(len(pseudo_derivs)), pseudo_derivs)
    #plt.show()
    return pseudo_derivs 

def determineSpecRowRanges(current_image,
                           spec_axis = 0, sum_method = 'sum',):
    #plt.imshow(current_image)
    #plt.show()
    perp_spec_axis = (spec_axis + 1) % 2
    #axis0_spec = np.sum(current_image, axis = 0)
    #plt.plot(range(len(axis0_spec)), axis0_spec, color = 'red') 
    perp_spec = np.sum(current_image, axis = perp_spec_axis)
    perp_len = len(perp_spec) 
    #plt.plot(range(perp_len), perp_spec, color = 'blue')

    perp_spec_derivs = np.gradient(perp_spec)
    #plt.scatter(range(len(perp_spec_derivs)), perp_spec_derivs)
    #plt.show()
    perp_line_step_up = np.argmax(perp_spec_derivs)
    perp_line_step_down = np.argmin(perp_spec_derivs)

    if perp_line_step_up > perp_line_step_down:
        print ('Cannot identify location of spectrum on image.  Returning 0s.')
        return [-1, -1]
    #perp_derivs = determineSmoothedDerives(current_image, smoothing = edge_finding_smoothing) 

    return [perp_line_step_up, perp_line_step_down]

def fitSingleLine(fit_xs, fit_ys, init_guess, bounds = ([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), show_fit = 0):
    
    fit_funct = lambda xs, A, mu, sig, shift: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift
    #plt.scatter(fit_xs, fit_ys)
    #plt.plot(fit_xs, fit_funct(fit_xs, *init_guess), c = 'r')
    if show_fit:
        plt.plot(fit_xs, fit_funct(fit_xs, *init_guess), c = 'red')
    try:
        #print ('[init_guess, bounds[0], bounds[1]] = ' + str([init_guess, bounds[0], bounds[1]]))
        #print ('init_guess = ' + str(init_guess))
        #print ('bounds = ' + str(bounds)) 
        fit_res = optimize.curve_fit(fit_funct, fit_xs, fit_ys, p0 = init_guess, bounds = bounds)[0].tolist()
        if show_fit:
            plt.plot(fit_xs, fit_funct(fit_xs, *fit_res), c = 'green')
        fit_sum_of_sqrs = np.sum((np.array(fit_ys) - np.array(fit_funct(fit_xs, *fit_res))) ** 2.0)
        mean_sum_of_sqrs = np.sum((np.array(fit_ys) - np.mean(fit_ys)) ** 2.0)
        fit_res = fit_res + [fit_sum_of_sqrs, mean_sum_of_sqrs] 
    except (RuntimeError, TypeError) :
        #print ('Failed to fit one possible line.')
        #print ('[fit_funct, fit_xs, fit_ys, init_guess] = ' + str([fit_funct, fit_xs, fit_ys, init_guess])) 
        return [-1, -1, -1, -1, -1, -1]
    #plt.plot(fit_xs, fit_funct(fit_xs, *(fit_res)), c = 'g')
    return fit_res 
    

def identifyLinesOnSlice(xs, ys,
                         std_thresh = 3, max_line_fit_width = 20, n_pix_above_thresh = 1, 
                         init_fit_width_guess = 2.0, background_bin_width = 20, show_spec = 0, verbose = 1):
    std_thresh = 2.5
    n_pix = len(xs)
    bg_from_binning = [np.median(ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
    bg_ys = ([ys[i] - bg_from_binning[i] for i in range(n_pix)])
    bg_ys_med = np.median(bg_ys)
    bg_ys_std = np.std(bg_ys)
    print ('Here 1' )
    bg_ys_std = [np.std(bg_ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
    pix_vals_above_std = [pix for pix in xs[1:-1] if bg_ys[pix] > bg_ys_med + std_thresh * bg_ys_std[pix]]
    #Our guesses for the centers of ghost lines are those pixels that are both part of a group of pixels that are some threshold above the noise and are local extremum
    peak_guesses = ([0] if ((np.mean(bg_ys[0:int(n_pix_above_thresh/2 + 0.5)]) > bg_ys_med + std_thresh * bg_ys_std[0]) and (ys[0] > ys[1])) else [] )
    peak_guesses = peak_guesses + [pix for pix in pix_vals_above_std if ((ys[pix] > ys[pix-1] and ys[pix] > ys[pix+1]) and (np.mean(bg_ys[max(0, pix-int(n_pix_above_thresh/2)):min(pix+int(n_pix_above_thresh + 0.5), n_pix-1)]) > bg_ys_med + std_thresh * bg_ys_std[pix])) ]
    peak_guesses = peak_guesses + ([xs[-1]] if ((np.mean(bg_ys[-int(n_pix_above_thresh/2 + 0.5):]) > bg_ys_med + std_thresh * bg_ys_std[-1]) and (ys[-1] > ys[-2])) else [] ) 
    #peak_guesses = (([0] if ((np.mean(bg_ys[0:int(n_pix_above_thresh/2 + 0.5)]) > bg_ys_med + std_thresh * bg_ys_std[0]) and (ys[0] > ys[1])) else [] )
    #                               + [pix for pix in pix_vals_above_std if ((ys[pix] > ys[pix-1] and ys[pix] > ys[pix+1]) and (np.mean(bg_ys[pix-int(n_pix_above_thresh/2):pix+int(n_pix_above_thresh + 0.5)]) > bg_ys_med + std_thresh * bg_ys_std[pix])) ]
    #                               + ([xs - 1] if ((np.mean(bg_ys[-int(n_pix_above_thresh/2 + 0.5):]) > bg_ys_med + std_thresh * bg_ys_std[-1]) and (ys[-1] > ys[-2])) else [] ) )
    print ('Here 2') 
    if verbose: print ('peak_guesses = ' + str(peak_guesses)) 
    if verbose: 
        plt.plot(xs, ys, c = 'blue')
        plt.plot(xs, bg_ys, c = 'red')
        plt.show() 
    if len(peak_guesses) == 0:
        print ('No significant peaks detected on slice.' ) 
    #for pix in pix_vals_above_std:
    #    [[] pix_vals_above_std]
    n_peak_guesses = len(peak_guesses)
    line_fits = [0 for peak in peak_guesses]
    for j in range(0, n_peak_guesses ):
        peak_guess = peak_guesses[j]
        if n_peak_guesses == 1:
            fit_xs = list( range(max(int(peak_guesses[j] - max_line_fit_width), xs[0]),
                                 min(int(peak_guesses[j] + max_line_fit_width), n_pix)) )
        elif j == 0:
            fit_xs = list( range(max(int(peak_guesses[j] - max_line_fit_width), 0),
                                 min(int(peak_guesses[j] + max_line_fit_width), int(peak_guesses[j] + peak_guesses[j+1]) // 2, xs[-1])) )
        elif j == n_peak_guesses - 1:
            fit_xs = list( range(max(int(peak_guesses[j] - max_line_fit_width), int(peak_guesses[j-1] + peak_guesses[j]) // 2, xs[0]),
                             min(int(peak_guesses[j] + max_line_fit_width), xs[-1])) )
        else:
            fit_xs = list( range(max(int(peak_guesses[j] - max_line_fit_width), int(peak_guesses[j-1] + peak_guesses[j]) // 2, xs[0]),
                             min(int(peak_guesses[j] + max_line_fit_width), int(peak_guesses[j] + peak_guesses[j+1]) // 2, xs[-1])) )
        fit_ys = ys[fit_xs[0]:fit_xs[-1] + 1]
        #print ('[fit_xs, fit_ys] = ' + str([fit_xs, fit_ys]))
        init_guess = [max(fit_ys) - bg_ys_med, peak_guess, init_fit_width_guess, bg_ys_med]
        lower_bounds = [0.0, fit_xs[0], 0.5, -np.inf ]
        lower_bounds = [min(lower_bounds[i], init_guess[i]) for i in range(len(init_guess))]
        upper_bounds = [np.inf, fit_xs[-1], xs[-1] - xs[0], init_guess[1] + init_guess[3]]
        upper_bounds = [max(upper_bounds[i], init_guess[i]) for i in range(len(init_guess))]
        line_fits[j] = fitSingleLine(fit_xs, fit_ys, init_guess, bounds = (lower_bounds, upper_bounds), show_fit = show_spec)

    if show_spec:
        plt.show()    
    #print ('line_fits = ' + str(line_fits)) 
    return line_fits 

def extendLinesIntoImage(range_to_extend, line_extensions, data_to_search, ref_line_ends, 
                         binning_search_width = 3, spec_axis = 0, max_sep_per_pix = 3,
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
                            max_sep_per_pix = 2.0, max_frac_intensity_per_pix = 0.1, min_detections = 10,
                            fit_line_center_index = 1, image_range_to_search = None,
                            binning_search_width = 1, line_bg_width = 10):
    continuous_lines = []
    #First, trace out the lines only where they were detected
    print ('lines_by_slice = ' + str(lines_by_slice)) 
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
    print ('continuous_lines[0] = ' + str(continuous_lines[0])) 
    
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
        
    for continuous_line in continuous_lines:
        plt.plot([point[2] for point in continuous_line], [point[0] for point in continuous_line])
    plt.show()
    #f, axarr = plt.subplots(2,2) 
    #axarr[0,0].plot([point[0] for point in continuous_line], [point[1] for point in continuous_line])
    #axarr[1,0].plot([point[0] for point in continuous_line], [point[-2] for point in continuous_line])
    #axarr[0,1].plot([point[0] for point in continuous_line], [point[-1] for point in continuous_line])
    #axarr[1,1].plot([point[0] for point in continuous_line], np.array([point[-2] for point in continuous_line]) / np.array([point[-1] for point in continuous_line]))
    #plt.show() 
    
    continuous_lines = [ line for line in continuous_lines if len(line) >= min_detections ]

    return continuous_lines 

def traceLinesOverRange(image, search_range,
                        spec_axis = 0, n_std_for_line = 5,
                        search_binning = 3, max_line_fit_width = 20,
                        n_pix_above_thresh = 1, width_guess = 3,
                        show_process = 0):
    print ('search_range = ' + str(search_range)) 
    pix_vals = list(range(int(search_range[0]), int(search_range[1] - search_binning) + 1))
    all_slices = [] 
    for i in range(len(pix_vals)):
        pix_val = pix_vals[i]
        print ('Computing lines for pix_val ' + str(pix_val)) 
        if spec_axis == 0: 
            spec_slice = image[pix_val:pix_val + search_binning, :]
        else:
            spec_slice = image[:, pix_val:pix_val + search_binning]
        if search_binning > 1: 
            spec_slice = np.median(spec_slice, axis = spec_axis)
        
        
        line_fits = identifyLinesOnSlice(range(len(spec_slice)), spec_slice,
                                               std_thresh = n_std_for_line, max_line_fit_width = max_line_fit_width,
                                               n_pix_above_thresh = n_pix_above_thresh, init_fit_width_guess = width_guess,
                                               show_spec = (i % 50 == 51), verbose =  (i % 50 == 0))
        #print ('line_fits = ' + str(line_fits)) 
        if i % 50 == 0:
            plt.plot(range(len(spec_slice)), spec_slice, c = 'blue')
            fit_funct = lambda xs, A, mu, sig, shift: A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift
            for line_fit in line_fits:
                plt.plot(range(len(spec_slice)), fit_funct(range(len(spec_slice)), *(line_fit[0:-2])), c = 'r') 
            #fit_spec = np.sum([fit_funct(range(len(spec_slice)), *(line_fit[0:-2])) for line_fit in line_fits], axis = 0)
            #plt.plot(range(len(spec_slice)), fit_spec) 
            plt.show() 
        all_slices = all_slices + [line_fits]

    return pix_vals, all_slices 

def traceStrongGhosts(current_image, spec_range,
                      ghosts_high = 1, ghost_search_buffer = [10, 50],
                      spec_axis = 0, min_pixels_for_ghost_peak = 20,  
                      n_std_for_ghost = 5, ghost_search_binning = 3,
                      max_line_fit_width = 20, ghost_width_guess = 1,
                      show_process = 0, n_pix_above_thresh = 1, search_for_ghost_extensions = None, 
                      min_detections = 10, max_sep_per_pix = 2.0,
                      max_frac_intensity_per_pix = 0.1, bg_fit_width = 10):
    perp_line_step_up, perp_line_step_down = spec_range
    all_ghost_slices = []
    ghost_pix_vals = [(perp_line_step_down + ghost_search_buffer[0] + i) if ghosts_high else perp_line_step_up - ghost_search_buffer[1] + i
                      for i in range(ghost_search_buffer[1] - ghost_search_buffer[0] - ghost_search_binning + 1)]
    ghost_pix_range = ( ([perp_line_step_down + ghost_search_buffer[0], perp_line_step_down + ghost_search_buffer[1]])
                        if ghosts_high else (perp_line_step_up - ghost_search_buffer[1], perp_line_step_up - ghost_search_buffer[0]) ) 
    print ('ghost_pix_range = ' + str(ghost_pix_range))

    all_ghost_slices = traceLinesOverRange(current_image, ghost_pix_range,
                                           spec_axis = spec_axis, n_std_for_line = n_std_for_ghost,
                                           search_binning = ghost_search_binning, max_line_fit_width = max_line_fit_width,
                                           n_pix_above_thresh = n_pix_above_thresh, width_guess = ghost_width_guess,
                                           show_process = show_process)[1]

    print ('search_for_ghost_extensions = ' + str(search_for_ghost_extensions)) 
    ghost_lines = identifyContinuousLines(ghost_pix_vals, all_ghost_slices, current_image, 
                                          max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix, min_detections = min_detections,
                                          fit_line_center_index = 1, image_range_to_search = search_for_ghost_extensions,
                                          binning_search_width = ghost_search_binning, line_bg_width = bg_fit_width) 
    return ghost_lines

def determineGhostSegments(full_ghosts, full_spec_len, spec_axis = 0, ghosts_n_sig_width = 5):
    
    n_ghosts = len(full_ghosts)
    sum_axis = (spec_axis + 1) % 2
    full_spec_len = np.shape(current_image)[spec_axis]
    xs = list(range(full_spec_len) ) 
    ghost_segments = [[-1, -1] for ghost in full_ghosts]
    ghost_range = [np.inf, -1]
    for i in range(n_ghosts):
        ghost = full_ghosts[i] 
        ghost_centers = np.array([ghost_slice[2] for ghost_slice in ghost])
        ghost_widths = np.array([ghost_slice[3] for ghost_slice in ghost]) 
        ghost_left_bound = np.min(ghost_centers - ghosts_n_sig_width * ghost_widths )
        if ghost_left_bound < 0: ghost_left_bound = 0
        ghost_right_bound = np.max(ghost_centers + ghosts_n_sig_width * ghost_widths)
        if ghost_right_bound > full_spec_len-1: ghost_left_bound = full_spec_len-1
        ghost_segments[i] = [ghost_left_bound, ghost_right_bound]
        ghost_pixels = [ghost_slice[0] for ghost_slice in ghost] 
        ghost_range = [min([ghost_range[0]] + ghost_pixels), max([ghost_range[1]] + ghost_pixels)]
        
    n_ghosts_merged = 0
    print ('ghost_segments before merging = ' + str(ghost_segments))
    for i in range(1, n_ghosts):
        if ghost_segments[i-1-n_ghosts_merged][1] > ghost_segments[i][0]:
            ghost_segments[i-1-n_ghosts_merged][1] = max(ghost_segments[i-1-n_ghosts_merged][1], ghost_segments[i][1])
            n_ghosts_merged = n_ghosts_merged + 1
        else:
            ghost_segments[i-n_ghosts_merged] = ghost_segments[i]
    if n_ghosts_merged > 0: 
        ghost_segments = ghost_segments[0:-n_ghosts_merged]
    print ('ghost_segments after merging = ' + str(ghost_segments))
    #ghost_range = [int(min([seg[0] for seg in ghost_segments])), int(max([seg[1] for seg in ghost_segments]))]
    print ('ghost_range = ' + str(ghost_range))
    
    return ghost_segments, ghost_range

def determineSubimagesForGhostCorrection(full_ghosts, current_image, spectrum_range, 
                                         spec_axis = 0, ghosts_n_sig_width = 5, clean_buffer = 5):

    n_ghosts = len(full_ghosts)
    full_spec_len = np.shape(current_image)[spec_axis]
    xs = list(range(full_spec_len)) 
    ghost_segments, ghost_range = determineGhostSegments(full_ghosts, full_spec_len, spec_axis = spec_axis, ghosts_n_sig_width = ghosts_n_sig_width)

    if ghost_range[1] < spectrum_range[0]:
        correction_range = [spectrum_range[0], spectrum_range[0] + (ghost_range[1] - ghost_range[0]) ]
        clean_range = [spectrum_range[1] - (ghost_range[1] - ghost_range[0]) + clean_buffer, spectrum_range[1]]
    else:
        correction_range = [spectrum_range[1] - (ghost_range[1] - ghost_range[0]), spectrum_range[1] ]
        clean_range = [spectrum_range[0], spectrum_range[0] + (ghost_range[1] - ghost_range[0]) - clean_buffer]
    print ('correction_range = ' + str(correction_range)) 
    if spec_axis == 0: 
        ghost_subimage = current_image[ghost_range[0]:ghost_range[1], :]
        correction_subimage = current_image[correction_range[0]:correction_range[1],:]
        clean_subimage = current_image[clean_range[0]:clean_range[1],:]
    else:
        ghost_subimage = current_image[:, ghost_range[0]:ghost_range[1]]
        correction_subimage = current_image[ :, correction_range[0]:correction_range[1] ]
        clean_subimage = current_image[:, clean_range[0]:clean_range[1]]

    mask_line = [np.any([(i > ghost_segment[0] and i < ghost_segment[1] ) for ghost_segment in ghost_segments]) for i in range(full_spec_len) ]
    return [ghost_subimage, ghost_segments, ghost_range, correction_subimage, clean_subimage, clean_range, mask_line] 
    

def matchSpectrumToGhost(ghost_subimage, ghost_segments, correction_subimage, mask_line, 
                         spec_axis = 0, background_fit_order = 2, ghosts_right = 0 ):
 
    full_spec_len = np.shape(correction_subimage)[(1+spec_axis)%2]
    print('full_spec_len = ' + str(full_spec_len))
    #xs = list(range(full_spec_len))
    xs = [pix - full_spec_len // 2 for pix in list(range(full_spec_len)) ]
        
    masked_ghost_subimage = ghost_subimage * mask_line
    
    #f, axarr = plt.subplots(2,1)
    #axarr[0].imshow(ghost_subimage)
    #axarr[1].imshow(masked_ghost_subimage)
    #plt.show()
    masked_ghost_spec = np.median(masked_ghost_subimage, spec_axis)

    #To measure a rough constant for the background, take the median of the lower half of the binned counts, per segment
    # Assumes the background for a single segment with ghosts can be treated as a constant, for our purposes.
    frac_vals_for_bg = 0.5
    for ghost_segment in ghost_segments:
        ghost_segment = [int(ghost_segment[0]), int(ghost_segment[1])] 
        segment_len = ghost_segment[1] - ghost_segment[0]
        ghost_bg = np.median(sorted(masked_ghost_spec[ghost_segment[0]+1:ghost_segment[1]])[0:int(segment_len * frac_vals_for_bg)])
        masked_ghost_spec[ghost_segment[0]+1:ghost_segment[1]] = masked_ghost_spec[ghost_segment[0]+1:ghost_segment[1]] - ghost_bg

    max_ghost_pix = np.argmax(masked_ghost_spec)
    max_ghost_peak = np.max(masked_ghost_spec)

    correction_spec = np.median(correction_subimage, axis = spec_axis)
    max_correction_pix = np.argmax(correction_spec)
    max_correction_peak = np.max(correction_spec)

    print ('[max_ghost_pix, max_ghost_peak] = ' + str([max_ghost_pix, max_ghost_peak])) 
    print ('[max_correction_pix, max_correction_peak] = ' + str([max_correction_pix, max_correction_peak]))

    f, axarr = plt.subplots(2,1)

    print ('[np.shape(ghost_subimage), np.shape(masked_ghost_spec), np.shape(np.median(ghost_subimage, axis = spec_axis))] = ' + str([np.shape(ghost_subimage), np.shape(masked_ghost_spec), np.shape(np.median(ghost_subimage, axis = spec_axis))])) 
    axarr[0].plot(range(full_spec_len), np.median(ghost_subimage, axis = spec_axis), c = 'red')
    axarr[0].plot(range(full_spec_len), masked_ghost_spec, c = 'green')
    axarr[1].plot(range(full_spec_len), correction_spec)
    plt.show() 
    
    #We may want to make this a second order polynomial...
    shift_sign = 1 if ghosts_right else -1
    alter_subimage_segment = lambda segment_to_alter, shift, scaling0, scaling1, x_vals: np.roll(segment_to_alter, int(shift)) * (scaling0 + np.array(x_vals) * scaling1)  * mask_line
    alter_subimage_segment = lambda segment_to_alter, shift, scaling0, scaling1, scaling2, x_vals: np.roll(segment_to_alter, int(shift)) * (scaling0 + np.array(x_vals) * scaling1 + np.array(x_vals) ** 2.0 * scaling2 ) * mask_line
    alter_correction_segment = lambda shift, scaling0, scaling1, scaling2, x_vals: alter_subimage_segment(correction_subimage, shift, scaling0, scaling1, scaling2, x_vals) 
    #plt.imshow(np.roll(correction_subimage, int(-100)) * (1.0 + np.array(xs) * 0.0))
    #plt.show() 
    ghost_fit_init_guess = [max_ghost_pix - max_correction_pix, max_ghost_peak / max_correction_peak, 0.0, 0.0 ]

    #Do a coarse fit in which we try to roughly place the location of the lines 
    coarse_fit_val = np.inf
    coarse_shift = 0
    coarse_fit = []
    print ('Working on coarse fits...') 
    for i in range(full_spec_len // 4):
        i = i * 4
        shift = shift_sign * xs[i]
        #print ('fitting with shift = ' + str(shift) + ' (i = ' + str(i) + ' of ' + str(full_spec_len) + ')')
        #init_guess = [ghost_fit_init_guess [1], ghost_fit_init_guess [2] ] 
        new_fit = optimize.minimize_scalar(lambda constant_scaling: np.log(np.sum((masked_ghost_spec - np.median(alter_correction_segment(shift, constant_scaling, 0.0, 0.0, xs ), spec_axis)) ** 2.0)))
        if new_fit['fun'] < coarse_fit_val:
            coarse_fit_val = new_fit['fun']
            coarse_shift = shift
            coarse_fit = new_fit 
            #print ('coarse_fit = ' + str(coarse_fit))
        #print ("[init_guess, new_fit['x']] = " + str([init_guess, new_fit['x']]))
    print ('Done with coarse fitting.') 

    coarse_fit_params = [coarse_shift, coarse_fit['x'], 0.0, 0.0]
    #once the course fit is done, allow for a slope in the ghost intensity function and move by individual pixels 
    print ('coarse_fit = ' + str(coarse_fit))
    print ('coarse_fit_params = ' + str(coarse_fit_params)) 
    coarse_fit_correction_spec =  np.median(alter_correction_segment(*coarse_fit_params, xs ), spec_axis) 
    #f, axarr = plt.subplots(2,1)
    plt.plot(range(full_spec_len), masked_ghost_spec, c = 'blue')
    plt.plot(range(full_spec_len), coarse_fit_correction_spec, c = 'red')
    plt.show()

    best_fit_val = coarse_fit_val
    best_fit_shift = coarse_shift 
    best_fit = coarse_fit
    fine_fit_search = 10 
    for i in range(coarse_shift - fine_fit_search, coarse_shift + fine_fit_search):
        shift = i
        print ('fitting with shift = ' + str(shift) + ' (iteration = ' + str(i - (coarse_shift - fine_fit_search) + 1) + ' of ' + str(len(range(coarse_shift - fine_fit_search, coarse_shift + fine_fit_search))) + ')')
        init_guess = [coarse_fit['x'], 0.0, 0.0]
        new_fit = optimize.minimize(lambda scaling_params: np.log(np.sum((masked_ghost_spec - np.median(alter_correction_segment(shift, *scaling_params, xs ), spec_axis)) ** 2.0)), init_guess)
        #print ('new_fit = ' + str(new_fit)) 
        if new_fit['fun'] < best_fit_val:
            best_fit_val = best_fit['fun']
            best_fit_shift = shift
            best_fit = new_fit 
            print ('best_fit = ' + str(best_fit))
        #print ("[init_guess, best_fit['x']] = " + str([init_guess, best_fit['x']]))

    print ('best_fit = ' + str(best_fit)) 
    best_fit_params = [best_fit_shift, *(best_fit['x'].tolist())]
    #once the course fit is done, allow for a slope in the ghost intensity function and move by individual pixels 
    print ('best_fit = ' + str(best_fit))
    print ('best_fit_params = ' + str(best_fit_params)) 
    best_fit_correction_spec =  np.median(alter_correction_segment(*best_fit_params, xs ), spec_axis) 
    plt.plot(range(full_spec_len), masked_ghost_spec, c = 'blue')
    plt.plot(range(full_spec_len), coarse_fit_correction_spec, c = 'red')
    plt.plot(range(full_spec_len), best_fit_correction_spec, c = 'green')
    plt.show()
        
    return [lambda image_segment_to_make_ghost: alter_subimage_segment(image_segment_to_make_ghost, *best_fit_params, xs),  best_fit_shift]

def getLineFunction(line,
                    spec_axis = 0,
                    position_order = 2, A_of_x_order = 2, sig_of_x_order = 2):
    
    if spec_axis == 0:
        ys = [line_part[0] for line_part in line]
        xs = [line_part[2] for line_part in line]
        ind_var = ys
        ind_var_range = [min(ys), max(ys)]
        line_center_poly = np.polyfit(ys, xs, position_order)
        position_funct = lambda val: np.poly1d(line_center_poly)(val)
    else:
        ys = [line_part[2] for line_part in line]
        xs = [line_part[0] for line_part in line]
        ind_var = xs
        ind_var_range = [min(xs), max(xs)]
        position_poly = np.polyfit(xs, ys, position_order)
        position_funct = lambda val: np.poly1d(line_center_poly)(val)
        
    As = [line_part[1] for line_part in line]
    sigs = [line_part[3] for line_part in line]
    A_poly = np.polyfit(ind_var, As, A_of_x_order)
    A_funct = lambda val: np.poly1d(A_poly)(val) * (val >= ind_var_range[0]) * (val <= ind_var_range[1]) 

    sig_poly = np.polyfit(ind_var, sigs, sig_of_x_order)
    sig_funct = lambda val: np.poly1d(sig_poly)(val) * (val >= ind_var_range[0]) *(val <= ind_var_range[1]) 

    return [position_funct, A_funct, sig_funct] 

def generateLineImage(data_to_mask, mask_shape, save_fits = 0, spec_axis = 0):
    line_mask = np.zeros(mask_shape)
    y_dist, x_dist = mask_shape
    x_mesh, y_mesh = np.meshgrid(range(x_dist), range(y_dist))
    if spec_axis == 0:
        ind_coord_mesh = y_mesh
        dep_coord_mesh = x_mesh 
    else:
        ind_coord_mesh = x_mesh
        dep_coord_mesh = y_mesh 
    for line in data_to_mask:
        position_funct, A_funct, sig_funct = getLineFunction(line, spec_axis = spec_axis) 
        line_array = A_funct(ind_coord_mesh) * np.exp(-(position_funct(ind_coord_mesh) - dep_coord_mesh)** 2.0 / (2.0 * sig_funct(ind_coord_mesh) ** 2.0) )
        plt.imshow(line_array)
        plt.show()
        line_mask = line_mask + line_array
    plt.imshow(line_mask)
    plt.show() 

    return line_mask 

def correctGhosts(current_image, current_header, spec_range, 
                  ghosts_n_sig_width = 10, ghosts_high = 1,
                  ghosts_right = 0, spec_axis = 0,
                  background_fit_order = 2, min_ghosts = 3,
                  trace_ghosts = 0, removeGhostByShiftingSpectrum = 1,
                  n_std_for_most_ghosts = 1.0, n_std_for_first_ghosts = 4.0,
                  ghost_search_buffer = [10, 50], min_pixels_for_ghost_peak = 10,
                  ghost_search_binning = 3, max_line_fit_width = 20,
                  ghost_width_guess = 1,  clean_buffer = 5,
                  min_detections_for_ghost_line = 10):
    
    strong_ghost_data = traceStrongGhosts(current_image, spec_range,
                                          ghosts_high = ghosts_high, spec_axis = spec_axis,
                                          n_std_for_ghost = n_std_for_first_ghosts, ghost_search_buffer = ghost_search_buffer,
                                          min_pixels_for_ghost_peak = min_pixels_for_ghost_peak, ghost_search_binning = ghost_search_binning,
                                          max_line_fit_width = max_line_fit_width, ghost_width_guess = ghost_width_guess,
                                          min_detections = min_detections_for_ghost_line )
    
    ghost_subimage, ghost_segments, ghost_range, correction_subimage, clean_subimage, clean_range, mask_line = determineSubimagesForGhostCorrection(strong_ghost_data, current_image, spec_range, 
                                                                                                                                                    spec_axis = spec_axis, ghosts_n_sig_width = ghosts_n_sig_width,
                                                                                                                                                    clean_buffer = clean_buffer)
    print ('ghost_range = ' + str(ghost_range))
    most_ghost_data = traceStrongGhosts(current_image, spec_range,
                                        ghosts_high = ghosts_high, spec_axis = spec_axis,
                                        n_std_for_ghost = n_std_for_most_ghosts, ghost_search_buffer = [ghost_search_buffer[0], ghost_search_buffer[0] + ghost_range[1] - ghost_range[0]], 
                                        min_pixels_for_ghost_peak = min_pixels_for_ghost_peak, ghost_search_binning = ghost_range[1] - ghost_range[0],
                                        max_line_fit_width = max_line_fit_width, ghost_width_guess = ghost_width_guess,
                                        show_process = 1, search_for_ghost_extensions = [min(spec_range[0], ghost_range[0]), max(spec_range[1], ghost_range[1])])
    #print ('most_ghost_data = ' + str(most_ghost_data))
    print ('most_ghost_data[0] = ' + str(most_ghost_data[0]))
    #sys.exit() 
    #(current_image, spec_range,
    #                  ghosts_high = 1, ghost_search_buffer = [10, 50],
    #                  min_pixels_for_ghost_peak = 20, spec_axis = 0, 
    #                  n_std_for_ghost = 3, ghost_search_binning = 3,
    #                  max_line_fit_width = 10, ghost_width_guess = 1 )
    ghostMaskData = generateLineImage(most_ghost_data, np.shape(current_image), save_fits = 0)
    current_image = current_image - ghostMaskData 
    n_ghosts = len(strong_ghost_data) 
    print ('We have identified ' + str(n_ghosts) + ' ghosts.')
    if removeGhostByShiftingSpectrum:
        #if trace_ghosts: 
        if n_ghosts < min_ghosts:
            print ('Needed to identify at least ' + str(min_ghosts) + ' to determine new ghost fit.  Will use archived one...')
            #ghost_fit_funct = loadArchiveGhostFit()
            current_header['GHOSTED'] = ('GhXXX','Used archived ghost solution.')
        else: 
            
            f, axarr = plt.subplots(3,1)
            axarr[0].imshow(ghost_subimage)
            axarr[1].imshow(correction_subimage)
            axarr[2].imshow(clean_subimage)
            plt.show() 
            ghost_subimage, ghost_segments, ghost_range
            print ('ghost_range = ' + str(ghost_range)) 
            ghost_fit_funct, axis_shift = matchSpectrumToGhost(ghost_subimage, ghost_segments, correction_subimage, mask_line, 
                                                               spec_axis = spec_axis, background_fit_order = background_fit_order, ghosts_right = ghosts_right)
            current_header['GHOSTED'] = ('GhXXX','Generated own ghost solution.')
            #saveGhostFit
            artificial_ghost_image = generateGhostProfileByShiftingSpectrum(current_image, spec_range, ghost_range, ghost_fit_funct, axis_shift, 
                                                                            ghosts_high = ghosts_high, spec_axis = spec_axis) #shift spectrum segment by ghost fit
    else:
        #We correct the ghosts by fitting the non-ghosted sections
        if n_ghosts < min_ghosts:
            print ('Needed to identify at least ' + str(min_ghosts) + ' to determine new ghost fit.  Will use archived one...')
            #ghost_fit_funct = loadArchiveGhostFit()
            current_header['GHOSTED'] = ('GhXXX','Used archived ghost solution.')
        else: 
            full_spec_len = np.shape(current_image)[(1+spec_axis)%2] 
            ghost_segments, ghost_range = determineGhostSegments(strong_ghost_data, full_spec_len, spec_axis = spec_axis, ghosts_n_sig_width = ghosts_n_sig_width)
            spectrum_subtracted_image = removeCleanSpectrumFromImage(current_image, spec_range, clean_subimage, spec_axis = spec_axis) 
            artificial_ghost_image = generateGhostProfileFromSpectrumlessImage(spectrum_subtracted_image, spec_range, ghost_range) 

    print ('ghost_fit_funct = ' + str(ghost_fit_funct))
    print ('axis_shift = ' + str(axis_shift)) 
    
    f, axarr = plt.subplots(1,2) 
    axarr[0].imshow(current_image)
    current_image = current_image - artificial_ghost_image
    axarr[1].imshow(current_image)
    plt.show()

    return current_image, current_header


def detectLines(current_image, spec_range,
                spec_axis = 0, n_std_for_lines = 5.0, 
                search_binning = 3, max_line_fit_width = 20,
                width_guess = 1, show_process = 0,
                max_sep_per_pix = 2.0, max_frac_intensity_per_pix = 0.1,
                min_detections = 10, fit_line_center_index = 1,
                search_for_extensions = None, bg_fit_width = 10):

    pix_slices, lines_by_slice = traceLinesOverRange(current_image, spec_range,
                                                     spec_axis = spec_axis, n_std_for_line = n_std_for_lines,
                                                     search_binning = search_binning, max_line_fit_width = max_line_fit_width,
                                                     n_pix_above_thresh = 1, width_guess = width_guess,
                                                     show_process = 0)

    print ('lines_by_slice = ' + str(lines_by_slice))
    lines = identifyContinuousLines(pix_slices, lines_by_slice, current_image, 
                                    max_sep_per_pix = max_sep_per_pix, max_frac_intensity_per_pix = max_frac_intensity_per_pix,
                                    min_detections = min_detections, fit_line_center_index = fit_line_center_index,
                                    image_range_to_search = search_for_extensions, binning_search_width = search_binning,
                                    line_bg_width = bg_fit_width) 

    return lines 
    
    #strong_ghost_data = traceStrongLines(current_image, spec_range,
    #                                      ghosts_high = ghosts_high, spec_axis = spec_axis,
    #                                      n_std_for_ghost = n_std_for_first_ghosts, ghost_search_buffer = ghost_search_buffer,
    #                                      min_pixels_for_ghost_peak = min_pixels_for_ghost_peak, ghost_search_binning = ghost_search_binning,
    #                                      max_line_fit_width = max_line_fit_width, ghost_width_guess = ghost_width_guess,
    #                                      min_detections = min_detections_for_ghost_line )

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
               max_sep_pix = 3.0):
    line_matches = []
    line_match_indeces = [] 
    for i in range(len(line_pixels_to_be_matched)):
        line_pixel_to_be_matched = line_pixels_to_be_matched[i]
        line_pixels_to_match = mu_of_wavelength_funct(line_wavelengths_to_match) 
        line_to_be_matched = mu_of_wavelength_funct(line_pixel_to_be_matched)
        line_seps = [abs(line_pixel_to_match - line_pixel_to_be_matched) for line_pixel_to_match in line_pixels_to_match]
        min_sep_index = np.argmin(line_seps)
        min_sep = line_seps[min_sep_index]
        if min_sep <= max_sep_pix:
            line_matches = line_matches + [[line_pixel_to_be_matched, line_wavelengths_to_match[min_sep_index]]]
            line_match_indeces = line_match_indeces + [i]
    line_sep = np.sum([np.abs(line_match[0] - line_match[1]) for line_match in line_matches])
    return [line_match_indeces, line_matches, line_sep]

def readFileIntoInterp(target_file, target_dir, n_ignore, convert_to_float = 1):
    cols = c.readInColumnsToList(target_file, file_dir = target_dir, n_ignore = n_ignore, convert_to_float = convert_to_float)
    interp = c.safeInterp1d(*cols)
    return interp 
    

#Curve fitting doesn't work very well.  Is there a way that we could just detect where lines are and determine where they are supposed to be? 
def determineWavelengthSolution(line_solutions, line_range, spec_range, ref_spec_file, 
                                spec_file_dir = '', throughput_file = 'default_throughput.txt', 
                                n_ignore_spec = 0, n_ignore_throughput = 0,
                                wavelength_solution_order = 1, wavelength_solution_drift_order = 2, 
                                course_search_param_range= [[-400.0, -100.0], [1.0, 1.5]], course_search_param_step = [26, 26]): 
    ref_spec_lines = c.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore_spec, convert_to_float = 1)

    throughput_interp = readFileIntoInterp(throughput_file, spec_file_dir, n_ignore_throughput, convert_to_float = 1)
    ref_spec_lines[1] = (np.array(ref_spec_lines[1]) * throughput_interp(ref_spec_lines[0])).tolist()

    print ('ref_spec_lines = ' + str(ref_spec_lines))
    slice_pixels = range(*spec_range)
    fitted_spectra = [[0.0 for pix in slice_pixels] for guess in range(wavelength_solution_order + 1)]
    best_match_params = [0.0, 0.0]
    best_matches = []
    best_match_sep = np.inf
    best_match_indeces = [] 

    wavelength_of_mu = lambda mu, lam0, lam1, lam2: lam0 + lam1 * mu + lam2 * mu ** 2.0
    wavelength_of_mu = lambda mu, lam0, lam1: lam0 + lam1 * mu
    mu_of_wavelength = lambda lam, mu0, mu1: mu0 + mu1 * lam 
    median_line_centers = [np.median([line_solution[0](pix) for pix in slice_pixels]) for line_solution in line_solutions]
    for test_mu0 in np.linspace(*(course_search_param_range[0]), course_search_param_step[0]):
        for test_mu1 in np.linspace(*(course_search_param_range[1]), course_search_param_step[1]):
            #print ('[test_lam0, test_lam1] = ' + str([test_lam0, test_lam1]))
            matched_line_indeces, matched_lines, matched_lines_sep = matchLines(np.array(median_line_centers), np.array(ref_spec_lines[0]), lambda pixels: wavelength_of_mu(pixels, test_mu0, test_mu1) )
            if (len(matched_lines) > len(best_matches)) or (len(matched_lines) == len(best_matches) and matched_lines_sep < best_match_sep):
                best_match_params = [test_mu0, test_mu1]
                best_matches = matched_lines
                best_match_sep = matched_lines_sep
                best_match_indeces = matched_line_indeces 
    print ('[best_match_indeces, best_match_params, best_matches, best_match_sep] = ' + str([best_match_indeces, best_match_params, best_matches, best_match_sep]))
    spec_pixels = np.arange(*line_range) 
    for i in range(spec_range[1] - spec_range[0]):
        pix = slice_pixels[i]
        line_centers = [line_solutions[index][0](pix) for index in best_match_indeces]
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
    f, axarr = plt.subplots(2,1) 
    axarr[0].scatter(slice_pixels, fitted_spectra[0])
    axarr[1].scatter(slice_pixels, fitted_spectra[1])
    plt.show()
    wavelength_polyfit = [np.polyfit(slice_pixels, fitted_spectra_term, wavelength_solution_drift_order) for fitted_spectra_term in fitted_spectra]
    print ('wavelength_polyfit = ' + str(wavelength_polyfit)) 

    fitted_wavelength_solution = [np.poly1d(fit_to_solution_term) for fit_to_solution_term in wavelength_polyfit]

    #mu(lam, y) 
    mu_of_wavelength_solution = lambda lam, y: np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution])(lam)
    wavelength_of_mu_solution = lambda mu, y: ((np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution]) - mu).roots)[0]
    print ('[[best_match[1] for best_match in best_matches], [mu_of_wavelength_solution(best_match[1], 550) for best_match in best_matches], [wavelength_of_mu_solution(best_match[0], 550) for best_match in best_matches]] = ' + str([[best_match[1] for best_match in best_matches], [mu_of_wavelength_solution(best_match[1], 550) for best_match in best_matches], [wavelength_of_mu_solution(best_match[0], 550) for best_match in best_matches]]))

    
    

    return [mu_of_wavelength_solution, wavelength_of_mu_solution]
    

def deconvolveSpectrum(current_image, spec_range, mu_of_wavelength_solution, strong_line_profiles, throughput_file, throughput_dir, n_ignore_throughput,
                       mu_index = 0, width_index = 2, width_fit_order = 2, spec_axis = 0, wavelength_range = [300.0, 1100.0], wavelength_step = 0.5 ):
    throughput_interp = readFileIntoInterp(throughput_file, throughput_dir, n_ignore_throughput, convert_to_float = 1) # Throughput(wavelength)
    spec_pixels = list(range(*spec_range))

    intensities = [[[], []] for i in range(len(spec_pixels))]
    #for i in range(len(spec_pixels)):
    for i in range(len(spec_pixels)): 
        pix = spec_pixels[i]
        print ('Computing intensity for pix = ' + str(pix))
        if spec_axis == 0: 
            spec_slice = current_image[pix, :]
        else:
            spec_slice = current_image[:, pix]
        #print ('i = ' + str(i))
        #print ('len(strong_lines[0]) = ' + str(len(strong_lines[0])))
        #print ('strong_lines[0] = ' + str(strong_lines[0]))
        #print ('strong_lines[0][i] = ' + str(strong_lines[0][i]))
        #print ('strong_lines[0][i][mu_index] = ' + str(strong_lines[0][i][mu_index]))
        #print ('strong_lines[0][i][width_index] = ' + str(strong_lines[0][i][width_index]))
        #print ('[strong_line[i][mu_index] for strong_line in strong_line_profiles] = ' + str([strong_line[i][mu_index] for strong_line in strong_line_profiles]))
        #print ('[strong_line[i][width_index] for strong_line in strong_line_profiles] = ' + str([strong_line[i][width_index] for strong_line in strong_line_profiles]))
        print ('[strong_line[mu_index](pix) for strong_line in strong_line_profiles] = ' + str([strong_line[mu_index](pix) for strong_line in strong_line_profiles]))
        print ('[strong_line[width_index](pix) for strong_line in strong_line_profiles] = ' + str([strong_line[width_index](pix) for strong_line in strong_line_profiles])) 
        width_funct = np.poly1d(np.polyfit([strong_line[mu_index](pix) for strong_line in strong_line_profiles], [strong_line[width_index](pix) for strong_line in strong_line_profiles], width_fit_order)) 
        #plt.scatter([strong_line[mu_index](pix) for strong_line in strong_line_profiles], [strong_line[width_index](pix) for strong_line in strong_line_profiles]) 
        #plt.plot(np.arange(min([strong_line[mu_index](pix) for strong_line in strong_line_profiles]), max([strong_line[mu_index](pix) for strong_line in strong_line_profiles])),
        #         width_funct(np.arange(min([strong_line[mu_index](pix) for strong_line in strong_line_profiles]), max([strong_line[mu_index](pix) for strong_line in strong_line_profiles]))))
        #plt.show() 
        spec_slice_pixels = list(range(len(spec_slice))) 
        widths = width_funct(spec_slice_pixels)
        #plt.plot(spec_slice_pixels, widths)
        #plt.show()
        #spec_slice_ft = np.fft.fft(spec_slice)
        #line_profile_ft = np.fft.fft(np.exp((-np.array(spec_slice_pixels) ** 2.0 ) / (2.0 * widths ** 2.0)))
        #fft_of_fft = np.fft.fft(spec_slice_ft / spec_slice_ft).real 
        #print ('[spec_slice_ft, line_profile_ft, fft_of_fft] = ' + str([spec_slice_ft, line_profile_ft, fft_of_fft]))
        #intensities = [np.sqrt(2.0 * np.pi * widths[j] ** 2.0) / throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) * fft_of_fft[j] if throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) > 0.0 else 0.0 for j in range(len(spec_slice_pixels))]

        #print ('computing convolution matrix...') 
        #convolution_matrix = np.array([(1.0 / np.sqrt(2.0 * np.pi * widths ** 2.0) *  np.exp(-(contributing_pixel - np.array(spec_slice_pixels)) ** 2.0 / (2.0 * widths ** 2.0))).tolist() for contributing_pixel in spec_slice_pixels])
        #print ('convolution_matrix = ' + str(convolution_matrix)) 
        #deconvolution_matrix = np.linalg.inv(convolution_matrix)
        #print ('deconvolution_matrix = ' + str(deconvolution_matrix))
        #print ('np.matmul(convolution_matrix, deconvolution_matrix) = ' + str(np.matmul(convolution_matrix, deconvolution_matrix))) 
        #intensities = np.matmul(deconvolution_matrix, spec_slice)

        intensities[i] = [[wavelength_of_mu_solution(spec_slice_pixels[j], pix) for j in range(len(spec_slice_pixels))],
                          [spec_slice[j] / throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix))[0] if throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) > 0.0 else 0.0 for j in range(len(spec_slice_pixels))] ]
        #f, axarr = plt.subplots(2,1)
        #axarr[0].plot(spec_slice_pixels, spec_slice)
        #axarr[1].plot(spec_slice_pixels, np.exp((-np.array(spec_slice_pixels) ** 2.0 ) / (2.0 * widths ** 2.0))) 
        #axarr[1].plot(spec_slice_pixels, intensities[i][1])
        #plt.show() 
        #plt.plot(spec_slice_pixels, [throughput_interp(wavelength_of_mu_solution(spec_slice_pixels[j], pix)) for j in spec_slice_pixels])
        #plt.plot(spec_slice_pixels, [np.sqrt(2.0 * np.pi * widths[j] ** 2.0) for j in spec_slice_pixels]) 
        #plt.show()

    intensity_interps = [c.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
    full_spec_interp = c.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
    doconvolved_lines = 1 
    print ('intensities = ' + str(intensities))
    f, axarr = plt.subplots(2,1)
    axarr[0].plot(np.arange(*wavelength_range, wavelength_step), intensity_interps[len(spec_pixels) // 2](np.arange(*wavelength_range, wavelength_step)))
    axarr[1].plot(np.arange(*wavelength_range, wavelength_step), full_spec_interp(np.arange(*wavelength_range, wavelength_step)))
    plt.show() 
    return 1 


if __name__ == "__main__":

    args = sys.argv[1:] 

    spec_file, target_dir, master_bias_file, master_dark_file, processed_file = args
    print ('[spec_file, target_dir, master_bias_file, master_dark_file, processed_file] = ' + str([spec_file, target_dir, master_bias_file, master_dark_file, processed_file]))
    
    print ('Reading in raw spectrum from ' + target_dir + spec_file) 
    current_image, current_header = readInRawSpect(spec_file, target_dir) #Read in raw spectrum

    processed_prefix = 'proc_' 

    #Overscan correct (?) ## Not currently set up to do this

    #[OPTIONAL] Make master bias
    master_bias_exists = os.path.isfile(target_dir + master_bias_file)
    if not(master_bias_exists):
        print ('Making master bias file.  Will be saved to ' + target_dir + master_bias_file) 
        master_bias_exists = makeMasterBias(master_bias_file, target_dir)
    if not(master_bias_exists):
        print ('Unable to find master bias file, ' + target_dir + master_bias_file + ', and also could not make it.')
        sys.exit() 

    #[OPTIONAL] Make master dark
    master_dark_exists = os.path.isfile(target_dir + master_dark_file)
    if not(master_dark_exists):
        print ('Making master dark file.  Will be saved to ' + target_dir + master_dark_file) 
        master_dark_exists = makeMasterDark(master_dark_file, target_dir, master_bias_file)
    if not(master_dark_exists):
        print ('Unable to find master dark file, ' + target_dir + master_dark_file + ', and also could not make it.  Returning without processing.')
        sys.exit() 

    #Bias Subtract
    current_image, current_header = biasSubtract(current_image, current_header, master_bias_file)

    current_image, current_header = darkSubtract(current_image, current_header, master_dark_file) 

    c.saveDataToFitsFile(np.transpose(current_image), processed_file, target_dir, header = current_header, overwrite = True, n_mosaic_extensions = 0)    

    #Dark subtract

    #Flat field correct (?) ## Not sure how to do this

    #Determine where the ghosts begin
    spec_axis = 0
    spec_range = determineSpecRowRanges(current_image, spec_axis = spec_axis)
    if spec_range[0] < 0:
        print ('Unable to identify spectrum.  Exiting...')
        sys.exit()

    print ('spec_range = ' + str(spec_range)) 
    #ref_spec = readInDataTextFile(ref_spec_file, spec_file_dir = archival_data_dir,
    #                              n_ignore = n_ignore_spec)
    #throughput = readInDataTextFile(throughput_file, spec_file_dir = archival_data_dir,
    #                                n_ignore = n_ignore_spec)

    ghosts_n_sig_width = 10
    background_fit_order = 2 
    ghosts_high = 1
    ghosts_right = 0
    min_ghosts = 3
    removeGhostByShiftingSpectrum = 0 
    clean_buffer = 5
    n_std_for_most_ghosts = 1.0
    n_std_for_first_ghosts = 4.0 
    min_detections_for_ghost_line = 10
    ghost_search_buffer = [10, 50]
    n_std_for_strong_line = 1.0
    search_binning = 3
    max_line_fit_width = 30.0
    line_width_guess = 1.0
    show_strong_lines = 1
    line_mean_fit_order = 2
    line_amplitude_fit_order = 2
    line_width_fit_order = 2 
    #wavelength_of_pix_solution_guess = [287, 0.83, 0.0] # based on historic, hand-determine solutions
    wavelength_of_pix_solution_guess = [291.0, 0.83] # based on historic, hand-determine solutions

    strong_lines = detectLines(current_image, spec_range,
                               spec_axis = spec_axis, n_std_for_lines = n_std_for_strong_line, 
                               search_binning = search_binning, max_line_fit_width = max_line_fit_width,
                               width_guess = line_width_guess, show_process = show_strong_lines )
    line_pix_vals = c.union([[line_slice[0] for line_slice in line] for line in strong_lines]) 
    line_range = [min(line_pix_vals), max(line_pix_vals)]
    print ('strong_lines = ' + str(strong_lines))
    print ('line_range = ' + str(line_range)) 
    #print ('len(strong_lines) = ' + str(len(strong_lines)))
    #line_masks = generateLineImage(strong_lines, np.shape(current_image), save_fits = 0, spec_axis = spec_axis) 
    strong_line_profiles = [getLineFunction(line,
                                            spec_axis = spec_axis, position_order = line_mean_fit_order,
                                            A_of_x_order = line_amplitude_fit_order, sig_of_x_order = line_width_fit_order)
                            for line in strong_lines]
    #print ('strong_line_profiles = ' + str(strong_line_profiles)) 

    course_line_match_param_range = [[-400.0, -100.0], [1.0, 1.5]]
    course_line_match_param_step = [31, 26] 
    
    archival_data_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/'
    spec_archival_info = {'KR1':{'spec_file':'KR1_lines.txt','n_lines_to_ignore':1},
                          'Gemini':{'spec_file':'GeminiSkyLines.txt','n_lines_to_ignore':14},
                          'throughput':{'spec_file':'OSELOT_throughput.txt','n_lines_to_ignore':0} }

    expected_spectrum = 'KR1' #KR1, Gemini, ...

    throughput_keyword = 'throughput' 
    ref_spec_file, n_ignore_spec = [spec_archival_info[expected_spectrum]['spec_file'], spec_archival_info[expected_spectrum]['n_lines_to_ignore']] 
    throughput_file, n_ignore_throughput = [spec_archival_info['throughput']['spec_file'], spec_archival_info['throughput']['n_lines_to_ignore']]

    # should be like: peak of line profile, mu, as a function of lambda and y: mu(lam, y) 
    mu_of_wavelength_solution, wavelength_of_mu_solution = determineWavelengthSolution(strong_line_profiles, line_range, spec_range, ref_spec_file, 
                                                                                       spec_file_dir = archival_data_dir, throughput_file = throughput_file,
                                                                                       n_ignore_spec = n_ignore_spec, n_ignore_throughput = n_ignore_throughput,
                                                                                       course_search_param_range= course_line_match_param_range, course_search_param_step =  course_line_match_param_step  )
    print ('[mu_of_wavelength_solution, wavelength_of_mu_solution] = ' + str([mu_of_wavelength_solution, wavelength_of_mu_solution]))
    #wavelength_of_mu_solution = np.interp1d([], )

    wavelength_range = [wavelength_of_mu_solution(0, (spec_range[1] - spec_range[0]) // 2), wavelength_of_mu_solution(1023, (spec_range[1] - spec_range[0]) // 2)]
    print('wavelength_range = ' + str(wavelength_range))
    deconvolved_spectrum = deconvolveSpectrum(current_image, spec_range, mu_of_wavelength_solution, strong_line_profiles, throughput_file, archival_data_dir, n_ignore_throughput)

    sys.exit() 


    #strong_line_curves = 
    
    current_image, current_header = correctGhosts(current_image, current_header, spec_range, 
                                                  ghosts_n_sig_width = ghosts_n_sig_width, ghosts_high = ghosts_high,
                                                  ghosts_right = ghosts_right, spec_axis = spec_axis,
                                                  background_fit_order = background_fit_order, min_ghosts = min_ghosts,
                                                  removeGhostByShiftingSpectrum = removeGhostByShiftingSpectrum , clean_buffer = clean_buffer,
                                                  n_std_for_most_ghosts = n_std_for_most_ghosts, n_std_for_first_ghosts = n_std_for_first_ghosts,
                                                  min_detections_for_ghost_line = min_detections_for_ghost_line,
                                                  ghost_search_buffer = ghost_search_buffer) 
    
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

    

    
