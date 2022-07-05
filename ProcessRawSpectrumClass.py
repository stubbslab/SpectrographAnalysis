import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import SashasAstronomyTools as sat

#Use reference emission lines pulled from this paper: https://ui.adsabs.harvard.edu/abs/2003A%26A...407.1157H/abstract
# at this link: http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/407/1157
#Stored in file CombinedReferenceSkyLines.txt
#More accurate, higher wavelength line intensities are available at this link: http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/581/A47
#

"""
Hi.  This file is designed to both reduce and extract spectra from image files taken with the OSELOT Sky Spectograph.
To process a single line, starting from the Command Line:
$ python
>>> import ProcessRawSpectrumClass as prsc
>>> import cantrips as can
>>> date = ['2021', '08', '20']
>>> focus_str = '25p0'
>>> target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut' + ''.join([str(elem) for elem in date]) + '/'
>>> processor = prsc.SpectrumProcessor(target_dir, show_fits = 0)
>>> bias_nums = list(range(19, 39+1, 1))
>>> KR1_nums = list(range(10, 19, 1))
>>> bias_imgs = ['Bias_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in bias_nums]
>>> can.saveListsToColumns(bias_imgs, 'BIAS.list', target_dir)
>>> KR1_imgs = ['KR1_f' + focus_str + '_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in KR1_nums]
>>> can.saveListsToColumns(KR1_imgs, 'KR1.list', target_dir )
>>> ref_spec_solution_file = 'OSELOTSWavelengthSolution.txt'
>>> processor.getWavelengthSolution(ref_spec_images_list = 'KR1.list', ref_spec_solution_file = ref_spec_solution_file, save_new_reference_spectrum = 1, ref_spec = None, show_fits = None)
>>> #As a check, process the refernece wavelength images:
>>> processor.pullCombinedSpectrumFromImages(KR1_imgs, show_fits = None, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked KR1 Spectrum', save_intermediate_images = 0, stacked_image_name = 'StackedKR1Image_img' + str(KR1_nums[0]) + 'To' + str(KR1_nums[-1]))
# >>> for i in range(len(KR1_imgs)):
# >>>     img = KR1_imgs[i]
# >>>     img_num = KR1_nums[i]
# >>>     processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)
#Now we should reinitiate the processor so that we don't try to match reference and sky lines
>>> processor = prsc.SpectrumProcessor(target_dir, show_fits = 0)
>>> dark_sky_nums = list(range(81, 315 + 1, 1))
>>> all_sky_nums = list(range(73, 322 + 1, 1))
>>> dark_sky_imgs = ['sky_f' + focus_str + '_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in dark_sky_nums]
>>> all_sky_imgs = ['sky_f' + focus_str + '_' + '_'.join([str(elem) for elem in date]) + '_' + str(i) + '.fits' for i in all_sky_nums]
>>> processor.pullCombinedSpectrumFromImages(dark_sky_imgs, show_fits = None, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked Spectrum', save_intermediate_images = 0, stacked_image_name = 'StackedSkyImage_img' + str(dark_sky_nums[0]) + 'To' + str(dark_sky_nums[-1]))
>>> for i in range(len(all_sky_imgs)):
>>>     img = all_sky_imgs[i]
>>>     img_num = all_sky_nums[i]
>>>     processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)
>>> processor.plotScaledLineProfilesInTime()
>>> processor.plotLineProfilesInTime()
>>> processor.saveSpecProcessor('FullNight_ut20201130.prsc', save_dir = None, )
# You can reload the saved spectrum processor using the following:
>>> import ProcessRawSpectrumClass as prsc
>>> target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut20201130/'
>>> processor_reloaded = prsc.SpectrumProcessor(target_dir, show_fits = 0)
>>> processor_reloaded.loadSpecProcessor('FullNight_ut20201130.prsc', load_dir = None)

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

def gaussian_deriv(xs, A, mu, sig):
    """
    The functional form a Gaussian derivative, used when searching for spectral lines
        in a 1D spectrum.
    """
    vals = A * np.exp(-(xs - mu) ** 2.0 / (2.0 * sig ** 2.0)) * -2.0 * (xs - mu) / (2.0 * sig ** 2.0)
    return vals

def polyFitVar(ind_vals, dep_vals, fit_order, n_std_for_rejection_in_fit):
    """
    Do a polynomial fit between an independent variable and a dependent
          variable, at some specified fit order.  The notable feature of
          this fit is that data points that are too disparate (indicated by
          the n_std_for_rejection_in_fit) parameter are not used in the fit.
    This polynomial fitting algorithm is used several times in a row,
          and is therefore defined here.
    """
    med = np.median(dep_vals)
    std = np.std(dep_vals)
    ind_vals_for_fit = [ind_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
    ind_vals_for_fit_range = [min(ind_vals_for_fit), max(ind_vals_for_fit)]
    dep_vals_to_fit = [dep_vals[i] for i in range(len(dep_vals)) if abs(dep_vals[i] - med) <= n_std_for_rejection_in_fit * std]
    var_poly_fit = np.polyfit(ind_vals_for_fit, dep_vals_to_fit, fit_order)
    var_funct = lambda val: np.poly1d(var_poly_fit)(val)

    return var_funct

class SpectrumProcessor:

    #For Gemini sky lines:
    # ref_file = 'GeminiSkyBrightness.txt', ref_file_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/', ref_file_delimiter = ' ', n_lines_to_ignore = 14
    def importReferenceSpectrum(self, ref_file = None, ref_data = None, ref_file_delimiter = ' ', n_lines_to_ignore = 0, ref_file_dir = ''):

        if ref_data == None:
            if ref_file == None:
                print ('You must specify either a reference spectrum data file or give reference spectrum data data (wavelength, throughput).  Not redifining reference spectrum interpolator. ')
                return 1
            else:
                ref_data = can.readInColumnsToList(ref_file, file_dir = ref_file_dir, n_ignore = n_lines_to_ignore, delimiter = ref_file_delimiter, convert_to_float = 1, verbose = 0)
        print('ref_data = ' + str(ref_data))
        self.ref_interp = scipy.interpolate.interp1d(ref_data[0], ref_data[1], fill_value = 0.0, bounds_error=False)

        return 1

    #For OSELOTS:
    # throughput_file = 'OSELOT_throughput.txt', throughput_file_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/', delimiter = ' ', n_lines_to_ignore = 0
    def importSystemThroughput(self, throughput_file = None, throughput_data = None, throughput_file_delimiter = ',', n_lines_to_ignore = 1, throughput_file_dir = None, abs_throughput_wavelength = None, abs_throughput_val = None, throughput_SN_cutoff = 1, throughput_fill = np.inf): #, throughput_fill = np.inf ):
        """
        Returns throughput interp in ADU per Rayleigh.  NOTE: you need to divide ADU
            results by the exposure time AND you need to divide by the wavelength
            range that your integrated pixel columns subtend.
        For OSELOTS running on this machine,
        throughput_file = 'OSELOTS_throughput.txt', throughput_file_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/', delimiter = ' ', n_lines_to_ignore = 0
        """
        if throughput_data == None:
            if throughput_file == None and self.throughput_file == None:
                print ('You must specify either a throughput data file or give throughput data (wavelength, throughput).  Not redifining throughput interpolator. ')
                return self.throughput_interp
            elif throughput_file == None:
                throughput_file = self.throughput_file
            if throughput_file_dir == None:
                throughput_file_dir = self.archival_data_dir
            if abs_throughput_wavelength == None:
                abs_throughput_wavelength = self.abs_throughput_wavelength
            if abs_throughput_val == None:
                abs_throughput_val = self.abs_throughput_val
            throughput_data = can.readInColumnsToList(throughput_file, file_dir = throughput_file_dir, n_ignore = n_lines_to_ignore, delimiter = throughput_file_delimiter, convert_to_float = 1, verbose = 0)
            for i in range(len(throughput_data[0])):
                #If the throughput either goes negative drops below some S/N, set throughput to a fill value
                if np.abs(throughput_data[1][i]) / np.abs(throughput_data[2][i]) < throughput_SN_cutoff or throughput_data[1][i] < 0:
                    throughput_data[1][i] = throughput_fill
            #We want throughput to fall to 0 immediately past last well measured point.  So add some fake points, at those bounds
            left_most_throughput_point = np.min([i for i in range(len(throughput_data[1])) if throughput_data[1][i] > 0])
            right_most_throughput_point = np.max([i for i in range(len(throughput_data[1])) if throughput_data[1][i] > 0 ])
            throughput_data = [ can.insertListElement(throughput_data[0], throughput_data[0][left_most_throughput_point] - 1, left_most_throughput_point ), can.insertListElement(throughput_data[1], 0, left_most_throughput_point ),  can.insertListElement(throughput_data[2], 0, left_most_throughput_point ) ]
            right_most_throughput_point = right_most_throughput_point + 1
            throughput_data = [ can.insertListElement(throughput_data[0], throughput_data[0][right_most_throughput_point] + 1, right_most_throughput_point + 1), can.insertListElement(throughput_data[1], 0, right_most_throughput_point + 1),  can.insertListElement(throughput_data[2], 0, right_most_throughput_point + 1) ]
            raw_throughput_interp = scipy.interpolate.interp1d(throughput_data[0], throughput_data[1] , fill_value = throughput_fill, bounds_error=False)
            throughput_data = [ throughput_data[0], raw_throughput_interp(throughput_data[0]) / raw_throughput_interp(abs_throughput_wavelength) / abs_throughput_val ]

        throughput_interp = scipy.interpolate.interp1d(throughput_data[0], throughput_data[1] , fill_value = throughput_fill, bounds_error=False)

        return throughput_interp

    def showSystemThroughput(self, throughput_file_name, throughput_data = None, throughput_file_delimiter = ',', n_lines_to_ignore = 1, throughput_file_dir = None, labelsize = 14, figsize = [7, 4]):
        """
        Make a plot of the OSELOTS throughput as a function of input wavelength.
        This data is measured in lab, not during the night.  This function is provided for convenience, so the user
            can check that throughput data we're using is consistent with expectations.
        """
        f, axarr = plt.subplots(1,1, figsize = figsize)
        if throughput_data == None:
            throughput_data = can.readInColumnsToList(self.throughput_file, file_dir = self.archival_data_dir, n_ignore = n_lines_to_ignore, delimiter = throughput_file_delimiter, convert_to_float = 1, verbose = 0)
        raw_throughput_interp = scipy.interpolate.interp1d(throughput_data[0], throughput_data[1] , fill_value = 0.0, bounds_error=False)
        throughput_data = [ throughput_data[0], raw_throughput_interp(throughput_data[0]) / raw_throughput_interp(self.abs_throughput_wavelength) * self.abs_throughput_val ]
        throughput_plot = axarr.plot(throughput_data[0], 1.0 / np.array(throughput_data[1]), c = 'red', linestyle = '--')[0]
        pivot_line = axarr.axvline(self.abs_throughput_wavelength, color = 'grey', alpha = 0.5, linestyle = '--')
        axarr.legend([throughput_plot, pivot_line], ['OSELOTS throughput', 'Location of absolute measurement'])
        axarr.set_xlabel('Sky wavelength (nm)', fontsize = labelsize)
        axarr.set_ylabel(r'Throughput (ADU Ry$^{-1}$ s$^{-1}$)', fontsize = labelsize)
        axarr.set_title('OSELOTS Absolute Throughput')

        plt.tight_layout()
        plt.savefig(self.archival_data_dir + throughput_file_name)
        return 1



    def CleanCosmics(self, image_dir, image_names, readnoise = 5.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0, maxiter = 2, new_image_prefix = 'crc_'):
        """
        Remove cosmic rays (or sharp, bright features) from an image.  Should be applied to
           any exposure that is longer than a few seconds. Uses the LA cosmics algorithm,
           implemented in python using the cosmics_py3.py package.
        Documentation for LA cosmic is available at: http://www.astro.yale.edu/dokkum/lacosmic/
        """
        for image_name in image_names:
            #print ('Beginning cosmic ray cleaning for image ' + image_dir + image_name)
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

    def plotBiasLevels(self, bias_list = 'BIAS.list'):
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

        print ('Making plot of bias level... ')
        bias_images=np.loadtxt(target_dir + bias_list, dtype='str')
        sat.measureStatisticsOfFitsImages(bias_images, ax = axarr[0], data_dir = target_dir, stat_type = 'mean', show_plot = 0, save_plot = 0, save_plot_name = target_dir + 'BiasLevels.pdf', ylabel = 'Mean counts in image', title = 'OSELOTS Bias Mean Level - ut' + ''.join(self.date))
        sat.measureStatisticsOfFitsImages(bias_images, ax = axarr[1], data_dir = target_dir, stat_type = 'std', show_plot = 0, save_plot = 0, save_plot_name = target_dir + 'BiasLevels.pdf', ylabel = 'Std of counts in image', title = 'OSELOTS Bias Scatter - ut' + ''.join(self.date))
        plt.tight_layout()
        plt.savefig(target_dir + 'BiasLevels.pdf')

        return 1


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
        #c.saveDataToFitsFile(np.transpose(med_bias), master_bias_file, target_dir, header = 'default', overwrite = True, n_mosaic_extensions = 0)
        print ('Master bias files created: ')
        print (target_dir + master_bias_image_file)
        print (target_dir + master_bias_level_file)
        return 1

    def biasSubtract(self, image_data, image_header, master_bias_image_file, master_bias_level_file, bias_dir):
        """
        Subtracts the measured bias level (determined from the makeMasterBias command)
           from the image given by image_data. The 2d bias structure is determined by
           the median stack of the bias images and the median bias level is determined
           by interpolating between the median bias levels of the individual bias
           images, in time.
        """
        start_time_str = image_header[self.obs_time_keyword]
        bias_structure_data, bias_header = can.readInDataFromFitsFile(master_bias_image_file, bias_dir)
        bias_levels = can.readInColumnsToList(master_bias_level_file, bias_dir, n_ignore = 1, delimiter = ',', verbose = 0)
        bias_levels = [[float(start_time) for start_time in bias_levels[0]], [float(bias_level) for bias_level in bias_levels[1]]]
        bias_interp = scipy.interpolate.interp1d(bias_levels[0], bias_levels[1], bounds_error = False, fill_value = 'extrapolate')
        start_time_float = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%SZ').timestamp()
        bias_level = float(bias_interp(start_time_float))
        image_data = image_data - bias_structure_data - bias_level
        image_header['BIASSUB'] = (str(datetime.utcnow() ), 'UTC of Bias subtraction')
        image_header['MBIAS'] = (master_bias_image_file, 'Name of Subtracted Master Bias File')
        image_header['BIASLEVL'] = (bias_level, 'Median bias level')

        return image_data, image_header

    def darkSubtract(self, image_data, image_header, master_dark_file, dark_dir, exp_time_keyword = 'EXPTIME'):
        """
        Subtract the 'dark' (meaning common mode) signal from an image. The
            dark data is produced using the makeMasterDark command.  The dark
            image is scaled by the ratio of the image exposure time (read from)
            the provided header) and the dark exposure time (read from the)
            header of the master dark image.
        """
        dark_data, dark_header = can.readInDataFromFitsFile(master_dark_file, dark_dir)
        exp_time = float(image_header[exp_time_keyword])
        image_data = image_data - dark_data * exp_time
        image_header['DARKSUB'] = (str(datetime.utcnow() ), 'UTC of Dark subtraction')
        image_header['MDARK'] = (master_dark_file, 'Name of Subtracted Master Bias File')

        return image_data, image_header

    def makeMasterDark(self, master_dark_file, target_dir, master_bias_file, master_bias_level_file,
                       dark_list = 'DARK.list', bias_sub_prefix = 'b_',
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
        print ('target_dir + dark_list = ' + str(target_dir + dark_list))
        dark_images=np.loadtxt(target_dir + dark_list, dtype='str')
        if len(np.shape(dark_images)) == 0:
            dark_images = [str(dark_images)]

        #bias correct the images
        exp_times = [-1 for dark_file in dark_images]
        for i in range(len(dark_images)):
            dark_file = dark_images[i]
            single_dark_data, single_dark_header = can.readInDataFromFitsFile(dark_file, target_dir)
            single_dark_data, single_dark_header = self.biasSubtract(single_dark_data, single_dark_header, master_bias_file, master_bias_level_file, self.bias_dir)
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

    def determineSpecRowRanges(self, current_image = None,
                               sum_method = 'sum', n_sig_deriv_spikes_for_spec = 1.5, n_sig_gauss_to_define_spec_width = 2.0 , sig_step = 0.5, showIDedLines = 1, save_perp_spec_image = 0, save_perp_spec_rows = 0, perp_spec_image_name = None, perp_spec_save_file = None):
        """
        Determine the range of image rows subtended by the spectrum.
        This is done by integrating the image ALONG rows, and looking for sudden fall-offs,
            indicating the edge of the spectrum.  The 'sudden' fall-offs are determined
            by looking at first derivative of the boxcar-smoothed binned-along row image.
        This spectral range is used by the class to determine which rows it should integrate
            to reduce the 2d image to a 1d spectrum.
        """

        if perp_spec_image_name == None:
            perp_spec_image_name = self.perp_spec_root + self.figure_suffix
        if perp_spec_save_file == None:
            perp_spec_save_file = self.perp_spec_save_file

        if current_image is(None):
            perp_spec_steps = can.readInColumnsToList(self.target_dir + perp_spec_save_file, delimiter = ' ', convert_to_int = 1, verbose = 0)
            perp_line_step_up, perp_line_step_down = [perp_spec_steps[0][0], perp_spec_steps[1][0]]
            return [perp_line_step_up , perp_line_step_down ]

        perp_spec_axis = (self.spec_axis + 1) % 2
        perp_spec = np.sum(current_image, axis = perp_spec_axis)
        if self.bg_std_buffer > 0:
            perp_spec_smooth = can.smoothList(perp_spec[self.bg_std_buffer:-self.bg_std_buffer], smooth_type = 'boxcar', averaging = 'median', params = [100])
        else:
            perp_spec_smooth = can.smoothList(perp_spec, smooth_type = 'boxcar', averaging = 'median', params = [100])
        perp_spec_peak_loc = np.argmax(perp_spec_smooth)
        perp_len = len(perp_spec_smooth)
        perp_spec_derivs = np.gradient(perp_spec_smooth)
        perp_derivs_med = np.median(perp_spec_derivs)
        perp_derivs_std = np.std(perp_spec_derivs)
        deriv_indeces_above_std = []
        while len(deriv_indeces_above_std) == 0:
            deriv_indeces_above_std = [i for i in range(len(perp_spec_derivs)) if abs (perp_spec_derivs[i] - perp_derivs_med) / perp_derivs_std >= n_sig_deriv_spikes_for_spec ]
            if len(deriv_indeces_above_std) == 0: n_sig_deriv_spikes_for_spec = n_sig_deriv_spikes_for_spec - sig_step
        #Two different ways to picking out the spectrum region:
        #left_slope_around_peak = np.min([index for index in deriv_indeces_above_std if index < perp_spec_peak_loc])
        #right_slope_around_peak = np.max([index for index in deriv_indeces_above_std if index > perp_spec_peak_loc])

        left_slope_around_peak, right_slope_around_peak = [np.argmax(perp_spec_derivs), np.argmin(perp_spec_derivs)]
        #print ('[left_slope_around_peak, right_slope_around_peak] = ' + str([left_slope_around_peak, right_slope_around_peak] ))
        #f, axarr = plt.subplots(2,1)
        #axarr[0].plot(range(perp_len), perp_spec_smooth, color = 'k')
        #axarr[1].plot(range(len(perp_spec_derivs)), perp_spec_derivs, c = 'k')
        #plt.show()

        perp_line_step_up, perp_line_step_down = [left_slope_around_peak, right_slope_around_peak]
        fit_spect_funct = lambda xs, A, l, r, A0: A * np.where(xs < r, 1, 0 ) * np.where(xs > l, 1, 0 ) + A0
        #fit_spect_funct = lambda xs, A, mu, sig, alpha, A0: A * np.exp(-(np.abs(np.array(xs) - mu)/ (np.sqrt(2.0) * sig )) ** alpha) + A0
        init_guess = [np.max(perp_spec_smooth) , perp_line_step_up, perp_line_step_down, np.median(perp_spec_smooth)]
        #init_guess = [np.max(perp_spec) , perp_spec_peak_loc, (perp_line_step_down - perp_line_step_up) / 2.0, 2.0, 0.0]
        fitted_profile = optimize.minimize(lambda params: np.sqrt(np.sum(np.array(fit_spect_funct(list(range(perp_len)), *params) - perp_spec_smooth) ** 2.0)) / perp_len, x0 = init_guess)['x']
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
            axarr[1].set_xlabel('row number (pix)')
            axarr[1].set_ylabel('Deriv of smoothed data (ADU/pix)')
            axarr[0].set_title('Determination of spectrum region by fitting to Sum Perpendicular to Spectrum')
            plt.tight_layout()
            if save_perp_spec_image:
                plt.savefig(perp_spec_image_name)
            if showIDedLines:
                plt.show()

        return [perp_line_step_up + self.bg_std_buffer, perp_line_step_down + self.bg_std_buffer]

    def fullSpecFitFunct(self, ref_xs, ref_ys, lines, ref_params):
        """
        The function that is minimized when determining the line width (i.e. focus) as a function
           of pixel position.  The positions and heights of the lines are taken as given and the
           widths of the lines in pixel space are varied.
        The seeing is assumed to conform to a polynomial of an order specified by the number of
           seeing fit parameters.
        """
        n_points = len(ref_xs)
        seeing_by_pixel_funct = np.poly1d(ref_params)
        fit_ys = np.sum([line[0] * np.exp(-(line[1] - ref_xs) ** 2.0 / (2.0 * seeing_by_pixel_funct(line[1] - n_points // 2) ** 2.0 ))  for line in lines], axis = 0)
        residual = np.sqrt(np.sum((fit_ys - np.array(ref_ys)) ** 2.0))
        return residual

    def simulFitLineWidths(self, ref_xs, ref_ys, ref_emission_lines, init_guess = None):
        """
        Determines the best fit polynomial form of the seeing vs pixel for the spectrum by
           minimizing the sum-of-square difference between the measured spectrum and a
           series of Gaussians at the line locations with widths specified by the polynomial.
        """
        if init_guess == None:
            init_guess = np.zeros(self.seeing_fit_order + 1)
            init_guess[-1] = self.init_seeing_guess
        best_seeing_funct = scipy.optimize.minimize(lambda params: self.fullSpecFitFunct(ref_xs, ref_ys, ref_emission_lines, params), init_guess)
        pivot_x = ref_xs[len(ref_xs) // 2]
        rescaled_best_seeing_params = [best_seeing_funct['x'][0], best_seeing_funct['x'][1] - 2 * best_seeing_funct['x'][0] * pivot_x,  best_seeing_funct['x'][0] * pivot_x ** 2.0 - best_seeing_funct['x'][1] * pivot_x + best_seeing_funct['x'][2]]
        return rescaled_best_seeing_params

    def SingleLineFunctionToMinimize(self, xs, ys, fit_width, fit_funct, fit_params):
        """
        The sum-of-square difference between a spectral line from a set of parameters
           and a subset of the measured spectrum.  This is the function that is
           minimized when fitting each line to the data.
        """
        central_pix = round(fit_params[1])
        if np.isnan(central_pix) or np.isnan(fit_width) or np.isnan(len(xs)) :
            print ('[fit_params, central_pix, fit_width, len(xs)] = ' + str([fit_params, central_pix, fit_width, len(xs)]))
            print ('One of the passed single-fitting line parameters was a nan.  That is not an acceptable set of parameters.  Returning an infinite fit result: ')
            return np.inf
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
        """
        Fit a single line to a section of the 1d OSELOTS spectrum.  The user has some autonomy
           here.  If pedestal is set to 1, the fit is performed with an additional zero-level.
           Otherwise, the fit assumes that the continuum has been approximately removed/that
           the background is not too significant to the fit.  The seeing can also be a free
           parameter or assumed to conform to an image-wide fit.
        The fit is a standard minimizing sum-of-square differences.
        If the user has set the show_fit flag to 1, then the system retuns enough data to make
            the plot in the main call (so that all fitted lines can be shown on the same)
            image.  Otherwise, just the fit is returned.
        """
        if width_from_focus_funct and seeing_fit_funct == None:
            seeing_fit_funct = self.seeing_fit_funct
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
        if np.any( [bounds[i][0] >= bounds[i][1] for i in range(len(bounds[0]))] ):
            print ('init_guess = ' + str(init_guess))
            print ('bounds = ' + str(bounds))
            print('[bounds[i][0] >= bounds[i][1] for i in range(len(bounds[0]))] = ' + str([bounds[i][0] >= bounds[i][1] for i in range(len(bounds[0]))]))
        if np.any(np.isnan(init_guess)): print ('[init_guess, bounds] = ' + str([init_guess, bounds]))
        #if verbose: print ('[fit_width, fit_funct, fit_params, init_guess] = ' + str([fit_width, fit_funct, init_guess] ))
        minim_res = optimize.minimize(lambda fit_params: self.SingleLineFunctionToMinimize(xs, ys, fit_width, fit_funct, fit_params), init_guess, bounds = bounds)
        #fit_res = optimize.minimize(lambda params: np.sqrt(np.sum((fit_funct(fit_xs[params[1] - fit_width:params[1] + fit_width + 1]) - np.array(fit_ys[params[1] - fit_width:params[1] + fit_width + 1])) ** 2.0)), init_guess)
        fit_res = minim_res['x'].tolist()
        if np.any(np.isnan(fit_res)):
            print ('Fitted line returned a nan.  We will reassign to the initial guess: ')
            fit_res = init_guess
        #If the returned best fit line has a height is below the lower bound, there was a flaw in our fit and we need to correct:
        if fit_res[0] < bounds[0][0]:
            fit_res[0] = bounds[0][0]
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
                             seeing_fit_funct = None, background_bin_width = 10, figsize = [10, 5] ):
        """
        Takes in a 1d spectrum (in the form of pixels or wavelengths as the x-variable and
            intensities as the y variable) and identifies lines in that spectrum.  The lines
            are identified by looking for peaks that are sufficiently deviant (sufficiently
            being indicated with the n_pix_above_thresh_for_new_line_in_slice parameter of
            the class).
        The function returns a list of line fits, each of which is a list of the following
            parameters: [Gaussian height, Gaussian mu, Gaussian sigma, pedestal level,
            the sum of square result at the best fit, the sum of squares if we have no lines].
            Those last two are useful diagnostic values, and are not used when using the
            fitted lines to extract values down the line.
        """
        std_thresh = self.std_thresh_for_new_line
        init_fit_width_guess = self.init_seeing_guess

        if max_line_fit_width == None:
            max_line_fit_width = self.width_pix_sample_to_fit_line
        n_pix_above_thresh = self.n_pix_above_thresh_for_new_line_in_slice
        search_width = self.centroid_adjust_size
        if len(peak_guesses) == 0:
            n_pix = len(xs)
            bg_from_binning = [np.median(ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
            bg_ys = ([ys[i] - bg_from_binning[i] for i in range(n_pix)])
            bg_ys_med = np.median(bg_ys)
            bg_ys_std = [np.std(bg_ys[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), n_pix)]) for i in range(n_pix)]
            print ('xs = ' + str(xs))
            print ('len(bg_ys) = ' + str(len(bg_ys)))
            print ('len(bg_ys_std) = ' + str(len(bg_ys_std)))
            pix_vals_above_std = [xs[i] for i in range(1, len(xs) - 1) if bg_ys[i] > bg_ys_med + std_thresh * bg_ys_std[i]]
            #pix_vals_above_std = [pix for pix in xs[1:-1] if bg_ys[pix] > bg_ys_med + std_thresh * bg_ys_std[pix]]
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
            plt.close('all')
            fig = plt.figure(figsize = figsize)
            spec_plot = plt.plot(xs, ys, c = 'blue')[0]
            plt.xlabel('Pixel number (column)')
            plt.ylabel('Binned spectrum (ADU)')
            plt.title('Fits to IDed lines on spectrum slice')
        for j in range(0, n_peak_guesses ):
            peak_guess = peak_guesses[j]
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
            init_guess = [max(max(fit_ys), 0.1), peak_guess, init_fit_width_guess, 0.0, 0.0 ]
            lower_bounds = [-np.inf, init_guess[1] - search_width, 0.2, -np.inf, -np.inf ]
            #lower_bounds = [min(lower_bounds[i], init_guess[i], 0.0) for i in range(len(init_guess))]
            lower_bounds = [max(min(lower_bounds[i], init_guess[i]), 0.0) for i in range(len(init_guess))]
            upper_bounds = [2.0 * init_guess[0], init_guess[1] + search_width, 10.0, init_guess[1] + init_guess[3], np.inf]
            upper_bounds = [max(upper_bounds[i], init_guess[i]) for i in range(len(init_guess))]
            if verbose:
                print ('fit_xs = ' + str(fit_xs))
                print ('fit_ys = ' + str(fit_ys))
                print ('init_guess = ' + str(init_guess))
                print ('lower_bounds = ' + str(lower_bounds))
                print ('upper_bounds = ' + str(upper_bounds))
                print ('search_width = ' + str(search_width) )
                print ('max_line_fit_width = ' + str(max_line_fit_width))
            if show_spec:
                #line_fits_and_plots = self.fitSingleLine(fit_xs, fit_ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
                line_fits_and_plots = self.fitSingleLine(xs, ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
                just_gauss_fit, gauss_on_pedestal_fit, line_fits[j] = [line_fits_and_plots, line_fits_and_plots, line_fits_and_plots[-1]]
                #fit_funct = lambda xs, A, mu, sig, shift, : A * np.exp(-(mu - np.array(xs)) ** 2.0 / (2.0 * sig ** 2.0)) + shift + 0.0 * (np.array(xs) - np.mean(xs))
                #print ('init_guess[0:4] = ' + str(init_guess[0:4]))
                #plt.plot(fit_xs, fit_funct(fit_xs, *(init_guess[0:4])), c = 'orange')
            else:
                #if verbose: print('init_guess = ' + str(init_guess))
                #line_fits[j] = self.fitSingleLine(fit_xs, fit_ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
                line_fits[j] = self.fitSingleLine(xs, ys, init_guess, fit_width = max_line_fit_width, bounds = (lower_bounds, upper_bounds), show_fit = show_spec, pedestal = fit_lines_with_pedestal, width_from_focus_funct = fit_line_width_with_seeing_funct, seeing_fit_funct = seeing_fit_funct, verbose = verbose)
        if n_peak_guesses < 1:
            just_gauss_fit = plt.plot(np.nan, np.nan, '-', color = 'red')[0]
            gauss_on_pedestal_fit = plt.plot(np.nan, np.nan, '-', color = 'green')[0]
            line_fits = []

        if show_spec:
            #print ('[spec_plot, just_gauss_fit, gauss_on_pedestal_fit] = ' + str([spec_plot, just_gauss_fit, gauss_on_pedestal_fit]))
            if n_peak_guesses >= 1:
                plt.legend([spec_plot, just_gauss_fit, gauss_on_pedestal_fit], ['Spectrum on slice', 'Just line fits', 'Line + pedestal fits'])
            else:
                plt.legend([spec_plot, just_gauss_fit, gauss_on_pedestal_fit], ['Spectrum on slice', 'Just line fits (none detected)', 'Line + pedestal fits (none detected)'])
            #plt.show()
            #plt.draw()
            #plt.pause(0.1)
            #plt.close('all')
            plt.draw()
            plt.pause(1)
        return line_fits

    def extendLinesIntoImage(self, range_to_extend, line_extensions, data_to_search, ref_line_ends,
                             binning_search_width = 3, max_sep_per_pix = 5,
                             max_frac_intensity_per_pix = 0.1, line_bg_width = 10,
                             max_seq_fit_failures = 3, max_seq_extreme_amps = 5,
                             bound_tolerance = 0.01):
        """
        When attempting to identify lines across the spectral range of an image, we sometimes
            miss sections where the line is undersampled.  Our method to fixing this problem
            is by "extending" lines that are identified only over a portion of the spectrum
            into the full spectral range.  This function performs that "extension."
        This extension is performed by moving row-by-row beyond the fitted line and looking
            for spectral peaks (with less required S/N) within a range of the original
            fitted line. The fit has a tolerance for a number of rows in which the fit can
            fail beore it gives up trying to extend the line.  It also has a maximum number
            of times that the fit can land outside of some bounds, to protect against
            identifying a different spectral line as an extension of the spectral line of
            interest.
        """
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
                                plot_title = 'Fitted lines',
                                max_sep_per_pix = 5.0, max_frac_intensity_per_pix = 0.1, min_detections = 10,
                                fit_line_center_index = 1, image_range_to_search = None,
                                binning_search_width = 1, line_bg_width = 10, show_start_line_fits = 1):
        """
        Takes in a series of fits of lines in individual spectral slices (usually an averaging of several
            rows) and attempts to stitch them together into a series of continous lines.
        This stitching is done by looking for line fits with a similar centroid position to lines
            identified on the previous slice.  Once this initial matching is done, the function attempts
            to extend partial lines by re-searching for line signals within the spectral range, this time
            with less required S/N than when the lines are first identified.
        """
        continuous_lines = []
        #First, trace out the lines only where they were detected
        #print ('lines_by_slice = ' + str(lines_by_slice))
        for i in range(len(lines_by_slice)):
            prev_continuous_lines = continuous_lines[:]
            pix_val = pix_vals[i]
            lines_in_slice = lines_by_slice[i]
            for line in lines_in_slice:
                #line = line.tolist()
                if line[fit_line_center_index] >= 0:
                    matched_to_line = 0
                    #Find the closest previously matched line, see if it is close enough.
                    #  If it is, lengthen it.  If it isn't, start a new line.
                    if len(prev_continuous_lines) > 0:
                        new_line_dists_from_prev_lines = [ abs(continuous_line[-1][fit_line_center_index+1] - line[fit_line_center_index]) for continuous_line in prev_continuous_lines ]
                        closest_line_index = np.argmin(new_line_dists_from_prev_lines)
                        if new_line_dists_from_prev_lines[closest_line_index] < max_sep_per_pix:
                            continuous_lines[closest_line_index] = continuous_lines[closest_line_index] + [[pix_val] + line]
                            matched_to_line = 1
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
                plt.xlabel('Column (pix)')
                plt.ylabel('Row (pix)')
                plt.title(plot_title)
            plt.show()

        continuous_lines = [ line for line in continuous_lines if len(line) >= min_detections ]

        return continuous_lines


    def consolidateLines(self, line_indeces):
        """
        Takes in a list of line locations and smoothing widths describing the
            identified lines in a 1d spectrum, at a variety of smoothings and
            consolidates that list into a single list of discrete lines.
        Assuming that lines that are sufficiently close together are in fact
           the same line (identified at different smoothings), the function
           consolidates the full list of lines down to a list of unique lines.
           In this matching procedure, priority is given to lines identified
           with smaller widths.
        """
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
                if np.any(line_too_close_to_existing_line):
                    if line_to_merge[1] > full_merged_lines[ [i for i in range(len(line_too_close_to_existing_line))][0] ][1]:
                         line_too_close_to_existing_line_indeces = [i for i in range(len(line_too_close_to_existing_line)) if line_too_close_to_existing_line[i]]
                         full_merged_lines[int(line_too_close_to_existing_line_indeces[0])] = line_to_merge + [width]
                else:
                    full_merged_lines = full_merged_lines + [line_to_merge + [width]]

        return full_merged_lines


    def detectLinesCentersInOneD(self, pixel_vals, spec_slice, stat_slice,
                                 spec_grad_rounding = 5, n_std_for_line = 5,
                                 show = 0, background_bin_width = 10,
                                 convolve_widths = np.arange(1.0, 3.0, 1.0) ):
        """
        Looks for significant emission and absorption lines in a 1D spectrum by
            detecting zero-crossings in the first derivative of the spectrum
            with values left and right of the zero-crossing that are
            sufficiently disparate from 0.  Sufficiently disparate is based on
            a the standard deviations measured from a slice of the spectrum from
            which statistics are computed (the 'stat_slice') and the n_std_for_line
            parameter.
        This search is performed on the spectrum at a range of smoothings.  This is
            done because different lines are best detected with different smoothings,
            particularly when the focus varies with incident wavelength.
        After the lines are picked out at a variety of smoothings (indicated with the
            convolve_widths parameter), they are consolidated into a single set of lines,
            by assuming that sufficiently proximate identified lines are in fact the
            same line.  This is done because strong lines are often detected at many
            smoothing levels.
        """
        spec_derivs = np.gradient(spec_slice)
        spec_derivs = np.around(spec_derivs, spec_grad_rounding)

        stat_derivs = np.gradient(stat_slice)

        stat_derivs = np.around(stat_derivs, spec_grad_rounding)
        deriv_median = np.median(stat_derivs)
        deriv_std = np.std(stat_derivs)
        deriv_std_dict = {pixel_vals[i]: np.std(stat_derivs[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), len(stat_derivs))]) for i in range(len(stat_derivs))}

        #deriv_median = np.median(spec_derivs)
        #deriv_std = np.std(spec_derivs)
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
                                     if ( abs(spec_derivs[absorbtion_deriv_turns[j][0]] - deriv_median) / deriv_std_dict[pixel_vals[(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2]] >= n_std_for_line
                                          and abs(spec_derivs[absorbtion_deriv_turns[j][1]] - deriv_median) / deriv_std_dict[pixel_vals[(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2]] >= n_std_for_line )
                                    ]
        #absorbtion_indeces = [(absorbtion_deriv_turns[j][0] + absorbtion_deriv_turns[j][1]) // 2 for j in range(len(deriv_absorbtion_crossings))
        #                             if ( abs(spec_derivs[absorbtion_deriv_turns[j][0]] - deriv_median) / deriv_std
        #                                  + abs(spec_derivs[absorbtion_deriv_turns[j][1]] - deriv_median) / deriv_std ) >= n_std_for_line * 2
        #                            ]
        emission_indeces = [(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2 for j in range(len(deriv_emission_crossing_indeces))
                                   if ( abs(spec_derivs[emission_deriv_turns[j][0]] - deriv_median) / deriv_std_dict[pixel_vals[(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2]] >= n_std_for_line
                                        and abs(spec_derivs[emission_deriv_turns[j][1]] - deriv_median) / deriv_std_dict[pixel_vals[(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2]] >= n_std_for_line )
                                  ]
        #emission_indeces = [(emission_deriv_turns[j][0] + emission_deriv_turns[j][1]) // 2 for j in range(len(deriv_emission_crossings))
        #                           if ( abs(spec_derivs[emission_deriv_turns[j][0]] - deriv_median) / deriv_std
        #                               +  abs(spec_derivs[emission_deriv_turns[j][1]] - deriv_median) / deriv_std) >= n_std_for_line * 2
        #                          ]
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
            bg_spec_deriv_std_dict = {i: np.std(bg_spec_deriv[max(i - int(background_bin_width / 2), 0):min(i + int(background_bin_width / 2 + 0.5), len(bg_spec_deriv))]) for i in range(len(bg_spec_deriv))}
            emission_line_indeces[width] = can.consolidateList([i for i in range(len(ref_spec_deriv)) if ref_spec_deriv[i] / bg_spec_deriv_std_dict[i] >  n_std_for_line])
            emission_line_indeces[width] = [[i + pixel_vals[0], ref_spec_deriv[i] / bg_spec_deriv_std_dict[i]] for i in emission_line_indeces[width]]
            absorbtion_line_indeces[width] = can.consolidateList([i for i in range(len(ref_spec_deriv)) if ref_spec_deriv[i] / bg_spec_deriv_std_dict[i] < -n_std_for_line ])
            absorbtion_line_indeces[width] = [[i + pixel_vals[0], ref_spec_deriv[i] / bg_spec_deriv_std_dict[i]] for i in absorbtion_line_indeces[width]]
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

        if show:
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
            spec = axarr[0].plot(pixel_vals, spec_slice, c = 'blue')[0]
            axarr[0].set_ylim(np.min(spec_slice) * 0.9, np.max(spec_slice) * 1.1)
            bg = axarr[0].plot(pixel_vals, stat_slice, c = 'red')[0]
            em_line = None
            ab_line = None
            for line in merged_absorbtion_line_indeces: ab_line = axarr[0].axvline(line[0], color = 'orange', linestyle = ':')
            for line in merged_emission_line_indeces: em_line = axarr[0].axvline(line[0], color = 'green', linestyle = ':')
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
                            draw_stats_below = 1, stat_region_buffer = 10, freq_of_plot_single_row_fits = 5,
                            fit_lines_with_pedestal = 1, fit_line_width_with_seeing_funct = 0 ):
        """
        Given a 2d spectrum (image) and a range of rows to search, this function
            attempts to trace spectral lines over that column range.  It searches
            for lines in some averaging of adjecent rows (indicated with the
            coarse_search_binning parameter) that are a certain number of
            standard deviations above some statistical level (drawn from a region
            indicated by the stat_region_buffer).

        This is the core function for generating a 2d model of lines in a spectrum.
           These 2d line traces are then used to determine how the spectrum curves
           in an image.  This curvature is then used to integrate the actual
           spectrum along the curvature, and then lines are searched for on that
           integrated spectrum.
        """
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
        plt.imshow(image)
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
                                                                                            background_bin_width = self.pix_bin_to_meas_background_noise)
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

            #show_process = 0
            line_fits = self.identifyLinesOnSlice(range(len(spec_slice)), spec_slice,
                                             max_line_fit_width = line_fit_width, peak_guesses = guess_line_centers,
                                             show_spec = (i % freq_of_plot_single_row_fits == 0) * show_process, verbose = 0,
                                             fit_lines_with_pedestal = fit_lines_with_pedestal, fit_line_width_with_seeing_funct = fit_line_width_with_seeing_funct,
                                             background_bin_width = self.background_bin_width_for_line_id_in_slice)

            sys.stdout.write("\r{0}".format('Found ' + str(len(line_fits)) + ' line centers for spectrum binned from ' + str(pix_val) + ' to ' + str(pix_val + fit_binning) + '.  (We are ' + str(int ((i+1) / len(pix_vals) * 100)) + '% done)...' ))
            sys.stdout.flush()
            #print ('Found following ' + str(len(line_fits)) + ' line centers for pixel values from ' + str(pix_val) + ' to ' + str(pix_val + fit_binning) + ': ' + str((np.around([line_fit[1] for line_fit in line_fits],3) ).tolist()) )
            all_slices = all_slices + [line_fits]

        return pix_vals, all_slices

    def get2DLineFunction(self, strong_lines, coord_center, anchor_pixel = None):
        """
        Determine the function that describes the spectral emission line geometries
           as a function of image positions (x and y in pixels).  This function
        The fit function at pixel (x,y) is the column-(i.e. x-) value of the
           spectral line that goes through position (x,y) if you trace the line
           back to an 'anchor' row (by default set up to be the center of the
           identified spectral region).
        This fit will be used integrate the 2d spectrum into the final 1d
           spectrum.
        """
        if anchor_pixel == None:
            anchor_pixel = self.anchor_parallel_pix
        coord_center = [0.0, anchor_pixel]
        #col_of_line_at_row = lambda x0s, ys, params: params[0] + (x0s - coord_center[0]) * params[1] + (ys - coord_center[1]) * params[2] + (x0s - coord_center[0]) ** 2.0 * params[3] + (ys - coord_center[1]) ** 2.0 * params[4] + (x0s - coord_center[0]) * (ys - coord_center[1]) * params[5]
        col_of_line_at_row = lambda x0s, ys, params: (x0s) * 1 + (ys - coord_center[1]) * params[0]  + (ys - coord_center[1]) ** 2.0 * params[1] + (x0s - coord_center[0]) * (ys - coord_center[1]) * params[2]
        #line_anchor_intercepts = [[line_elem[2] for line_elem in line if line_elem[0] == anchor_pixel][0] for line in strong_lines]
        line_anchor_intercepts = [line[np.argmin([abs(line_elem[0] - anchor_pixel) for line_elem in line])][2] for line in strong_lines]
        x0s_to_fit = np.array(can.flattenListOfLists([[line_anchor_intercepts[i] for line_elem in strong_lines[i]] for i in range(len(strong_lines))]) )
        ys_to_fit = np.array(can.flattenListOfLists([[line_elem[0] for line_elem in strong_lines[i]] for i in range(len(strong_lines))]) )
        xs_to_fit = np.array(can.flattenListOfLists([[line_elem[2] for line_elem in strong_lines[i]] for i in range(len(strong_lines))]) )

        funct_to_minimize = lambda params: np.sum((col_of_line_at_row(x0s_to_fit, ys_to_fit, params) - xs_to_fit) ** 2.0)
        #Init guess is no curvature
        #init_guess = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        init_guess = [0.0, 0.0, 0.0]
        full_2d_line_fit = optimize.minimize(funct_to_minimize, init_guess)
        print ('full_2d_line_fit = ' + str(full_2d_line_fit))
        full_2d_line_fit_params = full_2d_line_fit['x']
        #full_2d_line_fit_params = init_guess #COMMENT ME OUT!!!! I JUST EXIST TO TEST THE EFFECT OF CURVATURE CORRECITON

        fitted_2d_line_funct = lambda x0, y: col_of_line_at_row (x0, y, full_2d_line_fit_params)
        for line_anchor_intercept in line_anchor_intercepts:
            fitted_line_intercept = fitted_2d_line_funct(line_anchor_intercept, anchor_pixel)
            #print ('[line_anchor_intercept, anchor_pixel, fitted_line_intercept] = ' + str([line_anchor_intercept, anchor_pixel, fitted_line_intercept]))

        return fitted_2d_line_funct

    def getLineFunction(self, line,
                        n_std_for_rejection_in_fit = 3,
                        position_order = 2, A_of_x_order = 2, sig_of_x_order = 2,):
        """
        Takes in a "line" consisting of a series of Gaussian fits to slices of
           the 2d spectrum and returns polynomial fits to each of their
           characteristics (height, width, centroids) as a function of row.
        These polynomial fit parameters for each line are then fit together
           (outside of this function) to determine the image-wide characteristic
           line function.
        """
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

        position_funct = polyFitVar(ind_var, dep_var, position_order, n_std_for_rejection_in_fit)
        A_funct = polyFitVar(ind_var, As, position_order, n_std_for_rejection_in_fit)
        sig_funct = polyFitVar(ind_var, sigs, position_order, n_std_for_rejection_in_fit)

        return [ind_var, position_funct, A_funct, sig_funct]

    def fixContinuousLines(self, lines, line_profiles, n_bins = 21, n_empty_bins_to_drop = 2, n_hist_bins = 21, show_line_matches = 1):
        """
        A known problem when identifying reference lines is that nearby lines
           get confused.  This function "fixes" that by looking for evidence of
           a line that is jumping back and forth along two channels, possibly
           indicating that it is in fact two lines.
        """
        new_lines = [[] for line in lines]
        for i in range(len(lines)):
            line = lines[i]
            line_profile = line_profiles[i]
            #Determine if a line is getting smeared by having neighbors, and fix it if so
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
        """
        This function takes in an image and identifies 2d lines within that image.  It
            performs the two big steps in detecting lines: tracing them over the
            row range and determining which of those identified lines are good
            continuous lines.

        """
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
                                        line_bg_width = bg_fit_width, show_start_line_fits = show_process, plot_title = 'Fitted strong lines')

        return lines

    def readInDataTextFile(self, ref_spec_file,
                           spec_file_dir = '', n_ignore = 0,
                           throughput_file = 'default_throughput.txt'):
        """
        Reads in data from columns into a list, useful for reading
           in a reference spectrum.
        """
        ref_spec = can.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore, convert_to_float = 1, verbose = 0)
        #plt.plot(*ref_spec)
        #plt.show()

        return ref_spec

    def lineMatchingFunction(self, line_pixels, line_wavelengths, n_matches, wavelength_solution, verbose = 0):
        """
        We want to return the integrated distance (in pixel space) between detected lines
            and their wavelengths that they might correspond to, given the wavelength solution.
            We cannot have a single wavelength match two detections (or vice versa) so we
            perform the matching one line at a time, and remove the match from consideration
            after each match is done.
        We assume that line_wavelengths are sorted in the order in which they should be matched.
        We don't perform the matching here, to minimize computation time
        """
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
        """
        Matches lines identified with certain centroids to a set of known reference wavelengths.
           For this matching to work, the function requires a function to go from wavelength to
           pixel centroids. This matching is done recursively until all centroids or all
           wavelengths are matched up.
        This function is also used to determine the wavelength solution for OSELOTS, in which
           case the mu_of_wavelength_funct is varied continuously (at a level above this
           function).
        """
        if len(line_pixels_to_be_matched) == 0 or len(line_wavelengths_to_match) == 0:
            res = np.sum([match[-1] for match in best_matches])
            if not(print_params is None):
                print('[params, res] = ' + str([print_params, res] ))
            return best_matches
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
            min_sep_index = np.argmin(line_seps)
            min_sep = line_seps[min_sep_index]
            min_seps = min_seps + [min_sep]
            min_sep_indeces = min_sep_indeces + [min_sep_index]
        #Then, keep only the closest N lines, where N is either the number of reference lines or the number of lines identified in the spectrum, whichever is smaller
        best_match_index = np.argmin(min_seps)
        best_matches = best_matches + [[line_pixels_to_be_matched[best_match_index], line_wavelengths_to_match[min_sep_indeces[best_match_index]], min_seps[best_match_index]]]
        return self.matchLinesRecursive(line_pixels_to_be_matched, np.array(can.removeListElement(line_wavelengths_to_match.tolist(), min_sep_indeces[best_match_index])), mu_of_wavelength_funct, max_sep_pix = np.inf, best_matches = best_matches, print_params = print_params)


    """
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
    """

    def readFileIntoInterp(self, target_file, target_dir, n_ignore, convert_to_float = 1, delimiter = ','):
        """
        Reads in a file to a 1d interpreter.  Useful for reading in data like
            throughput functions into an easy-to-use class interpreter.
        """
        cols = can.readInColumnsToList(target_file, file_dir = target_dir, n_ignore = n_ignore, convert_to_float = convert_to_float, delimiter = delimiter, verbose = 0)
        interp = can.safeInterp1d(cols[0], cols[1])
        return interp

    """
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
    """


    #Curve fitting doesn't work very well.  Is there a way that we could just detect where lines are and determine where they are supposed to be?
    def determineWavelengthSolution(self, line_solutions, spec_range, ref_spec_file, ref_lines_file, #line_median_areas,
                                    spec_file_dir = '', throughput_file = 'default_throughput.txt',
                                    n_ignore_spec = 0, n_ignore_lines = 0, n_ignore_throughput = 1,
                                    wavelength_solution_drift_order = 2,
                                    coarse_search_param_range= [[-1000.0, -100.0], [1.0, 1.5]], coarse_search_param_step = [51, 51],
                                    solution_save_dir = '', save_solution_image = 1,
                                    save_solution = 1, show_solution = 1,  ):
        """
        Determines the wavelength solution for the spectrograph, given fits to 2d lines in an image
           (line_solutions) and the spectral text file with the reference spectrum (ref_spec_file).
           The function does the wavelength matching by doing the wavelength solution for a
           variety of wavelength solution values, assuming a linear solution (the
           coarse_search_param_range parameter).  Once the best of those coarse wavelength
           solutions is identified, the algorithm runs a sum-of-squares minimization around that
           starting seed.
        The wavelength matching (with a fixed wavelength solution) is done recursively,  using the
            matchLinesRecursive function.
        """
        coarse_search_param_range = self.ref_param_holder.getCoarseSearchParamRange()
        coarse_search_param_step = self.ref_param_holder.getCoarseSearchNParamStep()
        #self.ref_spec_lines = can.readInColumnsToList(ref_spec_file, file_dir = spec_file_dir, n_ignore = n_ignore_spec, convert_to_float = 1, delimiter = ',')
        self.ref_spec_just_lines = can.readInColumnsToList(ref_lines_file, file_dir = spec_file_dir, n_ignore = n_ignore_lines, convert_to_float = 1, delimiter = ',', verbose = 0)

        print ('[throughput_file, spec_file_dir, n_ignore_throughput] = ' + str([throughput_file, spec_file_dir, n_ignore_throughput]))
        throughput_interp = self.readFileIntoInterp(throughput_file, spec_file_dir, n_ignore_throughput, convert_to_float = 1)
        #self.ref_spec_lines[1] = [self.ref_spec_lines[1][0] / (((self.ref_spec_lines[0][1] + self.ref_spec_lines[0][0]) / 2.0 - self.ref_spec_lines[0][0]) * 2.0)] + [self.ref_spec_lines[1][i] / (-(self.ref_spec_lines[0][i] + self.ref_spec_lines[0][i-1]) / 2.0 + (self.ref_spec_lines[0][i+1] + self.ref_spec_lines[0][i]) / 2.0) for i in range(1, len(self.ref_spec_lines[0])-1)] + [self.ref_spec_lines[1][-1] / (((self.ref_spec_lines[0][-1] + self.ref_spec_lines[0][-2]) / 2.0 - self.ref_spec_lines[0][-2]) * 2.0)]
        #self.ref_spec_lines[1] = (np.array(self.ref_spec_lines[1]) * throughput_interp(self.ref_spec_lines[0])).tolist()

        slice_pixels = range(*spec_range)
        fitted_spectra = [[0.0 for pix in slice_pixels] for guess in range(self.wavelength_solution_order + 1)]
        best_match_params1 = [0.0, 0.0]
        best_match_params = [0.0, 0.0]
        best_matches = []
        best_match_sep = np.inf
        best_match_pixel_indeces = []
        best_match_wavelength_indeces = []

        #median_line_centers = [np.median([line_solution[1](pix) for pix in slice_pixels]) for line_solution in line_solutions]
        median_line_centers = [line_solution[1] for line_solution in line_solutions]
        print ('median_line_centers = ' + str(median_line_centers))

        wavelength_of_mu = lambda mu, lam0, lam1, lam2: lam0 / self.wavelength_scaling[0] + lam1 / self.wavelength_scaling[1] * mu + lam2 / self.wavelength_scaling[2] * mu ** 2.0
        mu_of_wavelength = lambda lam, mu0, mu1, mu2: mu0 / self.wavelength_scaling[0] + mu1 / self.wavelength_scaling[1] * lam + mu2 / self.wavelength_scaling[2] * lam ** 2.0
        #wavelength_of_mu = lambda mu, lam0, lam1: lam0 / self.wavelength_scaling[0] + lam1 / self.wavelength_scaling[1] * mu
        #mu_of_wavelength = lambda lam, mu0, mu1: mu0 / self.wavelength_scaling[0] + mu1 / self.wavelength_scaling[1] * lam
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
        print ('[np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), best_match_params] = ' + str([np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), best_match_params]))
        best_match_linear = optimize.minimize(lambda test_params: np.sum([match[-1] for match in self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, *(test_params.tolist() + [0.0])), print_params = None )]), best_match_params  , method = 'Nelder-Mead')

        print ('linear best match params: ' + str(best_match_params))
        best_match_quadratic = optimize.minimize(lambda test_params: np.sum([match[-1] for match in self.matchLinesRecursive(np.array(median_line_centers), np.array(self.ref_spec_just_lines[0]), lambda pixels: mu_of_wavelength(pixels, *test_params), print_params = None )]), best_match_linear['x'].tolist() + [0.0]  , method = 'Nelder-Mead')
        best_match_quadratic['x'] = np.array([ best_match_quadratic['x'][i] / self.wavelength_scaling[i] for i in range(len(best_match_quadratic['x'])) ])
        print ('quadratic best match params: ' + str(best_match_quadratic))
        best_match = best_match_linear['x'].tolist() + [0.0]
        #best_match = best_match_quadratic['x'].tolist()
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
            #fitted_pixels = np.poly1d(best_fit_terms)(self.ref_spec_lines[0])
            #fitted_pixel_interp = interpolate.interp1d(fitted_pixels, self.ref_spec_lines[1], bounds_error = False, fill_value = 0.0)
            #expected_spectrum = np.sum([np.array(fitted_pixel_interp(pix_part)) * 1.0 / np.sqrt(2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0) * np.exp(-(np.array(self.parallel_ref_spec_pixels) - pix_part) ** 2.0 / (2.0 * self.seeing_ref_fit_funct(pix_part) ** 2.0)) for pix_part in np.arange(self.parallel_ref_spec_pixels[0], self.parallel_ref_spec_pixels[-1], 0.1)], axis = 0) #]
            f, axarr = plt.subplots(2,1, figsize = self.default_figsize)
            #axarr[0].plot(self.parallel_spec_pixels, fitted_pixel_interp(self.parallel_spec_pixels) )
            parallel_ref_spec_wavelengths = [wavelength_of_mu_solution(pix) for pix in self.parallel_ref_spec_pixels]
            #print ('[len(self.parallel_ref_spec_pixels), len(parallel_ref_spec_wavelengths), len(expected_spectrum)] = ' + str([len(self.parallel_ref_spec_pixels),  len(parallel_ref_spec_wavelengths), len(expected_spectrum)]))
            #spec_plot = axarr[0].plot(parallel_ref_spec_wavelengths, np.array(expected_spectrum) / np.max(expected_spectrum))[0]
            normalized_line_intensities = [ self.ref_spec_just_lines[1][i] / throughput_interp(self.ref_spec_just_lines[0][i]) for i in range(len(self.ref_spec_just_lines[0])) ]
            normalized_line_intensities = (normalized_line_intensities / np.max(normalized_line_intensities)).tolist()
            spec_plot = axarr[0].scatter(self.ref_spec_just_lines[0], normalized_line_intensities, marker = 'x', c = 'cyan')
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

    """
    def create2DWavelengthSolutionCallableFunctions(self, wavelength_polyfit):

        #fitted_wavelength_solution = [np.poly1d(fit_to_solution_term) for fit_to_solution_term in wavelength_polyfit]
        mu_of_wavelength_solution = lambda lam, y: np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution])(lam)
        wavelength_of_mu_solution = lambda mu, y: ((np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution]) - mu).roots)[0]

        return [mu_of_wavelength_solution, wavelength_of_mu_solution]
"""
    #def loadWavelengthSolution(solution_file, load_dir = ''):
    #    fitted_wavelength_solution = np.load(archival_data_dir + solution_file)
    #    mu_of_wavelength_solution = lambda lam, y: np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution])(lam)
    #    wavelength_of_mu_solution = lambda mu, y: ((np.poly1d([solution_term(y) for solution_term in fitted_wavelength_solution]) - mu).roots)[0]
    #
    #    return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    def createWavelengthSolutionCallableFunctions(self, wavelength_poly_terms):
        """
        Given a set of polynomial terms describing the wavelength solution (pixel
           to wavelength), this function returns TWO callable functions: the pixel
           to wavelength solution computed directly from these parameters AND a
           wavelength to pixel solution, determined by inverting this polynomial.
        The wavelength solution for a night is saved as a plain text file, and
           can be reloaded.  This function takes those numbers (read by another
           function) and returns a callable polynomial.
        """
        print ('wavelength_poly_terms = ' + str(wavelength_poly_terms))
        mu_of_wavelength_solution = lambda lam: np.poly1d(wavelength_poly_terms)(lam)
        if self.wavelength_solution_order > 1:
            wavelength_of_mu_solution = lambda mu: ((np.poly1d(wavelength_poly_terms) - mu).roots)[-1]
        else:
            wavelength_of_mu_solution = lambda mu: ((np.poly1d(wavelength_poly_terms) - mu).roots)
        return [mu_of_wavelength_solution, wavelength_of_mu_solution]

    """
    #New curvature dict takes an x0 at the anchor pixel returns an x(y) function to parameterize the curve that intersecs x0 at the anchor pixel
    def determineMacroLineCurvatureDictNew(self, spec_range, strong_line_profiles, pix_index = 0, mu_index = 1, line_fit_order = 2, curvature_fit_order = 2, anchor_parallel_pix = None):
        if anchor_parallel_pix == None:
            anchor_parallel_pix = self.anchor_parallel_pix
        print ('anchor_parallel_pix = ' + str(anchor_parallel_pix))
        poly_fits = [ np.polyfit([], []) for i in range(line_fit_order) ]

    #Old curvature dict takes an x and a y and gives back the corresponding x0 where the contour that passes through x,y intersects the anchor pixel
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
    """

    def measureFullPixelSpectrum(self, current_image, spec_range, full_2d_line_profile_fit,
                                     anchor_cols_to_integrate_from = None, min_pix = 0, max_pix = 1023):
        """
        This function does the actual integration of a 2d spectrum into a 1d spectrum, including
            an integration along the line curvature (given by the full_2d_line_profile_fit).  The
            trace along curvature integrates a column that is one-pixel wide along columns.  The
            contribution of a pixel to a section of the spectrum is determined in proportion to
            the portion of the pixel in that column.  For example, if the line curvature passes
            has value 100.4 at row 400, pixels 99 and 100 of row 400 respectively contribute 10%
            and 90% to the value added to the overall integration.  This ensures that the total
            flux of the spectrum is the same as binning along columns directly.
        """
        spec_pixels = np.array(list(range(*spec_range)))
        if anchor_cols_to_integrate_from == None:
            if self.spec_axis == 0:
                anchor_cols_to_integrate_from  = range(len(current_image[0, :]))
            else:
                anchor_cols_to_integrate_from = range(len(current_image[:, 0]))
        #anchor_cols_to_integrate_from = range(400, 450)
        intensities = [ 0.0 for i in range(len(anchor_cols_to_integrate_from)) ]
        n_cols = len(anchor_cols_to_integrate_from)
        for i in range(n_cols):
            sys.stdout.write("\r{0}".format('Computing intensity for wavelength that intersects pixel (x, y) = ' + str((i, self.anchor_parallel_pix)) + '.  (We are ' + str(int (i / n_cols * 100)) + '% done)...' ))
            sys.stdout.flush()
            intensities_of_trace = [0.0 for row in spec_pixels]
            anchor_col = anchor_cols_to_integrate_from[i]
            anchor_left_col, anchor_right_col = [anchor_col - 0.5, anchor_col + 0.5]
            cols_of_left_trace = [full_2d_line_profile_fit(anchor_left_col, row) for row in spec_pixels]
            cols_of_right_trace = [full_2d_line_profile_fit(anchor_right_col, row) for row in spec_pixels]
            #Now we need to add up the intensity betwen the trace of the left pixel edge and right pixel
            #  edge, adding counts from pixels in proportion to how much of the pixel lies between the trace

            for j in range(len(spec_pixels)):
                spec_pix = spec_pixels[j]
                left_col = cols_of_left_trace[j]
                right_col = cols_of_right_trace[j]
                current_col = left_col
                included_col_proportions = {}
                while current_col < right_col:
                    next_col_edge = np.floor(current_col + 1.5) - 0.5
                    if next_col_edge >= right_col:
                        next_col_edge = right_col
                    col_prop = next_col_edge - current_col
                    included_col_proportions[int(current_col + 0.5)] = col_prop
                    current_col = next_col_edge
                if self.spec_axis == 0:
                    included_intensities = [current_image[spec_pix, col] * included_col_proportions[col] if (col > min_pix and col < max_pix) else 0.0 for col in included_col_proportions.keys()  ]
                else:
                    included_intensities = [current_image[col, spec_pix] * included_col_proportions[col] if (col > min_pix and col < max_pix) else 0.0 for col in included_col_proportions.keys() ]
                #print ('included_intensities = ' + str(included_intensities))
                intensities_of_trace[j] = np.sum(included_intensities)
            #print ('intensities_of_trace = ' + str(intensities_of_trace ))
            intensities[i] = np.sum(intensities_of_trace)
        print ('')
        #if self.show_fits:

        intensity_interp = interpolate.interp1d(anchor_cols_to_integrate_from, intensities)
        if self.show_fits:
            plt.close('all')
            f, axarr = plt.subplots(1,2, figsize = [10, 5])
            axarr[0].plot(anchor_cols_to_integrate_from, intensities , c = 'cyan')
            axarr[0].set_xlabel('Anchor column of trace (pix)')
            axarr[0].set_ylabel('Integrated intensity (ADU)')
            axarr[1].imshow(current_image)
            plt.show()
        return [anchor_cols_to_integrate_from, intensity_interp ]


    """
    #line_profile_dict takes in a pixel y and then a pixel x to determine the pixel x position where a line at that xy would trace to at y = 0 (center of the spectrum)
    def measureFullPixelSpectrumOld(self, current_image, spec_range, undo_line_curvature_dict,
                                 mu_index = 1, width_index = 3, width_fit_order = 2 ):
        spec_pixels = list(range(*spec_range))
        intensities = [ [[], []] for i in range(len(spec_pixels)) ]
        spec_slice_pixels = np.array(list(range(np.shape(current_image)[(self.spec_axis + 1) % 2])))

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
            intensities[i] = [[undo_line_curvature_dict[pix](spec_slice_pixel) for spec_slice_pixel in spec_slice_pixels ], spec_slice[:]]

        print ('Intensities computed.')
        #print ('intensities[0] = ' + str(intensities[0]))
        #print ('intensities[-1] = ' + str(intensities[-1]))
        intensity_interps = [can.safeInterp1d(intensity[0], intensity[1]) for intensity in intensities]
        #full_spec_interp = can.safeInterp1d( [lam for lam in np.arange(*wavelength_range, wavelength_step)], [np.sum([interp(lam) for interp in intensity_interps]) for lam in np.arange(*wavelength_range, wavelength_step)] )
        full_spec_interp = can.safeInterp1d( spec_slice_pixels, [np.sum([interp(slice_pixel) for interp in intensity_interps]) for slice_pixel in spec_slice_pixels] )
        #Don't do the curvature correcting
        #full_spec_interp = can.safeInterp1d( spec_slice_pixels, np.sum(current_image[spec_range[0]:spec_range[1], :], axis = 0))

        return spec_slice_pixels, full_spec_interp
    """

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

    def correctBackground(self, image, spec_range, background_buffer, background_size, background_low = 1, ):
        """
        Attempts to correct the background of the spectrum by inteprolating the
           image directly above and/or below the spectral region.  This
           background interpolation works okay, but not perfectly.  This is
           done using the fitMaskeImage interpolation method from the
           cantrips.py library.
        We have found that this method works imperfectly, as the background
           structure is typically non-uniform.  Therefore, we no longer do this
           by default, opting for a direct-light level correction.
        """
        background_cut_pixels = [int(self.mu_of_wavelength_solution(self.background_cut_wavelengths[0])), int(self.mu_of_wavelength_solution(self.background_cut_wavelengths[1]))]

        image_shape = np.shape(image)
        print ('[spec_range, background_cut_pixels, self.bg_fit_range] = ' + str([spec_range, background_cut_pixels, self.bg_fit_range] ))
        #masked_image[spec_range[0]:spec_range[1], background_cut_pixels[0]:background_cut_pixels[1] ] = 0.0
        clip_section = [[int(max(spec_range[0] - self.bg_fit_range[0][0], 0)), int(min(spec_range[1] + self.bg_fit_range[0][0], image_shape[0]))],
                               [int(max(background_cut_pixels[0] - self.bg_fit_range[1][0], 0)), int(min(background_cut_pixels[1] + self.bg_fit_range[1][1], image_shape[1]))]]
        print ('clip_section = ' + str(clip_section))
        clipped_image = image[clip_section[0][0]:clip_section[0][1], clip_section[1][0]:clip_section[1][1]]
        masked_image = clipped_image.copy()
        can.saveDataToFitsFile(np.transpose(clipped_image), 'ClippedImage.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        clipped_image_shape = np.shape(clipped_image)
        fit_x_lims = [-clipped_image_shape[1] // 2, clipped_image_shape[1] // 2 ]
        fit_y_lims = [-clipped_image_shape[0] // 2, clipped_image_shape[0] // 2]
        spec_box_section = [[background_cut_pixels[0] - clip_section[1][0] - clipped_image_shape[1] // 2, background_cut_pixels[1] - clip_section[1][0] - clipped_image_shape[1] // 2],
                            [spec_range[0] - clip_section[0][0] - clipped_image_shape[0] // 2, spec_range[1] - clip_section[0][0] - clipped_image_shape[0] // 2] ]

        mask_section = [[background_cut_pixels[0] - clip_section[1][0],background_cut_pixels[1] - clip_section[1][0]], [spec_range[0] - clip_section[0][0],spec_range[1] - clip_section[0][0]] ]
        #mask_section = clip_section
        masked_image[mask_section[1][0]:mask_section[1][1], mask_section[0][0]:mask_section[0][1]] = 0.0
        can.saveDataToFitsFile(np.transpose(masked_image), 'ImageWithSpecMask.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        can.saveDataToFitsFile(np.transpose(image), 'ImageWithBackgound.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        background_fit, background_fit_funct, clipped_background = can.fitMaskedImage(clipped_image, mask_region = spec_box_section, fit_funct = 'poly3', verbose = 0, param_scalings = [1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0], x_lims = fit_x_lims, y_lims = fit_y_lims, )
        x_mesh, y_mesh = np.meshgrid(range(0, image_shape[1]) , range(0, image_shape[0]))
        fit_x_mesh = x_mesh - clip_section[1][0] - clipped_image_shape[1] // 2
        fit_y_mesh = y_mesh - clip_section[0][0] - clipped_image_shape[0] // 2
        background = background_fit_funct(fit_x_mesh, fit_y_mesh)
        can.saveDataToFitsFile(np.transpose(background), 'BackgroundFit.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        image = image - background
        can.saveDataToFitsFile(np.transpose(image), 'BackgroundSubtractedImage.fits', self.target_dir, header = 'default', n_mosaic_extensions = 0)
        return image


    def processImages(self, spec_files_to_reduce = None, do_bias = None, do_dark = None, crc_correct = None, cosmic_prefix = None, save_stacked_image = None, save_image_name = None, apply_background_correction = 0, redetermine_spec_range = 1, scatter_correction_file = None, apply_scatter_correction = 1 ):
        """
        This processes the images specified in the spec_files_to_reduce text
            file.  This reduction includes a series of steps, including bias,
            dark (i.e. single mode) illumination patterns, cosmic ray
            correction, background correction (ideally not needed), and
            scatter correction.
        This is a relatively standard image reduction process, except for
            the scatter correction file.
        """
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
            master_bias_exists = os.path.isfile(self.bias_dir + self.master_bias_image_file)
            if not(master_bias_exists) or self.redo_master_bias:
                print ('Making master bias files.  Will be saved to: ' )
                print(self.bias_dir + self.master_bias_image_file)
                print(self.bias_dir + self.master_bias_level_file)
                master_bias_exists = self.makeMasterBias(self.master_bias_image_file, self.master_bias_level_file, self.bias_dir)
                self.redo_master_bias = 0
            if not(master_bias_exists):
                print ('Unable to find master bias file, ' + self.bias_dir + self.master_bias_file + ', and also could not make it.  Returning without processing.')
                return 0

        #[OPTIONAL] Make master dark
        if do_dark :
            master_dark_exists = os.path.isfile(self.dark_dir + self.master_dark_file)
            if not(master_dark_exists) or self.redo_master_dark:
                print ('Making master dark file.  Will be saved to ' + self.dark_dir + self.master_dark_file)
                master_dark_exists = self.makeMasterDark(self.master_dark_file, self.dark_dir, self.master_bias_image_file, self.master_bias_level_file, dark_list = self.master_dark_list)
                self.redo_master_dark = 0
            if not(master_dark_exists):
                print ('Unable to find master dark file, ' + self.dark_dir + self.master_dark_file + ', and also could not make it.  Returning without processing.')
                #sys.exit()

        #Bias and dark subtract
        for i in range(len(spec_files_to_reduce)):
            if do_bias or do_dark:
                self.current_images[i], self.current_headers[i] = self.biasSubtract(self.current_images[i], self.current_headers[i], self.master_bias_image_file, self.master_bias_level_file, self.bias_dir)
            if do_dark:
                self.current_images[i], self.current_headers[i] = self.darkSubtract(self.current_images[i], self.current_headers[i], self.master_dark_file, self.dark_dir)

        exp_times = [float(header[self.fits_exp_time_keyword]) for header in self.current_headers]
        self.current_images = [ self.current_images[i] / exp_times[i] for i in range(len(self.current_images)) ]
        self.current_image, self.current_header = self.stackImage(self.current_images, self.current_headers, combine = 'mean' )
        if (self.spec_range == None and apply_background_correction) or redetermine_spec_range:
            self.spec_range = self.determineSpecRowRanges(current_image = self.current_image, showIDedLines = self.show_fits, save_perp_spec_image = self.save_perp_spec_image, perp_spec_image_name = self.target_dir + self.perp_spec_image_name)
        else:
            print ('Using reference row range of spectrum... ')
        if apply_background_correction:
            self.current_image = self.correctBackground(self.current_image, self.spec_range, self.background_buffer, self.background_size, background_low = self.background_low, )
        #self.current_image = np.median(self.current_images, axis = 0)
        if not(scatter_correction_file is None) and apply_scatter_correction:
            scatter_correction = np.transpose(can.readInDataFromFitsFile(scatter_correction_file, self.scatter_data_dir)[0])
            if save_stacked_image and save_image_name != None:
                can.saveDataToFitsFile(np.transpose(self.current_image), save_image_name.split('.')[0] + '_NoScatterCorrection.fits' , self.target_dir, header = self.current_header)
                can.saveDataToFitsFile(np.transpose(scatter_correction), 'ScatterCorrection.fits' , self.target_dir, header = self.current_header)
            self.current_image = self.current_image - scatter_correction
        elif apply_scatter_correction:
            print ('Could not find file: "' + str(scatter_correction_file) + '", to do scatter correction.  So no scatter correction applied.')

        self.current_header = self.current_headers[0]
        self.current_header[self.fits_exp_time_keyword] = 1
        if save_stacked_image and save_image_name != None:
            can.saveDataToFitsFile(np.transpose(self.current_image), save_image_name , self.target_dir, header = self.current_header)

        print ('spec_files_to_reduce = '  + str(spec_files_to_reduce ))
        if self.remove_intermed_images and crc_correct:
            [os.remove(self.target_dir + file) for file in spec_files_to_reduce]

        return self.current_image, self.current_header

    def loadWavelengthSolution(self, solution_file = None, load_dir = ''):
        """
        This function reads in the plaintext file that includes the determined wavelength
            solution for OSELOTS.  It feeds those read in numbers to the
            createWavelengthSolutionCallableFunctions function to acquire a callable
            set of functions to go between wavelength and line centroids.
        """
        if solution_file == None:
            solution_file = self.ref_spec_solution_file
        #wavelength_poly_terms = np.load(archival_data_dir + solution_file)
        loaded_solution = can.readInColumnsToList(solution_file, file_dir = self.target_dir, n_ignore = 0, convert_to_float = 1, verbose = 1)[0]
        self.anchor_parallel_pix, self.wavelength_poly_terms = [int(loaded_solution[0]), loaded_solution[1:]]
        print ('wavelength_poly_terms = ' + str(self.wavelength_poly_terms))
        #mu_of_wavelength_solution, wavelength_of_mu_solution = create2DWavelengthSolutionCallableFunctions(wavelength_poly_terms)
        self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(self.wavelength_poly_terms)
        return self.mu_of_wavelength_solution, self.wavelength_of_mu_solution


    def showCurvatureFits(self, raw_lines, fitted_line_profiles, full_2d_line_fit,
                          n_lines_per_row = 8, plot_elem_size = [1.5, 3], line_colors = ['k','r','g'], line_styles = ['-','--','--'], line_alphas = [1.0, 0.5, 0.5],
                          xlabel = 'Column (pix)', ylabel = 'Row (pix)'):
        """
        This function plots the fitted line profiles in a 2d image.  The line colors
           are displayed in (arbitrary) increasing order.  This function is a Useful
           diagnostic to see if the plots are being correctly traced out.
        """
        n_lines = len(raw_lines)
        n_cols = min(n_lines, n_lines_per_row)
        n_rows = int(np.ceil(n_lines / n_lines_per_row))
        plot_h_size = n_cols * plot_elem_size[0]
        plot_v_size = n_rows * plot_elem_size[1]
        print ('[n_lines, n_cols, n_rows] = ' + str([n_lines, n_cols, n_rows] ))
        plt.close('all')
        spec_range_pix = list(range(self.spec_range[0], self.spec_range[1]))
        n_subplots = n_rows * n_cols
        f, axarr = plt.subplots(n_rows, n_cols, figsize = (plot_h_size, plot_v_size), squeeze = False, sharey = 'all')
        for i in range(n_lines):
            print ('Working on line ' + str(i) + ' of line ' + str(n_lines))
            raw_line = raw_lines[i]
            line_profile = fitted_line_profiles[i]
            anchor_parallel_pixel = self.anchor_parallel_pix
            anchor_col = line_profile[1](anchor_parallel_pixel)
            #fitted_line_column = int(np.round(line_profile[1](anchor_parallel_pixel)) )
            #full_curvature_profile = [undo_line_curvature_dict(row_pix)() for row_pix in spec_range_pix]
            col_num = i % n_lines_per_row
            row_num = i // n_lines_per_row
            ax = axarr[row_num, col_num]
            ax.plot([line_elem[2] for line_elem in raw_line], [line_elem[0] for line_elem in raw_line], color = line_colors[0], linestyle = line_styles[0], alpha = line_alphas[0])
            ax.plot([line_profile[1](x) for x in line_profile[0]], line_profile[0], color = line_colors[1], linestyle = line_styles[1], alpha = line_alphas[1])
            ax.plot([full_2d_line_fit(anchor_col, pix) for pix in spec_range_pix], spec_range_pix, color = line_colors[2], linestyle = line_styles[2], alpha = line_alphas[2])
            if row_num == 0:
                ax.set_xlabel(xlabel)
            if col_num == 0:
                ax.set_ylabel(ylabel)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
        for i in range(n_lines, n_subplots):
            col_num = i % n_lines_per_row
            row_num = -1
            ax = axarr[row_num, col_num]
            ax.set_xticks([])
        plt.show()
        return 1


    def integrateImageAlongCurvature(self, image_to_integrate, header, n_std_for_strong_line, save_intermediate_images = 0, define_new_anchor_pix = 0, redetermine_spec_range = 0, reference_image = 0, determine_new_line_curvature = 0, apply_background_correction = 0 ):
         """
         Takes a 2d image of a spectrum and returns a 1d spectrum, integrated
             along the measured curvature of the lines.  This function does
             so by stitching together many of the subfunctions above.  No heavy
             analysis is done in this function - it just stitches together
             other methods.
         """
         if self.spec_range == None or redetermine_spec_range:
             self.spec_range = self.determineSpecRowRanges(current_image = image_to_integrate, showIDedLines = self.show_fits, save_perp_spec_image = self.save_perp_spec_image, perp_spec_image_name = self.target_dir + self.perp_spec_image_name)
         else:
             print ('Using reference spectral range... ')
         print ('self.spec_range = ' + str(self.spec_range))
         if define_new_anchor_pix:
             self.anchor_parallel_pix = ( self.spec_range[1] + self.spec_range[0] ) // 2

         if save_intermediate_images:
             can.saveDataToFitsFile(np.transpose(self.current_image), self.processed_file, self.target_dir, header = self.current_header, overwrite = True, n_mosaic_extensions = 0)
             print ('Just saved processed file to: ' + self.target_dir + self.processed_file)

         if determine_new_line_curvature or self.strong_line_2d_profile_fit == None:
             if reference_image:
                 line_fit_width = self.width_pix_sample_to_fit_ref_line
             else:
                 line_fit_width = self.width_pix_sample_to_fit_line
             self.strong_lines = self.detectLinesInImage(self.current_image, self.spec_range,
                                                   n_std_for_lines = n_std_for_strong_line, line_fit_width = line_fit_width,
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

             #These give line fits in 2d => You provide an x0 (column along the central line) and a y (row number) and
             #   and these gives the x where the line profile that passes through x0, central line intersects row y.
             self.strong_line_2d_profile_fit = self.get2DLineFunction(self.strong_lines, [0.0, self.anchor_parallel_pix])
             if self.show_fits:
                 self.showCurvatureFits(self.strong_lines, self.strong_line_profiles, self.strong_line_2d_profile_fit)

         #We now sum the spectrum along the interpolated lines of curvature, as determined from the strong lines.

         print ('Binning spectrum, along curvature of lines...')
         #parallel_spec_pixels, self.full_pix_interp = self.measureFullPixelSpectrumOld(self.current_image, self.spec_range, undo_line_curvature_dict)
         parallel_spec_pixels, self.full_pix_interp = self.measureFullPixelSpectrum(self.current_image, self.spec_range, self.strong_line_2d_profile_fit)
         #We also do the same thing on a region above or below to determine the background statistics for our binned spectrum
         if self.background_low:
             stat_region = [max(0, self.spec_range[0] - self.background_buffer - (self.spec_range[1] - self.spec_range[0]) ), self.spec_range[0] - self.background_buffer ]
         else:
             stat_region = [self.spec_range[1] + self.background_buffer,
                            min(np.shape(self.current_image)[self.spec_axis], search_range[1] + self.background_buffer + (self.spec_range[1] - self.spec_range[0])) ]
         #print ('stat_region = ' + str(stat_region))
         print ('Binning background in the same way as the spectrum...')
         background_inserted_spectrum = np.copy(self.current_image)
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

         can.saveDataToFitsFile(background_inserted_spectrum, 'ArrayFromWhichBackgroundStatsAreComputed.fits', self.target_dir)
         self.stat_slice_pixels, self.stat_slice_interp = self.measureFullPixelSpectrum(background_inserted_spectrum, self.spec_range, self.strong_line_2d_profile_fit)

         full_pix_spectrum = self.full_pix_interp(parallel_spec_pixels)
         #full_pix_background = self.background_pix_interp(self.stat_slice_pixels)
         full_pix_background_stats = self.stat_slice_interp(self.stat_slice_pixels)

         return parallel_spec_pixels, full_pix_spectrum, full_pix_background_stats

    def initializeScatterKey(self, scatter_data_dir = None, scatter_data_key_file = None):
        """
        This function sets up the mapper between monochromatic wavelengths
           at which the scatter was measured and the corresponding .fits
           files containing the measured scatter pattern.  Ths result is a
           dictionary going between wavelength and fits file name (the data
           arrays themselves are not yet read in, as they would occupy too
           much RAM).
        """
        if scatter_data_dir == None:
            scatter_data_dir = self.scatter_data_dir
        if scatter_data_key_file == None:
            scatter_data_key_file = self.scatter_data_key_file
        scatter_data_cols = can.readInColumnsToList(scatter_data_key_file, file_dir = scatter_data_dir, n_ignore = 1, delimiter = ',', verbose = 0)
        scatter_data_wavelengths = [int(wave) for wave in scatter_data_cols[0]]
        scatter_data_files = [f for f in scatter_data_cols[1]]
        scatter_data_wavelengths , scatter_data_files = can.safeSortOneListByAnother(scatter_data_wavelengths, [scatter_data_wavelengths, scatter_data_files])
        self.scatter_data_mapping = [scatter_data_wavelengths , scatter_data_files]
        return 1

    def determineScatterFunctByWavelength(self, wavelength):
        """
        Determines the 2d scatter pattern at the specified wavelength. This
            is done by interpolating between pixel values for two scatter
            .fits files at the closest sampled longer and shorter wavelengths.
            The interpolating is done linearly.
        The result is a 2d spectrum, describing the scatter pattern for a
            single ADU at the specified wavelength.
        """
        closest_wavelength_index = np.argmin(np.abs(wavelength - np.array(self.scatter_data_mapping[0])))
        if closest_wavelength_index == 0 or closest_wavelength_index == len(self.scatter_data_mapping[0]) - 1:
            #print ('Wavelength ' + str(wavelength) + ' beyond mapping.  Will not correct this scatter. ')
            closest_wavelength_indeces = [closest_wavelength_index, closest_wavelength_index]
        elif wavelength > self.scatter_data_mapping[0][closest_wavelength_index]:
            closest_wavelength_indeces = [closest_wavelength_index, closest_wavelength_index + 1]
        else:
            closest_wavelength_indeces = [closest_wavelength_index - 1, closest_wavelength_index]
        closest_wavelengths = [self.scatter_data_mapping[0][index] for index in closest_wavelength_indeces]
        interp_images = [can.readInDataFromFitsFile(self.scatter_data_mapping[1][index], self.scatter_data_dir)[0] for index in closest_wavelength_indeces]
        print ('For wavelength ' + str(wavelength) + ', interpolating between images ' + str([self.scatter_data_mapping[1][closest_wavelength_indeces[0]], self.scatter_data_mapping[1][closest_wavelength_indeces[1]]]) )
        #If both indeces are the same, we're trying to interpolate beyond the sampled range.  So return 0s (no scatter correction).
        if closest_wavelength_indeces[0] == closest_wavelength_indeces[1]:
            scatter_image = interp_images[0] * 0.0
        else:
            scatter_image_slopes = (interp_images[1] - interp_images[0]) / (closest_wavelengths[1] - closest_wavelengths[0])
            scatter_image = interp_images[0] + (wavelength - closest_wavelengths[0]) * scatter_image_slopes
        return scatter_image

    def determineScatterFromScatterImages(self, parallel_spec_pixels, parallel_spec_wavelengths, full_pix_spectrum, full_pix_background_stats, scatter_image_file_name = None):
        """
        For a 1d spectrum given by parallel_spec_pixels (pixels),
            parallel_spec_wavelengths (wavelengths), full_pix_spectrum (spectral
            values), determines the 2d image of scattered light, inferred from
            the scatter image mapping.
        This is done by identifying which sections of the spectrum are
            sufficiently bright (exceeding the background by some specified
            standard deviation amount), interpolating the scatter patterns
            for those wavelengths, scaling them by the number of ADU at that
            wavelength, and adding those together to form a scatter pattern.
            This 2d map of overall scatter is what we return.
        """
        std_thresh_for_scatter_correct = self.std_thresh_for_scatter_correct
        pix_std = np.std(full_pix_background_stats)
        if scatter_image_file_name == None:
            scatter_image_file_name = self.scatter_image_file_name
        self.initializeScatterKey()
        bright_indeces = [i for i in range(len(parallel_spec_wavelengths)) if full_pix_spectrum[i] > std_thresh_for_scatter_correct * pix_std + full_pix_background_stats[i] ]

        plt.plot(parallel_spec_pixels, full_pix_spectrum, c = 'g')
        plt.plot(parallel_spec_pixels, full_pix_background_stats, c = 'k')
        plt.plot(parallel_spec_pixels, std_thresh_for_scatter_correct * pix_std + full_pix_background_stats, c = 'r')
        plt.scatter(bright_indeces, [ full_pix_spectrum[i] for i in bright_indeces], c = 'orange', marker = 'x')
        plt.show()
        scatter_image = self.determineScatterFunctByWavelength(parallel_spec_wavelengths[0]) * full_pix_spectrum[bright_indeces[0]]
        #scatter_image = self.determineScatterFunctByWavelength(self.wavelength_of_mu_solution(lines_to_scatter[0][1])) * lines_to_scatter[0][0] * np.sqrt(2.0 * np.pi) * lines_to_scatter[0][2]
        f, axarr = plt.subplots(1,2, figsize = (12, 6) )
        for i in bright_indeces[1:]:
            start = time.time()
            #sys.stdout.write('Determining scatter solution... We are ' + str(int (i / len(parallel_spec_wavelengths) * 100)) + '% done)...' )
            #sys.stdout.flush()
            #line_pixel, total_line_flux =  [lines_to_scatter[i][1], lines_to_scatter[i][0] * np.sqrt(2.0 * np.pi) * lines_to_scatter[i][2]]
            line_wavelength, wavelength_flux = [parallel_spec_wavelengths[i], full_pix_spectrum[i]]
            #line_wavelength = self.wavelength_of_mu_solution(line_pixel)
            new_scatter_image = self.determineScatterFunctByWavelength(line_wavelength) * wavelength_flux
            print ('[i, line_wavelength, self.mu_of_wavelength_solution(line_wavelength), wavelength_flux] = ' + str([i, line_wavelength, self.mu_of_wavelength_solution(line_wavelength), wavelength_flux]))
            scatter_image = scatter_image + new_scatter_image
            axarr[0].imshow(scatter_image)
            axarr[1].imshow(new_scatter_image)
            end = time.time()
        print ("")
        #plt.show()
        #plt.imshow(scatter_image)
        #plt.show()
        can.saveDataToFitsFile(scatter_image, scatter_image_file_name, self.scatter_data_dir, header = 'default')
        return scatter_image_file_name

    def reduceImagesTo1dSpectrum(self, images_to_reduce, n_std_for_strong_line, save_intermediate_images = 0, define_new_anchor_pix = 0, save_image_name = None, redetermine_spec_range = 1, crc_correct = None, reference_image = 0, bin_along_curve = 1, determine_new_line_curvature = 0, normalize = 0, apply_background_correction = 0, apply_scatter_correction = 0, spec_in_wavelengths = 1, save_perp_spec_to_file = 0, perp_spec_save_file = None):
        """
        This is the primary called function if the user has a list of
            signals images that they want to produce to a (single,
            stacked) 1d spectrum.  It performs the intermediate analysis
            steps (in the appropriate order).
        To measure how sky lines change over time, this function should
            be called on the sky images INDIVIDUALLY (each image being
            given as its own list of text files of length 1).  This is
            because all given images are averaged together, post analysis.
        """
        #Makes self.current_images point to an array of the coadded, reduced images_to_reduce
        self.image_roots = [spec_file[0:-len(self.data_image_suffix)] for spec_file in images_to_reduce]
        self.perp_spec_image_name = self.image_roots[0] + self.perp_spec_image_suffix + self.figure_suffix
        self.processed_file = self.image_roots[0] + self.processed_file_suffix + self.data_image_suffix
        self.processed_spectrum = self.image_roots[0] + self.processed_spectra_image_suffix + self.figure_suffix
        current_image, current_header = self.processImages(spec_files_to_reduce = images_to_reduce, save_image_name = save_image_name, crc_correct = crc_correct, apply_background_correction = apply_background_correction, scatter_correction_file = None, redetermine_spec_range = redetermine_spec_range)
        parallel_spec_pixels, full_pix_spectrum, full_pix_background_stats = self.integrateImageAlongCurvature(current_image, current_header, n_std_for_strong_line, save_intermediate_images = save_intermediate_images, define_new_anchor_pix = define_new_anchor_pix, redetermine_spec_range = 0, reference_image = reference_image, determine_new_line_curvature = determine_new_line_curvature )
        if spec_in_wavelengths:
            parallel_spec_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in parallel_spec_pixels]
            if apply_scatter_correction:
                scatter_image_file_name = self.determineScatterFromScatterImages(parallel_spec_pixels, parallel_spec_wavelengths, full_pix_spectrum, full_pix_background_stats)
                print ('scatter_image_file_name = ' + str(scatter_image_file_name))
                current_image, current_header = self.processImages(spec_files_to_reduce = images_to_reduce, save_image_name = save_image_name, crc_correct = crc_correct, apply_background_correction = apply_background_correction, scatter_correction_file = scatter_image_file_name, redetermine_spec_range = redetermine_spec_range)
                parallel_spec_pixels, full_pix_spectrum, full_pix_background_stats = self.integrateImageAlongCurvature(current_image, current_header, n_std_for_strong_line, save_intermediate_images = save_intermediate_images, define_new_anchor_pix = define_new_anchor_pix, redetermine_spec_range = 0, reference_image = reference_image, determine_new_line_curvature = determine_new_line_curvature)

        #If the throughput is 0, replace with infinity so that the normalized spectrum value is 0.
        if spec_in_wavelengths:
            throughput = self.throughput_interp(parallel_spec_wavelengths)
            nm_subtended_by_pixels = [self.wavelength_of_mu_solution(pix + 0.5) - self.wavelength_of_mu_solution(pix - 0.5) for pix in parallel_spec_pixels]
            #throughput[throughput == 0] = np.inf

            if self.show_fits:
                max_full = np.max(full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
                max_ref = max(full_pix_background_stats[self.spec_range[0]:self.spec_range[1]])
                spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum / max_full , c = 'g')[0]
                background = plt.plot(parallel_spec_wavelengths, full_pix_background_stats / max_full, c = 'r')[0]

            #I want the sky brightness in Ry/nm, and so I must divide out the wavelength range
            #  our pixels subtend.
            full_pix_spectrum = full_pix_spectrum / throughput / nm_subtended_by_pixels
            full_pix_background_stats = full_pix_background_stats / throughput / nm_subtended_by_pixels
            max_full = np.max(full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
            max_ref = max(full_pix_background_stats[self.spec_range[0]:self.spec_range[1]])
            if self.show_fits:
                throughput_plot = plt.plot(parallel_spec_wavelengths, (1.0 / self.throughput_interp(parallel_spec_wavelengths)) * np.max(self.throughput_interp(parallel_spec_wavelengths)), c = 'k')[0]
                norm_spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum / max_full , c = 'blue', )[0]
                norm_background = plt.plot(parallel_spec_wavelengths, full_pix_background_stats / max_full, c = 'orange')[0]
                full_pix_background_stats = full_pix_background_stats * self.throughput_interp(parallel_spec_wavelengths)
                plt.title('Re-measured spectrum, binned according to strong line fits')
                plt.xlabel('Start of bin line (pix)')
                plt.ylabel('Normalized y')
                plt.ylim(-0.05, 1.05)
                plt.legend([spec, background, throughput_plot, norm_spec, norm_background], ['Spec. - bg', 'bg', 'Throughput (Ry / ADU)', 'Throughput corr. Spec. - bg', 'Throughput corr. bg'])
                plt.show()
            if normalize:
                full_pix_spectrum = full_pix_spectrum / max_full
                full_pix_background_stats  = full_pix_background_stats / max_full
        else:
            full_pix_spectrum = np.sum(self.current_image[self.spec_range[0]:self.spec_range[1], :], axis = 0).tolist()
            parallel_spec_pixels = list(range(len(full_pix_spectrum)))
            print ('[binned_spec, parallel_spec_pixels] = ' + str([full_pix_spectrum, parallel_spec_pixels]))
            print ('[len(binned_spec), len(parallel_spec_pixels)] = ' + str([len(full_pix_spectrum), len(parallel_spec_pixels)]))
            #full_pix_background_stats = []
        if save_perp_spec_to_file:
            if perp_spec_save_file == None:
                 perp_spec_save_file = self.perp_spec_save_file
            with open(self.target_dir + perp_spec_save_file, 'w') as f:
                 f.write(str(self.spec_range[0]) + ' ' + str(self.spec_range[1]))

        if np.any([elem == np.nan or elem == np.inf for elem in full_pix_spectrum]):
            print ('Bad something!!!! ')
            print ('parallel_spec_pixels = ' + str(parallel_spec_pixels))
            print ('full_pix_spectrum = ' + str(full_pix_spectrum))
            print ('full_pix_background_stats = ' + str(full_pix_background_stats))
            print ('throughput = ' + str(throughput))
        return parallel_spec_pixels, full_pix_spectrum , full_pix_background_stats, self.current_header

    def measureSystemThroughtput(self, throughput_files_list, target_dir = None, show_fits = 1):
        """
        From a set of measurement files used to measure the throughput (ideally
            a stack of spectra of a known light source producing a uniform background
            illumination pattern), this function measures the throughput of OSELOTS.
        This function does not need to be run every time the instrument is used -
            only when the throughput needs to be remeasured.
        """
        if target_dir == None:
            target_dir = self.target_dir
        if show_fits != None:
            self.show_fits = show_fits
        if self.wavelength_of_mu_solution == None:
            self.getWavelengthSolution()
        throughput_files =  can.readInColumnsToList(throughput_files_list, file_dir = target_dir, n_ignore = 0,verbose = 0)[0]
        self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(throughput_files, self.n_std_for_strong_ref_line, define_new_anchor_pix = 1, save_intermediate_images = 1, crc_correct = 0, reference_image = 0, bin_along_curve = 0)
        parallel_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_ref_spec_pixels]
        max_throughput_ADU = np.max(self.full_ref_pix_spectrum)
        throughput_source = can.readInColumnsToList(self.ref_throughput_file, file_dir = self.archival_data_dir, n_ignore = 1, delimiter = ',', verbose = 0)
        ref_throughput_interp = scipy.interpolate.interp1d([float(elem) for elem in throughput_source[0]], np.array([float(elem) for elem in throughput_source[1]]) * np.array([float(elem) for elem in throughput_source[0]]) * self.energy_to_photon_scaling, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
        interped_throughput_ref = ref_throughput_interp(parallel_wavelengths).tolist()
        throughput = np.array(self.full_ref_pix_spectrum) * np.array(interped_throughput_ref)
        max_throughput = np.max(throughput)
        rel_throughput = throughput * max_throughput
        max_interped_throughput_ref = max(interped_throughput_ref )
        if show_fits:
            fig = plt.figure(figsize=(9, 3))
            measured = plt.plot(parallel_wavelengths, self.full_ref_pix_spectrum * max_throughput_ADU, c = 'g')[0]
            ref = plt.plot(parallel_wavelengths, np.array(interped_throughput_ref) * max_interped_throughput_ref, c = 'r')[0]
            throughput = plt.plot(parallel_wavelengths, rel_throughput, c = 'b')[0 ]
            plt.xlabel('Wavelength (nm)', fontsize= 14)
            plt.ylabel('Normalized intensity', fontsize= 14)
            plt.legend([measured, ref, throughput], ['DH2000 on OSELOTS', 'DH2000 Reported Intensity', 'Inferred Throughput'], fontsize = 10)
            plt.tight_layout()
            plt.show()
        new_thruoghput = can.saveListsToColumns([parallel_wavelengths, rel_throughput], self.throughput_file, self.archival_data_dir, sep = ',', header = 'Wavelengths(nm), Throughput')

        return 1


    def computeReferenceSpectrum(self, ref_spec_images_list = None, save_new_reference_spectrum = 1, data_dir = None, spec_save_dir = None, ref_spec = None, show = 0 ):
        """
        Computes the reference spectrum from a list of spectral reference images
            (usually an "arclamp", like the KR1), and saves the wavelength solution
            as callable functions (going wavelength to pixel and pixel to wavelength).
        """
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

        ref_spec_image_files = can.readInColumnsToList(ref_spec_images_list, file_dir = data_dir, n_ignore = 0, verbose = 0)[0]
        print ('ref_spec_image_files = ' + str(ref_spec_image_files))
        self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(ref_spec_image_files, self.n_std_for_strong_ref_line, define_new_anchor_pix = 1, save_intermediate_images = 1, crc_correct = 0, reference_image = 1, apply_background_correction = 0, spec_in_wavelengths = 0, save_perp_spec_to_file = save_new_reference_spectrum)
        if len(self.full_ref_pix_spectrum) < 1:
            print ('Failed to identify lines in reference image list ' + str(ref_spec_images_list))
            false_wavelength_solution = np.zeros(self.wavelength_solution_order)
            false_wavelength_solution[-1] = self.approx_pix_scale
            mu_of_wavelength_solution, wavelength_of_mu_solution = self.createWavelengthSolutionCallableFunctions(false_wavelength_solution)
            return [mu_of_wavelength_solution, wavelength_of_mu_solution]
        #print ('[full_pix_spectrum, full_pix_background_stats] = ' + str([full_pix_spectrum, full_pix_background_stats]))
        print ('self.full_ref_pix_background_stats = ' + str(self.full_ref_pix_background_stats))

        if show:
            f, axarr = plt.subplots(2,1)
            axarr[0].plot(self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, c = 'k')
            axarr[1].plot(self.parallel_ref_spec_pixels, self.full_ref_pix_background_stats, c = 'k')
            plt.show()
            plt.plot(self.full_ref_pix_background_stats)
            plt.show()
        self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings = self.detectLinesCentersInOneD(self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum, self.full_ref_pix_background_stats, spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_ref_line, show = self.show_fits, background_bin_width = self.pix_bin_to_meas_background_noise)
        print ('[self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings] = ' + str([self.full_ref_absorbtion_crossings, self.full_ref_emission_crossings]))
        self.full_ref_emission_crossings = [crossing for crossing in self.full_ref_emission_crossings if (crossing > 0 and crossing < self.parallel_ref_spec_pixels[-1])]

        self.full_ref_emission_fits = self.identifyLinesOnSlice(self.parallel_ref_spec_pixels, self.full_ref_pix_spectrum,
                                 max_line_fit_width = self.width_pix_sample_to_fit_ref_line, peak_guesses = self.full_ref_emission_crossings, show_spec = self.show_fits, verbose = 0,
                                 fit_lines_with_pedestal = 1, fit_line_width_with_seeing_funct = 0,  background_bin_width = self.background_bin_width_for_line_id_in_slice)

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

    def getWavelengthSolution(self, ref_spec_images_list = None, ref_spec_solution_file = None, save_new_reference_spectrum = 0, ref_spec = None, show_fits = None):
        """
        Initializes the wavelength solution for the spectral processor,
            either reading in a saved spectral solution or determining
            the spectral solution from a list of spectral images.
        """
        if show_fits != None:
            self.show_fits = show_fits
        if ref_spec_solution_file != None:
            self.ref_spec_solution_file = ref_spec_solution_file
        if save_new_reference_spectrum or not(os.path.isfile(self.target_dir + self.ref_spec_solution_file)):
            if ref_spec_images_list != None:
                self.ref_spec_images_list = ref_spec_images_list
            else:
                self.ref_spec_images_list = can.readInColumnsToList(self.target_dir + self.spec_images_file)
            self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.computeReferenceSpectrum(save_new_reference_spectrum = save_new_reference_spectrum, ref_spec = ref_spec, show = show_fits)
            if save_new_reference_spectrum:
                if ref_spec_solution_file == None:
                    self.ref_spec_solution_file = ref_spec_solution_file
        else:
            print ('Loading archival wavelength solution file')
            self.mu_of_wavelength_solution, self.wavelength_of_mu_solution = self.loadWavelengthSolution(solution_file = self.ref_spec_solution_file, load_dir = self.archival_data_dir)

        return 1

    def subtractContinuum(self, pixels, spec_of_pixels, used_continuum_seeds = None, continuum_smoothing = None, n_continuum_fit_seeds = None, min_line_vs_seed_sep = None, show_fits = None):
        """
        Subtract off the spline-fitted continuum of the 1d spectrum from the 1d spectrum. The
            continuum is determined by splinning between the spectrum at a pre-selected set of
            wavelengths (either given as the used_continuum_seeds parameter or read in from
            the continuum_seeds_file class variable).
        Subtracting the continuum is necessary because we are interested in measuring the
            intensities of the sky emission lines.
        """
        if used_continuum_seeds == None:
            used_continuum_seeds = can.readInColumnsToList(self.continuum_seeds_file, self.archival_data_dir, n_ignore = self.n_ignore_for_continuum_seeds, convert_to_float = 1, verbose = 0) [0]
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
        initial_continuum_seed_indeces = np.linspace(0, len(pixels)-1, n_continuum_fit_seeds + 1)[1:-1]
        initial_continuum_seed_indeces = [int(index) for index in initial_continuum_seed_indeces]
        #print('[np.min([abs(pixels[index] - line_center) for line_center in line_centers]) for index in initial_continuum_seed_indeces ] = ' + str([np.min([abs(pixels[index] - line_center) for line_center in line_centers]) for index in initial_continuum_seed_indeces ] ))
        #used_continuum_seed_indeces = [index for index in initial_continuum_seed_indeces if np.min([abs(pixels[index] - line_center) for line_center in line_centers]) >= min_line_vs_seed_sep ]
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
        """
        Used the measured throughput of OSELOTS to convert the measured spectrum
            in ADUs vs wavelength to a true sky measurement of Rayleighs vs
            wavelength. If the system throughput has not already been read in,
            this function will first initialize it (using the importSystemThroughput
            function).
        """
        print ('self.throughput_interp = ' + str(self.throughput_interp))
        if throughput_file != None or throughput_data != None or self.throughput_interp == None:
            self.throughput_interp = self.importSystemThroughput(throughput_file = throughput_file, throughput_data = throughput_data, throughput_file_delimiter = throughput_file_delimiter, n_lines_to_ignore = n_lines_to_ignore, throughput_file_dir = throughput_file_dir)
        throughput = np.array([self.throughput_interp(wave)[0] for wave in spec_wavelengths])
        corrected_spec = spec_to_correct * throughput
        print('[throughput, spec_to_correct, corrected_spec] = ' + str([throughput, spec_to_correct, corrected_spec]))
        return corrected_spec


    def measureStrengthOfLinesInImage(self, image_to_measure, show_fits = None, line_dict_id = None, redetermine_spec_range = 0):
        """
        For a single, unprocessed image file, this measures the strength of the
           emission lines in that image.  It stitches together all of the
           analysis, from reducing the 2d image to a 1d spectrum to subtracting
           the continuum and identifying lines.
        """
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

        pix_target_range = [max(int(self.mu_of_wavelength_solution(self.wavelength_target_range[0])), 0), min(int(self.mu_of_wavelength_solution(self.wavelength_target_range[1])), self.max_dimension)]

        full_absorbtion_crossings, full_emission_crossings = self.detectLinesCentersInOneD(parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_line, show = self.show_fits, background_bin_width = self.pix_bin_to_meas_background_noise)

        init_full_emission_fits = self.identifyLinesOnSlice(range(len(full_pix_spectrum )), full_pix_spectrum,
                                 max_line_fit_width = self.width_pix_sample_to_fit_line, peak_guesses = self.identified_full_emission_line_centers, show_spec = show_fits, verbose = 0,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1,
                                 background_bin_width = self.background_bin_width_for_line_id_in_slice)

        full_emission_centroids = [fit[1] for fit in init_full_emission_fits]
        seeing_fit_params = self.simulFitLineWidths(np.array(range(len(full_pix_spectrum ))), full_pix_spectrum, init_full_emission_fits)
        #print('seeing_fit_params = ' + str(seeing_fit_params ))
        seeing_fit_funct = np.poly1d(seeing_fit_params)
        #print ('seeing_fit_funct = ' + str(seeing_fit_funct))
        full_emission_fits = self.identifyLinesOnSlice(range(len(full_pix_spectrum )), full_pix_spectrum,
                                 max_line_fit_width = self.refined_width_pix_sample_to_fit_line,  peak_guesses = self.identified_full_emission_line_centers, show_spec = self.show_fits, verbose = 0,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 0, seeing_fit_funct = seeing_fit_funct,
                                 background_bin_width = self.background_bin_width_for_line_id_in_slice )
        full_emission_fits = [fit for fit in full_emission_fits if fit[1] > parallel_spec_pixels[1] and fit[1] < parallel_spec_pixels[-2] ]
        #print ('full_emission_fits = ' + str(full_emission_fits))
        self.identified_lines_dict[line_dict_id] = { }
        self.identified_lines_dict[line_dict_id][self.lines_in_dict_keyword] = {i:full_emission_fits[i] for i in range(len(self.full_emission_fits))}
        self.identified_lines_dict[line_dict_id][self.obs_time_keyword] = image_header[self.obs_time_keyword]
        identified_full_emission_line_centers = [fit[1] for fit in full_emission_fits]
        identified_full_emission_line_heights = [fit[0] for fit in full_emission_fits]
        identified_full_emission_line_floors = [fit[3] for fit in full_emission_fits]

        if self.show_fits or self.save_final_plot:
            self.plotFullAnalysisStepImage(parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, full_pix_background_stats, continuum_interp, full_pix_spectrum,
                                   identified_full_emission_line_centers, identified_full_emission_line_heights, identified_full_emission_line_floors, continuum_fit_points, self.wavelength_of_mu_solution,
                                   self.show_fits, self.save_final_plot, plot_title = 'Spectrum of image: ' + str(line_dict_id),
                                   save_image_name = image_to_measure[:-len(self.data_image_suffix)] + self.processed_multistep_spectra_image_suffix + self.figure_suffix)
            self.plotFullLineImage(parallel_spec_wavelengths, full_pix_spectrum, full_pix_background_stats,
                                   identified_full_emission_line_centers, identified_full_emission_line_heights, identified_full_emission_line_floors, self.wavelength_of_mu_solution,
                                   self.show_fits, self.save_final_plot, plot_title = 'Spectrum of image: ' + str(line_dict_id), noise_bin_width = self.pix_bin_to_meas_background_noise,
                                   save_image_name = image_to_measure[:-len(self.data_image_suffix)] +  self.processed_spectra_image_suffix + self.figure_suffix)
        if self.save_spectra:
            self.saveFullSpectrum(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]],
                                  data_save_name = image_to_measure[:-len(self.data_image_suffix)] +  self.processed_spectra_image_suffix + self.text_suffix)
            self.saveFullSpectrumWithSteps(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.no_background_sub_full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], self.continuum_interp, self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                           data_save_name = image_to_measure[:-len(self.data_image_suffix)] + self.processed_multistep_spectra_image_suffix  + self.text_suffix)
        return 1


    def plotLineProfilesInTime(self, n_subplots = 8, n_cols = 2, figsize = (8,16), line_variation_image_name = 'skyLineChangesOverTime.pdf', n_legend_col = 2, legend_text_size = 8, n_ticks = 11, xlabel = r'$\Delta t$ since first exp. (min)', ylabel ='Fitted volume of line (Ry)' , y_lims = [-0.05, 1.05], line_legend_n_sig_figs = 4):
        """
        This function plots the intensities of identified lines in time.  This
           function should be run AFTER all images have been reduced, as it
           relies on the existing library of line profiles that have already been
           generated.
        The resulting plot shows all line heights, on the same intensity scales,
           as they vary throughout the night.
        This function and the plotScaledLineProfilesInTime function show the same
           data, but in different ways.
        """
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

        max_height = 0.0
        max_volume = 0.0
        for i in range(len(line_numbers)):
             plot_number = [j for j in range(len(n_lines_per_plot)) if i < line_numbers_by_plot[1:][j]] [0]
             line_number = line_numbers[i]
             line_heights = [self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][0] for key in self.identified_lines_dict.keys()]
             line_sigmas = [self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][2] for key in self.identified_lines_dict.keys()]
             line_volumes = np.sqrt(2.0 * np.pi * line_sigmas ** 2.0) * line_heights
             max_height = np.max(line_heights + [max_height])
             max_volume = np.max(line_volumes + [max_volume])
             #y_lims = [min([y_lims[0]] + line_heights), max([y_lims[1]] + line_heights) ]
             image_numbers = [key for key in self.identified_lines_dict.keys()]
             #single_lines[plot_number] = single_lines[plot_number] + [axarr[plot_number // 2, plot_number % 2].plot(image_numbers[1:], line_heights[1:])[0] ]
             single_lines[plot_number] = single_lines[plot_number] + [axarr[plot_number // 2, plot_number % 2].plot(delta_ts[1:], line_volumes[1:])[0] ]

        for i in range(n_subplots):
             print ('Making subplot ' + str(i + 1) + ' of ' + str(n_subplots))
             line_identifiers = ['L' + str(line_number) + ': ' + str(can.round_to_n(self.wavelength_of_mu_solution(self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword][line_number][1]), line_legend_n_sig_figs )) + 'nm' for line_number in line_numbers[line_numbers_by_plot[i]:line_numbers_by_plot[i+1]] ]
             axarr[ i // 2, i % 2].legend(single_lines[i], line_identifiers, ncol = n_legend_col, prop={'size':legend_text_size} )
             axarr[ i // 2, i % 2].set_ylim([-0.01 * max_volume, max_volume ])
             if i // 2 == n_subplots // 2 -1 :
                  axarr[ i // 2, i % 2].set_xlabel(xlabel )
             if i % 2 == 0:
                  axarr[ i // 2, i % 2].set_ylabel(ylabel )
             axarr[ i // 2, i % 2 ].set_xticks
        print ('Now saving plot... ')
        plt.savefig(self.target_dir + line_variation_image_name )
        plt.close('all')
        return 1


    def plotScaledLineProfilesInTime(self, n_cols = 4, fig_v_elem = 0.75, fig_h_elem = 2,
                                     line_variation_image_name = 'skyLineChangesOverTimeScaled.pdf', line_variation_catalogue_name = 'skyLineChangesOverTime.lcat', cat_delimiter = ', ',
                                     line_id_text_size = 8, xlabel = r'$\Delta t$ since first exp. (min)', ylabel ='Fitted volumes of line (Ry)' ,
                                     line_box_buffer = 0.1, text_buffer_left = 0.03, text_top_buffer = 0.03, n_time_ticks = 5, n_std_to_show = 3,
                                      cmap_buffer = 0.03, cmap_bottom = 0.05, cmap_width = 0.02, cmap_height = 0.9, cbar_label = 'Line height above continuum',
                                      y_lims = [-0.05, 1.05], line_text_n_sig_figs = 4, line_legend_n_sig_figs = 4, labelsize = 10, title_size = 14, col_sep = 0.0,
                                      plot_elem_to_cbar_width_ratio = 1, cmap_str = 'viridis', cmap_title = 'Height of line over continuum (ADU)', n_std_color_lims = 2.0, color_bound_n_sig_rounding = 3):
        """
        This function plots the normalized intensities of identified lines in
           time.  This function should be run AFTER all images have been
           reduced, as it relies on the existing library of line profiles that
           have already been generated.
        The resulting plot consists of a single of cells, one for each
           identified line.  Each cell is normalized to show the entire line
           (within 5\sigma of its own variations), and therefore the cells'
           vertical scales should not be read in reference to each other.
           The color scale indicates the lines' true height.
        This function and the plotLineProfilesInTime function show the same
           data, but in different ways.
        """
        cmap = cm.get_cmap(cmap_str, 12)
        single_lines = [[] for i in range(n_cols)]
        spectra_ids = [key for key in list(self.identified_lines_dict.keys())]
        print ('Here are the image IDs of the images that I am going to plot lines for:  ')
        print (spectra_ids)
        ignore_ids = can.getUserInputWithDefault('Should I not plot any of these spectra? (y, Y, yes, Yes, YES, or 1 for "yes"; default 0): ', '0')
        ignore_ids = (ignore_ids in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
        ids_to_ignore = []
        while ignore_ids:
            id_to_ignore = can.getUserInputWithDefault('Which spectrum ID should I ignore next? (Just [RETURN] to finish identifying spectra to ignore): ', 'DONE')
            if id_to_ignore == 'DONE':
                ignore_ids = []
            else:
                if id_to_ignore in spectra_ids:
                    ids_to_ignore = ids_to_ignore + [id_to_ignore]
                elif id_to_ignore.isdigit() and int(id_to_ignore) in spectra_ids:
                    ids_to_ignore = ids_to_ignore + [int(id_to_ignore)]
                else:
                    print ('Your input id of ' + str(id_to_ignore) + ' is not a spectra ID.  Double check capitalization and whitespace? ')
        for id_to_ignore in ids_to_ignore:
            index = spectra_ids.index(id_to_ignore)
            spectra_ids = can.removeListElement(spectra_ids, index)
        if len(ids_to_ignore) > 0:
            print ('Okay, here are the spectra IDs I am now going to plot: ')
            print (spectra_ids)

        line_numbers = [key for key in self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword] ]
        line_ids = ['L' + str(int(line_number) + 1) for line_number in line_numbers ]
        line_wavelengths = [str(can.round_to_n(self.wavelength_of_mu_solution(self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword][line_number][1]), line_legend_n_sig_figs )) + 'nm' for line_number in line_numbers ]
        #line_identifiers = ['L' + str(int(line_number) + 1) + ': ' + str(can.round_to_n(self.wavelength_of_mu_solution(self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword][line_number][1]), line_legend_n_sig_figs )) + 'nm' for line_number in line_numbers ]
        print ('Here are the identified line numbers, and their corresponding wavelengths: ')
        print ([line_ids[i] + ': ' + line_wavelengths[i] for i in range(len(line_ids))])
        ignore_lines = can.getUserInputWithDefault('Should I not plot any of these lines? (y, Y, yes, Yes, YES, or 1 for "yes"; default 0): ', '0')
        ignore_lines = (ignore_lines in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
        lines_to_ignore = []
        while ignore_lines:
            line_to_ignore = can.getUserInputWithDefault('Which line ID should I ignore next (ID starting with "L", NOT wavelength)? (Just [RETURN] to finish identifying lines to ignore): ', 'DONE')
            if line_to_ignore == 'DONE':
                ignore_lines = []
            else:
                if line_to_ignore in line_ids:
                    lines_to_ignore = lines_to_ignore + [line_to_ignore]
                else:
                    print ('Your input ID of ' + str(line_to_ignore) + ' is not a line ID.  Double check capitalization and whitespace? ')
        for line_to_ignore in lines_to_ignore:
            index = line_ids.index(line_to_ignore)
            line_numbers = can.removeListElement(line_numbers, index)
            line_ids = can.removeListElement(line_ids, index)
            line_wavelengths = can.removeListElement(line_wavelengths, index)
        line_identifiers = [line_ids[i] + ': ' + line_wavelengths[i] for i in range(len(line_ids))]

        print ('Okay, here are the lines I am now going to show: ')
        print (line_identifiers)

        #line_numbers = line_numbers[0:20]

        n_lines_per_col = int(np.ceil(len(line_numbers) / n_cols))
        n_lines_per_col = [n_lines_per_col + (1 if (len(line_numbers) % n_lines_per_col - i) > 0 else 0)  for i in range(n_cols)]
        max_n_lines_per_col = max(n_lines_per_col)
        line_numbers_by_col = [0] + [np.sum(n_lines_per_col[0:i]) for i in range(1, n_cols)]  + [len(line_numbers)]
        n_extra_lines = len(line_numbers) % n_cols
        timestamps = [can.convertDateTimeStrToSeconds(self.identified_lines_dict[spectrum_id][self.obs_time_keyword], self.date_format_str) for spectrum_id in spectra_ids]
        min_time = np.min(timestamps)
        delta_ts = [(time - min_time) / 60 for time in timestamps]
        max_delta_t = np.max(delta_ts)

        line_table_to_save = [timestamps]

        f, axarr = plt.subplots(max_n_lines_per_col, n_cols, figsize = (fig_h_elem * n_cols, (fig_v_elem * max_n_lines_per_col + fig_v_elem / plot_elem_to_cbar_width_ratio) * (1 + cmap_buffer * 2) ), squeeze = False )
        #f = plt.figure(constrained_layout=True, figsize = (fig_h_elem * n_cols +  fig_h_elem / plot_elem_to_cbar_width_ratio, fig_v_elem * max_n_lines_per_col))
        #gs = gridspec.GridSpec(ncols=n_cols * plot_elem_to_cbar_width_ratio, nrows=max_n_lines_per_col, figure=f)
        plt.subplots_adjust(wspace=col_sep, hspace=0)
        all_line_heights = [[self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][0] for key in spectra_ids] for line_number in line_numbers]
        all_line_sigmas = [[self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][2] for key in spectra_ids] for line_number in line_numbers]
        all_line_min, all_line_max, all_line_mean, all_line_std = [np.min(all_line_heights), np.max(all_line_heights), np.mean(all_line_heights), np.std(all_line_heights)]
        line_color_min, line_color_max = [max(all_line_mean - n_std_color_lims * all_line_std, 0.0), all_line_mean + n_std_color_lims * all_line_std]
        line_color_scaling_funct = lambda line_heights: (line_heights  - line_color_min) / (line_color_max - line_color_min)
        yticks_set = [[] for i in range(n_cols)]
        yticklabels_set = [[] for i in range(n_cols)]

        all_axes = [[] for i in range(n_cols)]
        for i in range(n_cols):
            for j in range(max_n_lines_per_col):
                axarr[j, i].set_yticks([])
                axarr[j, i].set_xticks([])
                #if j >= n_lines_per_col[i] and j < max_n_lines_per_col - 1:
                #    print ('For [j, i] = ' + str([j, i] ) + ', we should be removing xticks... ')
                #    axarr[j, i].plot([0,1,2], [0,1,2])

        for i in range(len(line_numbers)):
             plot_number = [j for j in range(len(n_lines_per_col)) if i < line_numbers_by_col[1:][j]] [0]
             col_num = i // max_n_lines_per_col
             artificial_row_num = i % max_n_lines_per_col
             #print ('[plot_elem_to_cbar_width_ratio * col_num, plot_elem_to_cbar_width_ratio * (col_num + 1)] = ' + str([plot_elem_to_cbar_width_ratio * col_num, plot_elem_to_cbar_width_ratio * (col_num + 1)] ))
             ax = axarr[artificial_row_num, col_num]
             #ax = f.add_subplot(gs[artificial_row_num, plot_elem_to_cbar_width_ratio * col_num:plot_elem_to_cbar_width_ratio * (col_num + 1)])
             all_axes[col_num] = all_axes[col_num] + [ax]
             line_number = line_numbers[i]
             line_heights = np.array(all_line_heights[i] )
             line_sigmas = np.array(all_line_sigmas[i])
             line_volumes = np.sqrt(2.0 * np.pi * line_sigmas ** 2.0) * line_heights
             line_table_to_save = line_table_to_save + [line_volumes]
             line_median, line_std = [np.median(line_volumes), np.std(line_volumes)]
             #line_heights = [self.identified_lines_dict[key][self.lines_in_dict_keyword][line_number][0] for key in self.identified_lines_dict.keys()]
             line_min, line_max = [0.0, line_median + n_std_to_show * line_std]
             #line_box_bounds = [artificial_row_num / max_n_lines_per_col, (artificial_row_num + 1) / max_n_lines_per_col - line_box_buffer / max_n_lines_per_col ]
             line_box_bounds = [0.0, (1.0 - line_box_buffer - text_top_buffer)]
             #yticks_set[plot_number] = yticks_set[plot_number] + [(line_box_bounds[1] + line_box_bounds[0]) / 2.0]
             #yticklabels_set[plot_number] = yticklabels_set[plot_number] + [line_identifiers[i]]
             on_plot_text = ax.text((delta_ts[-1] - delta_ts[0]) * text_buffer_left, 1 - text_top_buffer, line_identifiers[i], fontsize = labelsize, verticalalignment = 'top' )
             scaled_line_volumes = np.array([line_box_bounds[0] + (line_box_bounds[1] - line_box_bounds[0]) * (height - line_min) / (line_max - line_min) for height in line_volumes ])
             line_segment_middles = np.array([(line_volumes[i] + line_volumes[i+1]) / 2.0 for i in range(1, len(line_volumes) - 1)])
             line_color_coefs = line_color_scaling_funct(line_segment_middles)
             xs, ys = delta_ts[1:], scaled_line_volumes[1:]
             points = np.array([xs, ys]).T.reshape(-1, 1, 2)
             segments = np.concatenate([points[:-1], points[1:]], axis=1)
             colors = cmap(line_color_coefs)
             colors = [tuple(color_elem) for color_elem in colors]

             lc = LineCollection(segments, colors = colors)
             lc.set_linewidth(0.5)
             #y_lims = [min([y_lims[0]] + line_heights), max([y_lims[1]] + line_heights) ]
             image_numbers = [key for key in self.identified_lines_dict.keys()]
             #single_lines[plot_number] = single_lines[plot_number] + [axarr[plot_number // 2, plot_number % 2].plot(image_numbers[1:], line_heights[1:])[0] ]
             #single_lines[plot_number] = single_lines[plot_number] + [ax.plot(delta_ts[1:], scaled_line_heights[1:], c=cm.hot(line_heights), edgecolor='none')[0] ]
             #single_lines[plot_number] = single_lines[plot_number] + [ax.plot(delta_ts[1:], scaled_line_heights[1:], c=cm.hot(line_heights) )[0] ]
             #single_lines[plot_number] = single_lines[plot_number] + [ax.scatter(delta_ts[1:], scaled_line_heights[1:], c=colors, marker = '.' ) ]
             single_lines[plot_number] = single_lines[plot_number] + [ax.add_collection(lc) ]
             #ax.set_yticks([])
             ax.set_xlim(0.0, max_delta_t)
             ax.set_xticks([])
             ax.set_ylim([0.0, 1.0])
        for i in range(n_cols):
            col_num = i
            ax = axarr[-1, col_num]
            ax.set_xlabel(r'$\Delta t$ (min)', fontsize = labelsize )
            xticks = [int(elem) for elem in np.linspace(0.0, max_delta_t, n_time_ticks + 1)][1:-1]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize = labelsize)
            #ax.set_yticks([])
            #yticks = yticks_set[i]
            #yticklabels = yticklabels_set[i]
            #ax.set_yticks(yticks)
            #ax.set_yticklabels(yticklabels, fontsize = labelsize, rotation = 90, verticalalignment = 'center')
        #dummy_scat = ax.scatter([0.0, max_delta_t], [0.0, 1.0], color = cmap([0.0, 1.0]), alpha = 1.0 )

        #cbar_ax = gs[0, -1]
        #ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
        #plt.tight_layout()
        f.subplots_adjust(top = 1 - 1 / (max_n_lines_per_col * plot_elem_to_cbar_width_ratio + 1) - cmap_buffer )
        cbar_ax = f.add_axes([0.15, 1 - 1 / (max_n_lines_per_col * plot_elem_to_cbar_width_ratio + 1) + cmap_buffer, 0.7, 0.03 ])
        cbar = f.colorbar(cm.ScalarMappable(cmap=cmap), cax = cbar_ax, shrink=0.95, aspect = 40, orientation = 'horizontal')
        cbar.set_ticks(np.arange(0, 1.1, 0.5))
        cbar.set_label(cbar_label, fontsize = title_size)
        print ('[line_color_min, line_color_max ] = ' + str([line_color_min, line_color_max ] ))
        cbar.ax.tick_params(labelsize=labelsize)
        cbar.set_ticklabels([r'$>$' + str(can.round_to_n(line_color_min, color_bound_n_sig_rounding )) + r' Ry nm$^{-1}$', str(can.round_to_n((line_color_max + line_color_min) / 2.0, color_bound_n_sig_rounding )) + r' Ry nm$^{-1}$' , r'$<$' + str(can.round_to_n(line_color_max, color_bound_n_sig_rounding )) + r' Ry nm$^{-1}$'] )
        print ('Saving plot of line intensities in time ... ')
        #plt.tight_layout()
        plt.savefig(self.target_dir + line_variation_image_name )
        plt.close('all')
        #plt.show()

        header = cat_delimiter.join(['Obs time (s)'] + line_identifiers)
        can.saveListsToColumns(line_table_to_save, line_variation_catalogue_name, self.target_dir, header = header, sep = cat_delimiter)
        """

        for i in range(n_subplots):
             print ('Making subplot ' + str(i + 1) + ' of ' + str(n_subplots))
             line_identifiers = ['L' + str(line_number) + ': ' + str(can.round_to_n(self.wavelength_of_mu_solution(self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword][line_number][1]), line_legend_n_sig_figs )) + 'nm' for line_number in line_numbers[line_numbers_by_plot[i]:line_numbers_by_plot[i+1]] ]
             axarr[ i // 2, i % 2].legend(single_lines[i], line_identifiers, ncol = n_legend_col, prop={'size':legend_text_size} )
             axarr[ i // 2, i % 2].set_ylim([*y_lims])
             if i // 2 == n_subplots // 2 -1 :
                  axarr[ i // 2, i % 2].set_xlabel(xlabel )
             if i % 2 == 0:
                  axarr[ i // 2, i % 2].set_ylabel(ylabel )
             axarr[ i // 2, i % 2 ].set_xticks
        print ('Now saving plot... ')
        plt.savefig(self.target_dir + line_variation_image_name )
        plt.close('all')
        """
        return 1


    def saveFullSpectrum(self, parallel_spec_wavelengths, full_pix_spectrum, spectrum_statistics_slice,
                          data_save_name = 'NO_NAME_SAVED_FIGURE_OF_OSELOTS_LINES.txt', save_dir = None,
                          header = 'Wavelength (nm), ' + r'Spectrum Radiance (Ry nm$^{-1}$), ' + r'Background Radiance (Ry nm$^{-1}$), '):
        """
        Saves a data file of the measured spectrum, in true sky units
            (Rayleigh/nm vs wavelength) and the identified sky lines.
            The measured radiance of the backgrond (subject to the
            same processing steps as the main spectrum) is provided
            as a source of statistical uncertainty.
        """
        if save_dir == None:
            save_dir = self.target_dir
        can.saveListsToColumns([parallel_spec_wavelengths, full_pix_spectrum, spectrum_statistics_slice], data_save_name, save_dir, sep = ', ', header = header)
        return 1


    def plotFullLineImage(self, parallel_spec_wavelengths, full_pix_spectrum, spectrum_statistics_slice,
                                full_emission_line_centers, full_emission_heights, full_emission_floors, wavelength_of_mu_solution,
                                show_fits, save_final_plot,
                                xlabel = 'Sky wavelength (nm)', spec_ylabel = r'Radiance (Ry nm$^{-1}$)', noise_ylabel = 'Noise ' + r'(Ry nm$^{-1}$)', plot_title = None, xticks = np.arange(400, 1301, 100),
                                xlims = None, ylims = None, save_image_name = 'NO_NAME_SAVED_FIGURE_OF_OSELOTS_LINES.txt', legend_pos = [[0.1, 0.75], 'center left'],
                                noise_bin_width = 10.0, labelsize = 14 ):
        """
        Saves a plot of the measured spectrum, in true sky units (Rayleigh/nm
            vs wavelength) and the identified sky lines.  This command requires
            that all of the data to be plotted (spectrum, lines) be passed as
            arguments.
        """
        if ylims == None:
            ref_val = np.max(full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
            spec_ylims = [-0.05 * ref_val, 1.05 * ref_val]
            noise_ylims = spec_ylims
        else:
            spec_ylims = ylims
            noise_ylims = spec_ylims
        if xlims == None:
            xlims = self.wavelength_target_range
        #fig, axarr = plt.subplots(2,1, figsize = [self.default_figsize[0], self.default_figsize[1] / 2])
        fig = plt.figure(figsize = [self.default_figsize[0], self.default_figsize[1] / 2])
        gs = fig.add_gridspec(5, 1)
        gs.update(hspace = 0.0)
        noise_ax = fig.add_subplot(gs[0, 0])
        spec_ax = fig.add_subplot(gs[1:, 0])

        processed_spec = spec_ax.plot(parallel_spec_wavelengths, full_pix_spectrum, c = 'k', zorder = 2)[0]
        spectrum_noise = [np.std(spectrum_statistics_slice[max(i - int(noise_bin_width / 2), 0):min(i + int(noise_bin_width / 2 + 0.5), len(spectrum_statistics_slice))]) for i in range(len(spectrum_statistics_slice)) ]
        processed_noise = noise_ax.plot(parallel_spec_wavelengths, spectrum_noise, c = 'r', zorder = 2)[0]
        identified_lines = spec_ax.scatter([wavelength_of_mu_solution(center) for center in full_emission_line_centers], np.array(full_emission_heights) + np.array(full_emission_floors), marker = 'x', c = 'orange', zorder = 3)
        identified_line_centers = [spec_ax.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'orange', linewidth = 0.75) for center in full_emission_line_centers]
        orig_line_centers = [spec_ax.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'cyan', linewidth = 0.75) for center in self.identified_full_emission_line_centers]
        spec_ax.set_xlabel(xlabel, fontsize = labelsize)
        spec_ax.set_ylabel(spec_ylabel, fontsize = labelsize)
        noise_ax.set_ylabel(noise_ylabel, fontsize = labelsize)
        #plt.suptitle(plot_title)
        spec_ax.legend([processed_spec, processed_noise, identified_lines ], ['Continuum-subtracted sky spectrum', 'Spectrum root variance', 'Identified sky lines'], bbox_to_anchor=legend_pos[0], loc = legend_pos[1], )
        noise_ax.set_xticks(xticks)
        noise_ax.set_xticklabels([])
        spec_ax.tick_params(axis="x",direction="in")
        spec_ax.set_xticks(xticks)
        spec_ax.set_xticklabels(xticks, fontsize = labelsize )
        #plt.tight_layout()
        print ('spec_ylims = ' + str(spec_ylims))
        spec_ax.set_ylim(*spec_ylims)
        noise_ax.set_ylim(*noise_ylims)
        spec_ax.set_xlim(*xlims)
        noise_ax.set_xlim(*xlims)
        if save_final_plot:
            print ('Saving processed sky spectrum to figure named ' + save_image_name + '!')
            plt.savefig(self.target_dir + save_image_name )
            if not (show_fits):
                plt.close('all')
        if show_fits:
            plt.show()
        plt.close('all')
        return 1

    def saveFullSpectrumWithSteps(self, parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, full_pix_background_stats, continuum_interp, full_pix_spectrum,
                                  data_save_name = 'NO_NAME_SAVED_FIGURE_OF_OSELOTS_LINES.txt', save_dir = None,
                                  header = 'Wavelength (nm), ' + 'Continuum-Subtracted Spectrum Radiance (Ry/nm), ' + 'Background Radiance (Ry/nm), ' + 'Spectrum before throughput (ADU), ' + 'Sky spectrum pre-continuum subtraction (Ry/nm), ' + ' Throughput (Ry/ADU), ' + ' Splined Continuum(Ry/nm)'):
        """
        Saves a data file of the measured spectrum, in true sky units (Rayleigh/nm
            vs wavelength) and the identified sky lines.  Also listed are
            some steps in the data reduction process.  Also listed are the
            statistics of the background (region with no spectrum), subject to the
            same reduction process.
        """
        if save_dir == None:
            save_dir = self.target_dir

        interped_throughput = self.throughput_interp(parallel_spec_wavelengths)
        pre_throughput_spec = no_background_sub_full_pix_spectrum / interped_throughput
        continuum_estimate = continuum_interp(parallel_spec_wavelengths)
        data_to_save = [parallel_spec_wavelengths, full_pix_spectrum, full_pix_background_stats, pre_throughput_spec, no_background_sub_full_pix_spectrum, interped_throughput, continuum_estimate]
        can.saveListsToColumns(data_to_save, data_save_name, save_dir, sep = ', ', header = header)


        return 1



    def plotFullAnalysisStepImage(self, parallel_spec_wavelengths, no_background_sub_full_pix_spectrum, full_pix_background_stats, continuum_interp, full_pix_spectrum,
                                full_emission_line_centers, full_emission_heights, full_emission_floors, continuum_fit_points, wavelength_of_mu_solution,
                                show_fits, save_final_plot, xlabel = 'Sky wavelength (nm)', ylabel = 'Relative intensity (normalized)', plot_title = None, xticks = np.arange(400, 1301, 100),
                                xlims = None, ylims = None, save_image_name = 'NO_NAME_SAVED_FIGURE_OF_OSELOTS_LINES.txt', legend_pos = [[0.1, 0.75], 'center left']):
        """
        Saves a plot of the measured spectrum, in true sky units (Rayleigh/nm
            vs wavelength) and the identified sky lines.  Also plotted are
            various steps in the data reduction process. This command requires
            that all of the data to be plotted (spectrum, lines) be passed as
            arguments.
        """
        if ylims == None:
            ylims = self.norm_ylims
        if xlims == None:
            xlims = self.wavelength_target_range
        fig = plt.figure(figsize = [self.default_figsize[0], self.default_figsize[1] / 2])
        norm_val = np.max(no_background_sub_full_pix_spectrum[self.spec_range[0]:self.spec_range[1]])
        full_spec = plt.plot(parallel_spec_wavelengths, np.array(no_background_sub_full_pix_spectrum) / np.max(no_background_sub_full_pix_spectrum), c = 'blue', zorder = -1)[0]
        #background_spec = plt.plot(parallel_spec_wavelengths, full_pix_background_stats, c = 'red', zorder = -2)[0]
        interped_throughput = self.throughput_interp(parallel_spec_wavelengths)
        pre_throughput_spec = plt.plot(parallel_spec_wavelengths, no_background_sub_full_pix_spectrum / interped_throughput / np.max(no_background_sub_full_pix_spectrum / interped_throughput), c = 'purple', zorder = -2)[0]
        continuum_estimate = plt.plot(parallel_spec_wavelengths, continuum_interp(parallel_spec_wavelengths) / np.max(np.max(no_background_sub_full_pix_spectrum / interped_throughput)), c = 'green', zorder = 0, linestyle = '--')[0]
        background_sub_spec = plt.plot(parallel_spec_wavelengths, full_pix_spectrum / np.max(full_pix_spectrum), c = 'k', zorder = 2)[0]
        identified_lines = plt.scatter([wavelength_of_mu_solution(center) for center in full_emission_line_centers], (np.array(full_emission_heights) + np.array(full_emission_floors)) / np.max(full_pix_spectrum), marker = 'x', c = 'orange', zorder = 3)
        continuum_sampling_points = plt.scatter(continuum_fit_points[0], np.array(continuum_fit_points[1]) / np.max(no_background_sub_full_pix_spectrum), marker = 'o', color = 'green', zorder = 1)
        identified_line_centers = [plt.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'orange', linewidth = 0.75) for center in full_emission_line_centers]
        orig_line_centers = [plt.axvline(wavelength_of_mu_solution(center), linestyle = '--', color = 'cyan', linewidth = 0.75) for center in self.identified_full_emission_line_centers]
        throughput = plt.plot(parallel_spec_wavelengths, [ylims[1] * 2 if np.isinf(elem) else elem for elem in interped_throughput ] / np.max([elem for elem in interped_throughput if not(np.isinf(elem))]), c = 'red', zorder = -4, alpha = 1.0, linestyle = '--' )[0]
        #for i in range(len(full_emission_heights)):
        #    plt.text(self.wavelength_of_mu_solution(full_emission_line_centers[i]), -500 + (i % 3) * 100, str(i), color = 'k', zorder = 4, fontsize = 8, horizontalalignment = 'center', )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print ('plot_title = ' + str(plot_title))
        plt.title(plot_title)
        plt.legend([pre_throughput_spec, throughput, full_spec, continuum_estimate, continuum_sampling_points, background_sub_spec, identified_lines], ['Spectrum before throughput (ADU)', 'Throughput (Ry / ADU)', 'Sky spectrum (Ry)', 'Continuum estimate (Ry)', 'Continuum interpolation points', 'Continuum-subtracted sky spectrum (Ry)', 'Identified sky lines'], bbox_to_anchor=legend_pos[0], loc = legend_pos[1], )
        #plt.scatter(ref_line_wavelengths, ref_line_heights , marker = 'x')
        plt.xticks(xticks)
        #plt.tight_layout()
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

    def pullCombinedSpectrumFromImages(self, spec_images_to_stack, show_fits = None, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked Spectrum', save_intermediate_images = 0, stacked_image_name = None, apply_background_correction = 1, apply_scatter_correction = 0):
        """
        This function pulls the 1d spectrum from the average of the provided list
            of images.  This combined spectrum is useful for identifying lines,
            particularly faint lines, in the source file that aren't as easily
            identified in individual images.
        The standard reduction pipeline uses these combined images to identify all
            lines that are then looked for in the individual images.  These lines
            are then looked for in the individual images.  It's easier to detect
            these faint lines in individual images if we know where to look first.
        The lines identified in this combined spectrum are also the lines whose
            intensities are traced out in time and plotted by the
            plotScaledLineProfilesInTime functions.
        """
        if show_fits != None:
            self.show_fits = show_fits
        if self.wavelength_of_mu_solution == None:
            self.getWavelengthSolution()
        if stacked_image_name == None:
            stacked_image_name = self.stacked_image_name
        if line_dict_id == None:
            line_dict_id = self.stacked_image_line_dict_id
        #print ('self.n_std_for_strong_line = ' + str(self.n_std_for_strong_line))
        self.parallel_spec_pixels, self.full_pix_spectrum, self.full_pix_background_stats, self.stacked_header = self.reduceImagesTo1dSpectrum(spec_images_to_stack, self.n_std_for_strong_line, save_image_name = self.stacked_image_name + self.data_image_suffix, save_intermediate_images = save_intermediate_images, determine_new_line_curvature = 1, apply_background_correction = apply_background_correction, apply_scatter_correction = apply_scatter_correction)
        print ('self.full_pix_spectrum = ' + str(self.full_pix_spectrum))
        self.parallel_spec_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_spec_pixels]

        self.no_background_sub_full_pix_spectrum = self.full_pix_spectrum[:]
        self.continuum_interp, self.full_pix_spectrum, self.continuum_fit_points = self.subtractContinuum(self.parallel_spec_pixels, self.full_pix_spectrum, show_fits = show_fits)
        self.continuum = self.continuum_interp(self.parallel_spec_wavelengths)
        full_wavelengths = [self.wavelength_of_mu_solution(pix) for pix in self.parallel_spec_pixels]

        pix_target_range = [max(int(self.mu_of_wavelength_solution(self.wavelength_target_range[0])), 0), min(int(self.mu_of_wavelength_solution(self.wavelength_target_range[1])), self.max_dimension) ]
        self.full_absorbtion_crossings, self.full_emission_crossings = self.detectLinesCentersInOneD(self.parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], spec_grad_rounding = 5, n_std_for_line = self.n_std_for_full_line, show = self.show_fits, background_bin_width = self.pix_bin_to_meas_background_noise )

        self.full_emission_fits = self.identifyLinesOnSlice(self.parallel_spec_pixels[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                 max_line_fit_width = self.width_pix_sample_to_fit_line, peak_guesses = self.full_emission_crossings, show_spec = self.show_fits, verbose = 0,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1, background_bin_width = self.background_bin_width_for_line_id_in_slice)
        full_emission_centroids = [fit[1] for fit in self.full_emission_fits ]
        self.seeing_fit_params = self.simulFitLineWidths(np.array(self.parallel_spec_pixels) , self.full_pix_spectrum, self.full_emission_fits)

        self.seeing_fit_funct = np.poly1d(self.seeing_fit_params)
        #print ('self.seeing_fit_funct = ' + str(self.seeing_fit_funct))

        self.full_emission_fits = self.identifyLinesOnSlice(range(len(self.parallel_spec_pixels)), self.full_pix_spectrum,
                                 max_line_fit_width = self.refined_width_pix_sample_to_fit_line, peak_guesses = full_emission_centroids, show_spec = self.show_fits, verbose = 0,
                                 fit_lines_with_pedestal = 0, fit_line_width_with_seeing_funct = 1,   background_bin_width = self.background_bin_width_for_line_id_in_slice)
        self.full_emission_fits = [fit for fit in self.full_emission_fits if (fit[1] > max(self.parallel_spec_pixels[1], self.mu_of_wavelength_solution(self.wavelength_target_range[0]))
                                                                              and fit[1] < min(self.parallel_spec_pixels[-2], self.mu_of_wavelength_solution(self.wavelength_target_range[1])) )]
        #print ('self.full_emission_fits = ' + str(self.full_emission_fits))
        self.identified_full_emission_line_centers = [fit[1] for fit in self.full_emission_fits]
        full_emission_heights = [fit[0] for fit in self.full_emission_fits]
        full_emission_floors = [fit[3] for fit in self.full_emission_fits]
        full_emission_pixels_vs_widths = [[fit[1], fit[2]] for fit in self.full_emission_fits]
        self.identified_lines_dict[self.stacked_image_keyword] = {}
        self.identified_lines_dict[self.stacked_image_keyword][self.lines_in_dict_keyword] = {i:self.full_emission_fits[i] for i in range(len(self.full_emission_fits))}
        self.identified_lines_dict[self.stacked_image_keyword][self.obs_time_keyword] = self.stacked_header[self.obs_time_keyword]
        #print('self.full_emission_pixels_vs_widths = ' + str(self.full_emission_pixels_vs_widths))
        #self.seeing_fit_funct = np.poly1d(can.polyFitNSigClipping( [emit[0] for emit in full_emission_pixels_vs_widths], [emit[1] for emit in full_emission_pixels_vs_widths], 2, self.sig_clip_for_line_width) [3])
        if self.show_fits or self.save_final_plot:
            print ('self.full_pix_spectrum = ' + str(self.full_pix_spectrum))
            self.plotFullAnalysisStepImage(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.no_background_sub_full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], self.continuum_interp, self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                    self.identified_full_emission_line_centers, full_emission_heights, full_emission_floors, self.continuum_fit_points, self.wavelength_of_mu_solution,
                                    self.show_fits, self.save_final_plot, plot_title = plot_title,
                                    save_image_name = stacked_image_name  + self.processed_multistep_spectra_image_suffix + self.figure_suffix)

            self.plotFullLineImage(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]],
                                    self.identified_full_emission_line_centers, full_emission_heights, full_emission_floors, self.wavelength_of_mu_solution,
                                    self.show_fits, self.save_final_plot, plot_title = plot_title, noise_bin_width = self.pix_bin_to_meas_background_noise,
                                    save_image_name = stacked_image_name  + self.processed_spectra_image_suffix + self.figure_suffix)

        if self.save_spectra:
            self.saveFullSpectrum(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]],
                                  data_save_name = stacked_image_name  + self.processed_spectra_image_suffix + self.text_suffix)
            self.saveFullSpectrumWithSteps(self.parallel_spec_wavelengths[pix_target_range[0]:pix_target_range[1]], self.no_background_sub_full_pix_spectrum[pix_target_range[0]:pix_target_range[1]], self.full_pix_background_stats[pix_target_range[0]:pix_target_range[1]], self.continuum_interp, self.full_pix_spectrum[pix_target_range[0]:pix_target_range[1]],
                                           data_save_name = stacked_image_name  + self.processed_multistep_spectra_image_suffix + self.text_suffix)

        return 1

    def loadSpecProcessor(self, load_name, load_dir = None):
        """
        This function loads a saved version of this class into python memory.  This is
            very useful, as it prevents the user from having to reprocess a stack of
            images when reexamining data.
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
        self.anchor_parallel_pix = float(l_anchor_pix[1])
        l_spec_range = load_object.readline().split(' ')
        l_spec_range[-1] = l_spec_range[-1][:-1]
        self.spec_range = [int(term) for term in l_spec_range[1:]]
        l_parallel_spec_pixels = load_object.readline().split(' ')
        l_parallel_spec_pixels [-1] = l_parallel_spec_pixels [-1][0:-1]
        self.parallel_spec_pixels = [float(val) for val in l_parallel_spec_pixels[1:]]
        l_full_pix_spectrum = load_object.readline().split(' ')
        l_full_pix_spectrum[-1] = l_full_pix_spectrum[-1][0:-1]
        self.full_pix_spectrum = [float(val) for val in l_full_pix_spectrum[1:]]
        l_no_background_sub_full_pix_spectrum = load_object.readline().split(' ')
        l_no_background_sub_full_pix_spectrum[-1] = l_no_background_sub_full_pix_spectrum[-1][0:-1]
        self.no_background_sub_full_pix_spectrum = [float(val) for val in l_no_background_sub_full_pix_spectrum[1:]]
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
        """
        This function saves the current instance of this class into a loadable
            python object file.  This is very useful, as it prevents the user
            from having to reprocess a stack of images when reexamining data.
        """
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
        save_object.write( 'no_background_sub_full_pix_spectrum' + ' '  + ' '.join([str(can.round_to_n(val, 5)) for val in self.no_background_sub_full_pix_spectrum]) )
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
        """
        This function initializes a bunch of parameters that the processor
           class uses in analyzing a night of data.  These parameters range
           from the name of master bias and dark files to the number of
           standard deviations needed to identify an emission line as
           significant.
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
        self.text_suffix = self.ref_param_holder.getTextSuffix()
        self.perp_spec_root = self.ref_param_holder.getOrthogonalSpectrumFileRoot()
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
        self.std_thresh_for_scatter_correct = self.ref_param_holder.getStdThreshForScatterCorrecting()

        self.perp_spec_save_file = self.perp_spec_root + self.text_suffix

        self.fits_exp_time_keyword = self.ref_param_holder.getExpTimeKeyword()

        self.wavelength_range = self.ref_param_holder.getWavelengthRangeOfInterest()

        self.throughput_interp = None
        self.mu_of_wavelength_solution = None
        self.wavelength_of_mu_solution = None
        self.spec_range = None

        return 1


    def __init__(self, target_dir,
                 master_bias_prefix = 'BIAS', master_dark_prefix = 'DARK', ref_spec = 'KR1', bias_dir = None, dark_dir = None,
                 processed_file_suffix = '_proc', processed_spectra_image_suffix = '_spec', perp_spec_image_suffix = '_perp_spec', processed_prefix = 'proc_',
                 data_image_suffix = '.fits', save_image_suffix = '.pdf', list_suffix = '.list', sig_clip_for_line_width = 3.5, save_stacked_image = 1, save_spectra = 1,
                 crc_correct = 1, do_bias = 1, do_dark = 1, redo_master_bias = 0, redo_master_dark = 0, cosmic_prefix = 'crc_', show_fits = 1, save_final_plot = 1, save_perp_spec_image = 1, spec_axis = 0,
                 background_buffer = 10, background_size = 100, background_low = 1, n_std_for_strong_line = 20.0, n_std_for_full_line = 10.0,
                 archival_data_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/',
                 ref_params_file = 'OSELOTSDefaults.txt', ref_params_dir = '/Users/sashabrownsberger/Documents/sashas_python_scripts/skySpectrograph/',
                 scatter_data_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut20211210/', scatter_data_key_file = 'scatter_map_Mono_2021_12_10.txt',
                 remove_intermed_images = 1, stacked_image_name_root = None, date = None, throughput_file = None):

        self.date = date

        self.ref_param_holder = ref_param.CommandHolder(spectrograph_file = ref_params_file, defaults_dir = ref_params_dir)
        self.initialize_params_from_ref_params()
        if stacked_image_name_root != None:
            self.stacked_image_name = stacked_image_name_root
        self.spec_archival_info = {'KR1':{'spec_file':'KR1LinesSpec.csv','n_spec_lines_to_ignore':1, 'lines_file':'KR1_lines_all.txt', 'n_lines_lines_to_ignore':1},
                                   'HG2':{'spec_file':'HG2LinesSpec.csv','n_spec_lines_to_ignore':1, 'lines_file':'HG2_lines_for_OSELOTS.txt', 'n_lines_lines_to_ignore':1},
                              #'Gemini':{'spec_file':'GeminiSkyLines.txt','n_lines_to_ignore':14},
                             'Gemini':{'spec_file':'GeminiSkyBrightness.txt','n_lines_to_ignore':14},
                              'throughput':{'spec_file':'OSELOTS_throughput.txt','n_lines_to_ignore':1} }
        self.scatter_data_dir = scatter_data_dir
        self.scatter_data_key_file = scatter_data_key_file
        self.scatter_image_file_name = self.scatter_data_key_file.split('.')[0] + '.fits'
        #data_image_suffix = self.ref_param_holder.getImageSuffix()
        #list_suffix = self.ref_param_holder.getImageSuffix()
        #master_bias_prefix = self.ref_param_holder.getMasterBiasPrefix()
        #master_dark_prefix = self.ref_param_holder.getMasterDarkPrefix()
        #self.spec_files = spec_files
        self.do_bias = do_bias
        self.do_dark = do_dark
        self.redo_master_bias = redo_master_bias
        self.redo_master_dark = redo_master_dark
        self.ref_spec = ref_spec
        self.show_fits = show_fits
        self.save_spectra = save_spectra
        self.crc_correct = crc_correct
        self.save_perp_spec_image = save_perp_spec_image
        self.save_stacked_image = save_stacked_image
        self.save_final_plot = save_final_plot
        self.ref_sky_lines_data = can.readInColumnsToList(self.ref_sky_lines_file, self.archival_data_dir, n_ignore = self.ref_sky_lines_file_n_ignore, convert_to_float = 1, verbose = 0)
        self.identified_lines_dict = {}
        self.xlims = [450, 1350]
        self.norm_ylims = [-0.2, 1.2]
        self.ylims = [-0.2, 1.2]
        self.remove_intermed_images = remove_intermed_images
        if throughput_file != None:
            self.throughput_file = throughput_file
        self.throughput_interp = self.importSystemThroughput()
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        #spec_files = spec_file.replace('[','')
        #spec_files = spec_files.replace(']','')
        #spec_files = spec_files.split(',')
        self.target_dir = target_dir
        if bias_dir == None:
            self.bias_dir = self.target_dir
        else:
            self.bias_dir = bias_dir
        if dark_dir == None:
            self.dark_dir = self.target_dir
        else:
            self.dark_dir = dark_dir
        print ('self.dark_dir = ' + str(self.dark_dir))
        self.seeing_fit_params = np.zeros(self.seeing_fit_order + 1)
        self.seeing_fit_params[-1] = self.init_seeing_guess
        self.seeing_fit_funct = np.poly1d(self.seeing_fit_params)
        self.strong_line_2d_profile_fit = None

if __name__ == "__main__":
    date = ['2022', '05', '19'] # date = ['2022', '05', '17']
    date_str = '_'.join(date)
    f_pos = '24p2'
    arc_lamp_str = 'HG2'
    do_dark = 1
    scatter_correct = 0
    background_correct = 0
    determine_spec_sol = 1 #Should be set to 1 when a new set of reference spectral data is taken, from arc lamp source
    target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/' + date_str + '/'
    scatter_data_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/' + date_str + '/'
    scatter_data_key_file = 'scatter_map_Mono_' + '_'.join(date) + '.txt'
    print ('scatter_data_dir + scatter_data_key_file = ' + str(scatter_data_dir + scatter_data_key_file))
    #processor = SpectrumProcessor(target_dir, show_fits = 0, scatter_data_dir = scatter_data_dir, scatter_data_key_file = scatter_data_key_file, date = date, do_dark = do_dark)
    processor = SpectrumProcessor(target_dir, show_fits = 0, date = date, do_dark = do_dark, ref_spec = arc_lamp_str)

    dark_nums = list(range(16, 29, 3)) # list(range(210, 241, 3))
    sky_nums = list(range(31, 83, 3)) + list(range(251, 501, 3))# list(range(58, 209, 3)) + list(range(243, 811, 3))
    sky_nums = sky_nums[0:-10]
    dark_sky_nums = sky_nums[0:-10] #sky_nums[30:-30]
    arc_lamp_nums = [12] # list(range(41, 51))
    bias_nums = can.flattenListOfLists([[num - 1, num + 1] for num in dark_nums]) + can.flattenListOfLists([[num - 1, num + 1] for num in sky_nums])
    bias_nums = bias_nums + list(range(85, 250))
    bias_nums = can.safeSortOneListByAnother(bias_nums, [bias_nums])[0]
    sky_nums = sky_nums + [503]
    bias_imgs = ['Bias_' + date_str + '_' + str(i) + '.fits' for i in bias_nums]
    can.saveListsToColumns(bias_imgs, processor.master_bias_list, target_dir)
    #processor.plotBiasLevels(bias_list = 'BIAS.list')
    #dark_nums = []
    dark_imgs = ['Dark_f' + f_pos + '_' + date_str + '_' + str(i) + '.fits' for i in dark_nums]
    can.saveListsToColumns(dark_imgs, processor.master_dark_list, target_dir)
    arc_lamp_imgs = [arc_lamp_str + '_f' + f_pos + '_' + date_str + '_' + str(i) + '.fits' for i in arc_lamp_nums]
    can.saveListsToColumns(arc_lamp_imgs, arc_lamp_str + processor.ref_param_holder.getListSuffix(), target_dir )
    ref_spec_solution_file = processor.ref_param_holder.getRefSpecSolutionFile()
    if determine_spec_sol:
        processor.getWavelengthSolution(ref_spec_images_list = arc_lamp_str + processor.ref_param_holder.getListSuffix(), ref_spec_solution_file = ref_spec_solution_file, save_new_reference_spectrum = 1, ref_spec = None, show_fits = 0)
        #As a check, process the reference wavelength images:
        processor.pullCombinedSpectrumFromImages(arc_lamp_imgs, show_fits = 0, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked ' + arc_lamp_str + ' Spectrum', save_intermediate_images = 0, stacked_image_name = 'Stacked' + arc_lamp_str + 'Image_img' + str(arc_lamp_nums[0]) + 'To' + str(arc_lamp_nums[-1]), apply_background_correction = background_correct, apply_scatter_correction = 0)
        for i in range(len(arc_lamp_imgs)):
            img = arc_lamp_imgs[i]
            img_num = arc_lamp_nums[i]
            processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)

    #Now we should reinitiate the processor so that we don't try to match reference and sky lines
    processor = SpectrumProcessor(target_dir, show_fits = 0, date = date, do_dark = do_dark, ref_spec = arc_lamp_str)

    dark_sky_imgs = ['sky_f' + f_pos + '_' + date_str + '_' + str(i) + '.fits' for i in dark_sky_nums]
    sky_imgs = ['sky_f' + f_pos + '_' + date_str + '_' + str(i) + '.fits' for i in sky_nums]

    #all_sky_nums = [161, 162, 163]
    #dark_sky_imgs = [ 'Mono_' + str(800) + 'nm_f' + focus_str + '_' + '_'.join(date) + '_' + str(j) + '.fits' for j in all_sky_nums]
    #all_sky_imgs = [ 'Mono_' + str(800) + 'nm_f' + focus_str + '_' + '_'.join(date) + '_' + str(j) + '.fits' for j in all_sky_nums ]

    processor.pullCombinedSpectrumFromImages(dark_sky_imgs, show_fits = None, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked Spectrum', save_intermediate_images = 0, stacked_image_name = 'StackedSkyImage_img' + str(dark_sky_nums[0]) + 'To' + str(dark_sky_nums[-1]), apply_scatter_correction = scatter_correct)
    processor.saveSpe
    for i in range(len(sky_imgs)):
        img = sky_imgs[i]
        img_num = sky_nums[i]
        processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)
    processor.plotScaledLineProfilesInTime()
    processor.plotLineProfilesInTime()
    processor_python_obj_save_file = 'FullNight_ut' + ''.join(date) + '.prsc'
    processor.saveSpecProcessor(processor_python_obj_save_file, save_dir = None, )
    print ('You can reload the saved spectrum processor using the following (in the Python environment): ')
    print ('import ProcessRawSpectrumClass as prsc')
    print ("target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut'" + ''.join(date) + " + '/' " )
    print ('processor_reloaded = prsc.SpectrumProcessor(target_dir, show_fits = 0)')
    print ("processor_reloaded.loadSpecProcessor(processor_python_obj_save_file, load_dir = None)")
