import bashVarContainer as bvc
import cantrips as can
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

class CommandHolder:

    def getOSELOTSVar(self, var_key):
        return self.varContainer.getVarFromDictPython(var_key)

    def getAmpCorrectColRange(self):
        return can.recursiveStrToListOfLists(self.getPISCOVar('amp_correct_col_range1x1'), elem_type_cast = int)

    def getAmpGainCorrectionWidth(self):
        return int(self.getPISCOVar('amp_gain_cor_bin_width'))

    def getAmpGainCorrectionFitSigClip(self):
        return int(self.getPISCOVar('amp_gain_cor_fit_sig_clip'))

    def getAmpGainCorrectionFitOrder(self):
        return int(self.getPISCOVar('amp_gain_correction_fit_order'))

    def getAmpGainPixelThreshold(self):
        return int(self.getPISCOVar('amp_gain_correct_count_threshold'))

    def getGainCorrectionPrefix(self):
        return str(self.getPISCOVar('gain_correct_prefix'))

    def getFilterHeaderKeyword(self):
        return str(self.getPISCOVar('filter_header_keyword'))

    def getNormSectKeyword(self):
        return str(self.getPISCOVar('normalization_sect_keyword'))

    def getAmpFitColRangeHeaderKeyword(self):
        return str(self.getPISCOVar('gain_fit_col_range_keyword'))

    def getAmpFitHeaderKeyword(self):
        return str(self.getPISCOVar('amp_fit_keyword'))

    def get1x1CropRegion(self):
        return can.recursiveStrToListOfLists((self.getPISCOVar('crop_1x1')), elem_type_cast = int)

    def getDefaultFigSize(self):
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('default_fig_size')), elem_type_cast = int)

    def  getNCombinedHeaderKeyword(self):
        return str(self.getPISCOVar('n_combined_header_keyword'))

    def getFilters(self):
        return can.recursiveStrToListOfLists(self.getPISCOVar('filters'))

    def getMasterFlatLabel(self):
        return str(self.getOSELOTSVar('master_flat_root'))

    def getMasterDarkLabel(self):
        return str(self.getOSELOTSVar('master_dark_root'))

    def getMasterBiasLabel(self):
        return str(self.getOSELOTSVar('master_bias_root'))

    def getRefThroughputFile(self):
        return str(self.getOSELOTSVar('ref_throughput_file'))

    def getRefSkyLinesFile(self):
        return str(self.getOSELOTSVar('ref_sky_lines_file'))

    def getNIgnoreRefSkyLinesFile(self):
        return int(self.getOSELOTSVar('n_ignore_ref_sky_lines_file'))

    def getWavelengthToPhotonScaling(self):
        return float(self.getOSELOTSVar('energy_to_photon_conversion'))

    def getSingleFilterFlatFiles(self):
        single_filt_strs = self.getFilters()
        list_suffix = self.getListSuffix()
        flat_list_name = self.getMasterFlatLabel()
        return [flat_list_name + single_filt_str + list_suffix for single_filt_str in single_filt_strs]

    def getPixelThresholds(self):
        return [int(threshold) for threshold in can.recursiveStrToListOfLists(self.getPISCOVar('pixel_thresholds'))]

    def getWavelengthRangeOfInterest(self):
        "Get wavelength range, in nm, that you want to anlalyze the spectrum over."
        return [float(bound) for bound in can.recursiveStrToListOfLists(self.getOSELOTSVar('fit_data_wavelength_range'))]

    def getBackgroundCutWavelengths(self):
        "The wavelengths that mark the edges of the spectrum.  Sections of detector below and above this range will be used to fit the background. "
        return [float(bound) for bound in can.recursiveStrToListOfLists(self.getOSELOTSVar('background_cut_wavelengths'))]

    def getPixelScale(self):
        return float(self.getPISCOVar('pixel_scale'))

    def getAmplifierGains(self):
        return [float(gain) for gain in can.recursiveStrToListOfLists(self.getPISCOVar('gains'))]

    def getSuffix(self):
        return str(self.getPISCOVar('single_band_suffix_suffix') )

    def getSingleBandSuffixPrefix(self):
        return str(self.getPISCOVar('single_band_suffix_prefix'))

    def getSingleBandSuffixes(self):
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        filt_strs = self.getFilters()
        image_extension = self.getImageExtension()
        return [single_band_suffix_prefix + filt_str + image_extension for filt_str in filt_strs]

    def getLeftRightAmpSuffixes(self):
        return can.recursiveStrToListOfLists((self.getPISCOVar('left_right_suffix')))

    #When stitching, we do need to cut off the right overscan section, as it sits at the juncture
    # between left and right amp.
    #This is separate from the crop sections.
    def getBinning1x1Cut(self):
        return int(self.getPISCOVar('binning_1x1_cut'))

    def getFlatSingleAmpMeasureSections1x1(self):
         return can.recursiveStrToListOfLists((self.getPISCOVar('flat_single_amp_measure_sections_1x1')), elem_type_cast = int)

    def getFlatStitchedMeasureSections1x1(self):
         return can.recursiveStrToListOfLists((self.getPISCOVar('flat_stitched_measure_sections_1x1')), elem_type_cast = int)

    def getGoodFlatADUs(self):
        return can.recursiveStrToListOfLists((self.getPISCOVar('flat_good_adu_levels')), elem_type_cast = int)

    def getAmpHeaderKeyword(self):
         return self.getPISCOVar('amp_keyword')

    def getExpTimeKeyword(self):
        return self.getPISCOVar('exp_time_keyword')

    def getBiasSubKeyword(self):
        return self.getPISCOVar('bias_sub_keyword')

    def getFlatNormKeyword(self):
         return self.getPISCOVar('flat_norm_keyword')

    def getDateObsKeyword(self):
        return self.getPISCOVar('date_obs_keyword')

    def getExpStartKeyword(self):
        return self.getPISCOVar('crop_keyword')

    def getCropKeyword(self):
        return self.getPISCOVar('exp_start_keyword')

    def getObjectMaxFluxKeywordPrefix(self):
        return self.getPISCOVar('obj_peak_flux_keyword_prefix')

    def getStarPositionKeywordPrefix(self):
        return self.getPISCOVar('star_position_keyword_prefix')

    def getGoodObjectKeywordPrefix(self):
        return self.getPISCOVar('good_object_keyword_prefix')

    def getStarVsGalKeywordPrefix(self):
        return self.getPISCOVar('star_gal_keyword_prefix')

    def getKeywordBandSuffix(self):
        return self.getPISCOVar('mcat_keyword_band_suffix')

    def getExpEndKeyword(self):
        return self.getPISCOVar('exp_end_keyword')

    def getBiasXPartitions(self):
        return int(self.getPISCOVar('bias_combine_x_partitions'))

    def getBiasYPartitions(self):
        return int(self.getPISCOVar('bias_combine_y_partitions'))

    def getNMosaics(self):
        return int(self.getPISCOVar('n_mosaic_extensions'))

    def getBinning(self):
        return int(self.getPISCOVar('binning'))

    def getCropPrefix(self):
        return str(self.getPISCOVar('crop_prefix'))

    def getOverscanPrefix(self):
        return str(self.getPISCOVar('overscan_prefix'))

    def getNormalizationPrefix(self):
         return str(self.getPISCOVar('normalization_prefix'))

    def getBiasSubPrefix(self):
        return str(self.getPISCOVar('bias_correction_prefix'))

    def getFlatNormPrefix(self):
        return self.getPISCOVar('flat_correction_prefix')

    def getOverscanFitOrder(self):
        return int(self.getPISCOVar('overscan_fit_order'))

    def getOverscanBuffer(self):
        return int(self.getPISCOVar('overscan_buffer'))

    def getOverscanSections1x1(self):
        return can.recursiveStrToListOfLists((self.getPISCOVar('overscan_sections_1x1')), elem_type_cast = int)

    def getListSuffix(self):
        return str(self.getOSELOTSVar('list_suffix'))

    def getImageSuffix(self):
        return str(self.getOSELOTSVar('image_suffix'))

    def getStartExposureKeyword(self):
        return str(self.getOSELOTSVar('exposure_start_keyword'))

    def getLinesDictKeyword(self):
        return str(self.getOSELOTSVar('ided_lines_keyword'))

    def getStackedKeyword(self):
        return str(self.getOSELOTSVar('stacked_image_keyword'))

    def getDateFormatString(self):
        return str(self.getOSELOTSVar('date_format'))

    def getCosmicPrefix(self):
        return str(self.getOSELOTSVar('cosmic_prefix'))

    def getArchivalDataDir(self):
        """
        The directory where we keep the archival/calibration data.
        """
        return str(self.getOSELOTSVar('archival_data_dir'))

    def getBackgroundBuffer(self):
        """
        The separation, in pixels, between the section with the spectrum
            and the section with the background.
        """
        return int(self.getOSELOTSVar('background_buffer'))

    def getBackgroundSize(self):
        """
        ???
        """
        return int(self.getOSELOTSVar('background_size'))

    def getBackgroundLow(self):
        """
        A boolean (1 or 0).  If True (1), the background is measured
           below or left (depending on the direction) of the spectrum.
        """
        return int(self.getOSELOTSVar('background_low'))

    def getSpecAxis(self):
        """
        Is the spectrum displayed along columns (value 0) or along
            rows (value 1).
        """
        return int(self.getOSELOTSVar('spec_axis'))

    def getProcessedFileSuffix(self):
        """
        The suffix to indicate we have processed an image.
        """
        return str(self.getOSELOTSVar('processed_file_suffix'))

    def getStackedImageName(self):
        """
        The name of the stacked image.
        """
        return str(self.getOSELOTSVar('stacked_image_prefix'))

    def getFigureSuffix(self):
        """
        The suffix (data type) of saved plots and non-fits images.
        """
        return str(self.getOSELOTSVar('figure_suffix'))

    def getProcessedSpectrumSuffix(self):
        """
        The suffix to indicate we have a spectrum.
        """
        return str(self.getOSELOTSVar('spectrum_suffix'))

    def getOrthogonalBinOfSpectrumSuffix(self):
        """
        The suffix to name the plot showing the binning
            orthogonal to the spectrum.
        """
        return str(self.getOSELOTSVar('perp_spec_suffix'))

    def getNStdForStrongRefLines(self):
        """
        The number of stds above background for a line to
           be id'd as a strong line, used for determinig
           curvature.
        """
        return float(self.getOSELOTSVar('n_std_for_strong_line_ref'))

    def getNStdForFullRefLines(self):
        """
        The number of stds above background for a line in
           the reference spectrum to be id'd as a true
           line, used for determinig curvature.
        """
        return float(self.getOSELOTSVar('n_std_for_full_line_ref'))

    def getNStdForStrongLines(self):
        """
        The number of stds above background for a line in
           the reference spectrum to be id'd as a strong
           line, used for determinig curvature.
        """
        return float(self.getOSELOTSVar('n_std_for_strong_line'))

    def getNStdForFullLines(self):
        """
        The number of stds above background for a line to
           be id'd as a true line, used for naming lines
           after integrating along curvature.
        """
        return float(self.getOSELOTSVar('n_std_for_full_line'))

    def getSigClipForLineWidth(self):
        """
        ???
        """
        return float(self.getOSELOTSVar('sig_clip_for_line_width'))

    def getNSigGhostsWidth(self):
        """
        ???
        """
        return float(self.getOSELOTSVar('ghosts_n_sig_width'))

    def getGhostsHigh(self):
        """
        Do ghosts get shifted high (1), or down (0) of the spectrum?
        """
        return int(self.getOSELOTSVar('ghosts_high'))

    def getGhostsRight(self):
        """
        Do ghosts get shifted right (1), or left (0) of the spectrum?
        """
        return int(self.getOSELOTSVar('ghosts_right'))

    def getMinGhosts(self):
        """
        ???
        """
        return int(self.getOSELOTSVar('min_ghosts'))

    def getRemoveGhostsByShifting(self):
        """
        Do we want to try removing ghosts by shifting the specturm and subtracting?
        """
        return int(self.getOSELOTSVar('remove_ghosts_by_shifting'))

    def getBackgroundFitOrder(self):
        """
        Order of polynomial used to fit background.
        """
        return int(self.getOSELOTSVar('background_fit_order'))

    def getNStdForMostGhosts(self):
        """
        Number of STD for something to be id'd as a ghost, once we think a ghost is there .
        """
        return float(self.getOSELOTSVar('n_std_for_most_ghosts'))

    def getNStdForFirstGhosts(self):
        """
        Number of STD for id'ing the first ghost.
        """
        return float(self.getOSELOTSVar('n_std_for_first_ghosts'))

    def getGhostSearchBuffer(self):
        """
        ???
        """
        return can.recursiveStrToListOfLists((self.getPISCOVar('ghost_search_buffer')), elem_type_cast = int)

    def getCleanBuffer(self):
        """
        ???
        """
        return int(self.getOSELOTSVar('clean_buffer'))

    def getMinDetectionsForIdentifingALine(self):
        """
        The number of pixels in which an initially detected line must be detected
            to be kept as a line in 2d.
        """
        return int(self.getOSELOTSVar('min_detections_for_ident_as_line'))

    def getStrongLineSearchBinning(self):
        """
        Get the type of binning done to look for strong lines.  Options are:
           "full" - ???
           ???
        """
        return str(self.getOSELOTSVar('strong_line_search_binning'))

    def getStrongLineFitBinning(self):
        """
        Get the number of rows to coadd to look for centroids of strong lines.
        """
        return int(self.getOSELOTSVar('strong_line_fit_binning'))

    def getMaxFittedLineWidth(self):
        """
        Get the upper bounds on the Gaussian widths used by the line fitter.
        """
        return int(self.getOSELOTSVar('max_line_fit_width'))

    def getInitialLineWidthGuesss(self):
        """
        Get the initial guess of the Gaussian line width used by the
            line fitter.
        """
        return float(self.getOSELOTSVar('line_width_guess'))

    def getContinuumSmoothing(self):
        """
        Get the width of the Gaussian used to smooth the spectrum,
           prior to determining the continuum.
        """
        return int(self.getOSELOTSVar('continuum_smoothing_pix'))

    def getContinuumSeedFile(self):
        """
        Get the file where we list the seeds to measure the continuum.
        """
        return str(self.getOSELOTSVar('continuum_seed_file'))

    def getNIgnoreContinuumSeedFile(self):
        """
        Get number of lines to ignore in the file where we list the
            seeds to measure the continuum.
        """
        return int(self.getOSELOTSVar('n_ignore_in_seed_file'))

    def getNContinuumSeeds(self):
        """
        Get the number of seeds that will be placed to measure
            the spectrum continuum.
        """
        return int(self.getOSELOTSVar('n_continuum_fit_seeds'))

    def getThroughputFile(self):
        """
        Return the name of the file that contains the import
            throughput (frac in / frac out vs wavelength.)
        """
        return self.getOSELOTSVar('throughput_file')

    def getMaxContinuumSeedLineSep(self):
        """
        Get the minimium separation, in pixels, between an id'd
            line and a seed used for measuring the continuum.
        """
        return int(self.getOSELOTSVar('min_line_vs_seed_sep_pix'))

    def getMaxSepForLineTrace(self):
        """
        Get the maximum separation to trace adjance lines, in pixels.
        """
        return float(self.getOSELOTSVar('max_sep_per_pixel_for_line_trace'))

    def getParallelSmoothing(self):
        """
        Get the pixel smoothing for measuring along spectrum.
        """
        return int(self.getOSELOTSVar('parallel_smoothing'))

    def getInitSeeing(self):
        """
        Get the initial guess of the seeing - used for the initial crude fit to get line centroids.
        """
        return float(self.getOSELOTSVar('init_seeing_guess'))

    def getSeeingOrder(self):
        """
        Get the polynomial order of the seeing width function.
        """
        return int(self.getOSELOTSVar('seeing_fit_order'))

    def getStdThreshForNewLinesOnSlice(self):
        """
        Get the number of background stds above which pixels must
            be to be considered a possible line in a cross-slice.
        """
        return float(self.getOSELOTSVar('slice_std_thresh_for_iding_new_lines'))

    def getPixelWidthToFitALine(self):
        """
        Get width in pixels of the subslice of a spectrum to fit to a line.
        """
        return float(self.getOSELOTSVar('pixel_width_to_fit_line'))

    def getPixelWidthToFitARefLine(self):
        """
        Get width in pixels of the subslice of a spectrum to fit to a reference line.
        """
        return float(self.getOSELOTSVar('pixel_width_to_fit_ref_line'))


    def getRefinedPixelWidthToFitALine(self):
        """
        Get width in pixels of the subslice of a spectrum to fit to a line
        in the second, more refined fit.
        """
        return float(self.getOSELOTSVar('refined_pixel_width_to_fit_line'))


    def getLineDictIDOfStackedImage(self):
        """
        Get the name of the keyword to represent the stacked image in the
           dictionary of identified lines.
        """
        return str(self.getOSELOTSVar('stack_line_dict_id'))

    def getBackgroundSigClip(self):
        """
        Get the sigma clipping of the to measure the background std.
        """
        return float(self.getOSELOTSVar('sig_clip_background'))

    #self.width_pix_sample_to_fit_line = self.ref_param_holder.getPixelWidthToFitToALine()
    #self.n_pix_above_thresh_for_new_line_in_slice = self.ref_param_holder.getNPixToIDANewLine()


    def getNPixAboveThreshForNewLine(self):
        """
        Get the minimum number of pixels for a line to be newly identified on a slice.
        """
        return float(self.getOSELOTSVar('n_pix_above_thresh_for_new_line_in_slice'))

    def getBackgroundBinWidthForNewLineOnSlice(self):
        """
        Get the smoothing applied to the background to identify a new line on a slice.
        """
        return float(self.getOSELOTSVar('background_bin_width'))

    def getMaxPixelsThatACentroidCanBeAdjusted(self):
        """
        Get the number of pixels that a line centroid can shift in a fit.
        """
        return float(self.getOSELOTSVar('max_pix_centroid_shift'))

    def getLineMeanFitOrder(self):
        """
        Polynomial order for fitting the centroid of the traced lines.
        """
        return float(self.getOSELOTSVar('line_mean_fit_order'))

    def getLineAmplitudeFitOrder(self):
        """
        Polynomial order for fitting the amplitude of the traced lines.
        """
        return float(self.getOSELOTSVar('line_amplitude_fit_order'))

    def getLineWidthFitOrder(self):
        """
        Polynomial order for fitting the width of the traced lines.
        """
        return float(self.getOSELOTSVar('line_width_fit_order'))

    def getGhostSearchBuffer(self):
        """
        ???
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('ghost_search_buffer')), elem_type_cast = float)

    def getBackgroundFitRegion(self):
        """
        Get image range over which to fit background. These are the length
        of pixels left, right, above, and below the spectrum region to do
        the fits.
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('background_fit_pix_range')), elem_type_cast = int)

    def getWavelengthOfPixSolutionGuess(self):
        """
        Get background wavelength solution guess.
            Based on historic, hand-determine solutions.
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('wavelength_of_pix_solution_guess')), elem_type_cast = float)

    def getCoarseSearchParamRange(self):
        """
        Get range of linear params to determine wavelength solution.
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('coarse_search_param_range')), elem_type_cast = float)

    def getRefSpecSolutionFile(self):
        """
        The default name of the reference spectrum solution.
        """
        return str(self.getOSELOTSVar('ref_spec_solution_file'))

    def getCoarseSearchNParamStep(self):
        """
        Get number of linear params we sample.
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('coarse_search_n_steps')), elem_type_cast = int)

    def getWavelengthSolutionOrder(self):
        """
        Get the order of the wavelength solution.
        """
        return int(self.getOSELOTSVar('wavelength_solution_order'))

    def getWavelengthScalings(self):
        """
        Get the fit scalings of the wavelength terms.
        """
        return can.recursiveStrToListOfLists((self.getOSELOTSVar('wavelength_fit_scalings')), elem_type_cast = float)

    def getSigSepForDistinctLines(self):
        """
        Get number of std for two lines to be labeled as distinct.
        """
        return float(self.getOSELOTSVar('min_sig_sep_for_distinct_lines'))

    def getPixScale(self):
        """
        Get approximate pixel scale.
        """
        return float(self.getOSELOTSVar('approx_pix_scale'))

    def getCrudeGaussianSmoothWidth(self):
        """
        Get gaussian width to due crude spectrum matching, in pixels.
        """
        return float(self.getOSELOTSVar('crude_fit_gauss_width'))

    def getBiasList(self):
        bias_label = self.getMasterBiasLabel()
        list_suffix = self.getListSuffix()
        return bias_label + list_suffix

    def getDarkList(self):
        dark_label = self.getMasterDarkLabel()
        list_suffix = self.getListSuffix()
        return dark_label + list_suffix

    def getFlatList(self):
        flat_label = self.getMasterFlatLabel()
        list_suffix = self.getListSuffix()
        return flat_label + list_suffix

    def getMasterBiasName(self):
        bias_label = self.getMasterBiasLabel()
        image_suffix = self.getImageSuffix()
        return bias_label + image_suffix

    def getMasterDarkName(self):
        dark_label = self.getMasterDarkLabel()
        image_suffix = self.getImageSuffix()
        return dark_label + image_suffix

    def getMasterFlatName(self):
        flat_label = self.getMasterFlatLabel()
        image_suffix = self.getImageSuffix()
        return flat_label + image_suffix

    def getSingleBandFlatLists(self):
        single_band_suffix_prefix = self.getSingleBandSuffixPrefix()
        flat_label = self.getMasterFlatLabel()
        list_suffix = self.getListSuffix()
        filt_strs = self.getFilters()
        return [flat_label + single_band_suffix_prefix + filt_str + list_suffix for filt_str in filt_strs]

    def getFullFlatList(self):
        return str(self.getOSELOTSVar('full_flat_list'))

    def getMasterBiasRoot(self):
        return str(self.getOSELOTSVar('master_bias_root'))

    def getMasterFlatRoot(self):
        return str(self.getOSELOTSVar('master_flat_root'))

    def getSharedCalDir(self):
        base_dir = str(self.getPISCOVar('pisco_base_dir'))
        cal_dir = str(self.getPISCOVar('pisco_cal_dir'))
        return base_dir + cal_dir

    def getImageExtension(self):
        return str(self.getPISCOVar('image_extension'))

    def getCatalogueExtension(self):
        return str(self.getPISCOVar('catalogue_extension'))

    def getXYPositionTextFileSuffix(self):
        return str(self.getPISCOVar('xypos_test_suffix'))

    def getXYPositionFitsFileSuffix(self):
        return str(self.getPISCOVar('xypos_fits_suffix'))

    def getIndicatorOfHeaderInCatalogue(self):
        return str(self.getPISCOVar('catalogue_header_identifier'))

    def getRoughRAKeyword(self):
        return str(self.getPISCOVar('rough_ra_keyword'))

    def getRoughDecKeyword(self):
        return str(self.getPISCOVar('rough_dec_keyword'))

    def getAstrometryLowScale(self):
        return float(self.getPISCOVar('astrometry_solver_low_scale'))

    def getAstrometryHighScale(self):
        return float(self.getPISCOVar('astrometry_solver_high_scale'))

    def getAstrometryScaleUnits(self):
        return str(self.getPISCOVar('astrometry_scale_units') )

    def getParallelSmoothing(self):
        return float(self.getOSELOTSVar('parallel_smoothing', ))

    def getBackgroundStdBuffer(self):
        """
        Get the buffer, in pixels, from the edge to compute background stdev.
        """
        return int(self.getOSELOTSVar('background_buffer_for_measuring_std'))

    def __init__(self, spectrograph_file = 'OSELOTSDefaults.txt', defaults_dir = '/Users/sashabrownsberger/Documents/sashas_python_scripts/skySpectrograph/'):
        self.defaults_file = defaults_dir + spectrograph_file
        self.varContainer = bvc.bashVarContainerObject(container_file = spectrograph_file, container_dir = defaults_dir, readInFromFile = True, delimiter = '|')
        #print ('self.varContainer.var_dict = ' + str(self.varContainer.var_dict))
