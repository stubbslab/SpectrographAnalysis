import cantrips as can
import numpy as np
import matplotlib.pyplot as plt
import AstronomicalParameterArchive as apa
import scipy
import os
import ProcessRawSpectrumClass as prsc

if __name__=="__main__":
    astro_arch = apa.AstronomicalParameterArchive()
    date_strs = ['2022', '04', '21']
    focus_str = '22p15'
    labelsize = 20
    ticksize = 18
    target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut' + ''.join(date_strs) + '/'
    save_throughput_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/'
    processor = prsc.SpectrumProcessor(target_dir)
    first_bias_indeces = can.flattenListOfLists([[25 + i * 3, 25 + i * 3 + 2] for i in range(5)]) + [40] + can.flattenListOfLists([[41 + i * 3, 41 + i * 3 + 2] for i in range(7)])

    wavelengths = list(range(570, 631, 5)) + list(range(640, 1081, 20))
    start_index = 62

    #Process bias data
    mono_bias_indeces = can.flattenListOfLists([[ start_index + i * 3 , start_index + i * 3 + 2] for i in range(len(wavelengths)) ])
    redo_bias = 0
    master_bias_image_file = 'BIAS.fits'
    master_bias_level_file = 'BIAS.txt'
    bias_list_file = 'Bias.list'
    biases = [ 'Bias_' + '_'.join(date_strs) + '_' + str(index) + '.fits' for index in first_bias_indeces + mono_bias_indeces]
    can.saveListsToColumns(biases, bias_list_file, target_dir)
    if not (os.path.isfile(target_dir + master_bias_image_file)) or redo_bias:
        processor.makeMasterBias(master_bias_image_file, master_bias_level_file, target_dir, bias_list = bias_list_file )

    #Process dark data
    master_dark_image_file = 'DARK.fits'
    redo_dark = 0
    dark_list_file = 'Dark.list'
    dark_indeces = [14, 15, 16, 17 ,18]
    darks = ['Dark_f' + focus_str  + '_' + '_'.join(date_strs) + '_' + str(index) + '.fits' for index in dark_indeces]
    can.saveListsToColumns(darks, dark_list_file, target_dir)
    if not (os.path.isfile(target_dir + master_dark_image_file)) or redo_dark:
        processor.makeMasterDark(master_dark_image_file, target_dir, master_bias_image_file, master_bias_level_file,)

    #Determine spectral solution and determine where the spectral components should approximately fall on detector
    KR1_indeces = [9, 10, 11, 12, 13]
    KR1_list_file = 'KR1.list'
    redo_spec_solution = 0
    KR1_imgs = ['KR1_f' + focus_str + '_' + '_'.join(date_strs) + '_' + str(i) + '.fits' for i in KR1_indeces]
    can.saveListsToColumns(KR1_imgs, KR1_list_file , target_dir )
    #ref_spec_solution_file = 'OSELOTSWavelengthSolution.txt'
    ref_spec_solution_file = processor.ref_spec_solution_file
    if not(os.path.isfile(target_dir + ref_spec_solution_file)):
        redo_spec_solution = 1
    processor.getWavelengthSolution(ref_spec_images_list = 'KR1.list', ref_spec_solution_file = ref_spec_solution_file, save_new_reference_spectrum = redo_spec_solution, ref_spec = None, show_fits = 0)
    if not(redo_spec_solution):
        processor.spec_range = processor.determineSpecRowRanges()
        print ('processor.spec_range = ' + str(processor.spec_range ))
        print ('processor.spec_range = ' + str(processor.bg_std_buffer))
    expected_pix_centers = [ int(processor.mu_of_wavelength_solution(wave)) for wave in wavelengths ]

    #Process spectral data
    redo_image_processing = 0
    mono_indeces = can.flattenListOfLists([[ start_index + i * 3 + 1] for i in range(len(wavelengths)) ])
    mono_images = [ 'Mono' + str(wavelengths[i]) + 'nm_f' + focus_str + '_' + '_'.join(date_strs) + '_' + str(mono_indeces[i]) + '.fits' for i in range(len(wavelengths)) ]
    processed_mono_images = ['proc_' + mono_image for mono_image in mono_images]
    #save_stacked_image_name = 'VLM635_f23p7_2021_05_29_MEDIAN.fits'
    #sum_box = [[866, 882], [450, 704]]
    #exp_time = 1.0
    for i in range(len(mono_images)):
        if not (os.path.isfile(target_dir + processed_mono_images[i])) or redo_image_processing:
            processor.processImages(spec_files_to_reduce = [mono_images[i]], do_bias = 1, do_dark = 1, crc_correct = 1, apply_background_correction = 0, redetermine_spec_range = 0, apply_scatter_correction = 0, save_stacked_image = 1, save_image_name = processed_mono_images[i] )

    #Count up the flux from the region of the image.
    box_width = 30
    height_padding = 5
    bg_box_width_ext = 20
    bg_box_height_ext = 30
    regions = [ [ [pix_center - box_width // 2, pix_center + box_width // 2], [int(processor.spec_range[0]) - height_padding, int(processor.spec_range[1]) + height_padding] ] for pix_center in expected_pix_centers]
    bg_regions = [ [[pix_center - box_width // 2 - bg_box_width_ext // 2, pix_center + box_width // 2 + bg_box_width_ext // 2],
                   [int(processor.spec_range[0]) - height_padding - bg_box_height_ext // 2, int(processor.spec_range[1]) + height_padding + bg_box_height_ext // 2 ]] for pix_center in expected_pix_centers]

    region_area = box_width
    bg_region_area = - region_area
    ADU_rates = []
    peak_cols = []
    ADU_rate_uncertainties = []
    for i in range(len(processed_mono_images)):
        region = regions[i]
        bg_region = bg_regions[i]
        image_data, header = can.readInDataFromFitsFile(processed_mono_images[i], target_dir)
        exp_time = float(header['EXPTIME'])
        gain =  header['GAIN']
        data_snip = image_data[region[1][0]:region[1][1], region[0][0]:region[0][1]]
        one_d_data_snip = np.sum(data_snip, axis = 0)
        max_snip_col = np.argmax(one_d_data_snip)
        try:
            init_guess = [max(one_d_data_snip), max_snip_col, box_width // 4, 0.0]
            fit_funct = lambda xs, A, mu, sig, shift: A * np.exp(- ((xs - mu) ** 2.0 / (2.0 * sig ** 2.0) )) + shift
            gauss_fit = scipy.optimize.curve_fit(fit_funct, np.array(range(len(one_d_data_snip))), one_d_data_snip, p0 = init_guess)
            box_peak = gauss_fit[0][1]
        except RuntimeError:
            box_peak = max_snip_col
        peak_cols = peak_cols + [box_peak + bg_region[0][0]]
        bg_snip = image_data[bg_region[1][0]:bg_region[1][1], bg_region[0][0]:bg_region[0][1]]
        counts_in_image = np.sum(data_snip)
        bg_counts_in_image = np.sum(bg_snip) - counts_in_image
        region_area = np.shape(data_snip)[0] * np.shape(data_snip)[1]
        bg_region_area = np.shape(bg_snip)[0] * np.shape(bg_snip)[1] - region_area
        ADU_rates = ADU_rates + [(counts_in_image - region_area * bg_counts_in_image / bg_region_area) / exp_time]
        ADU_rate_uncertainties = [ np.sqrt(np.abs(counts_in_image) + (region_area / bg_region_area) ** 2.0 * np.abs(bg_counts_in_image)) / np.sqrt(2) ]

    print ('peak_cols = ' + str(peak_cols))
    peak_wavelengths = [can.round_to_n(processor.wavelength_of_mu_solution(col), 6) for col in peak_cols]
    print ('peak_wavelengths = ' + str(peak_wavelengths))
    print ('Given wavelengths = ' + str(wavelengths))
    lin_fit = np.polyfit(wavelengths, peak_wavelengths, 1)
    #plt.scatter(wavelengths, peak_wavelengths - np.poly1d(lin_fit) (wavelengths))
    #plt.show()
    PD_curr_data_file = 'PD_1M_int_sphere_Data.txt'
    PD_QE_data_file = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/solarCell/qe/Hamamatsu_Photodiode_S2281_Spectral_Power_Response.txt'
    PD_QE_data = can.readInColumnsToList(PD_QE_data_file, delimiter = ' ', n_ignore = 1)
    PD_QE_wavelengths, PD_QE_A_per_W = [ [float( wave) for wave in PD_QE_data[0]], [float(qe) for qe in PD_QE_data[1]] ]
    PD_QE_interp = scipy.interpolate.interp1d(PD_QE_wavelengths, PD_QE_A_per_W)

    PD_curr_data_file = 'PD_1M_int_sphere_Data.txt'
    current_plot_arr_elem_size = 2
    PD_data = can.readInColumnsToList(target_dir + PD_curr_data_file, n_ignore = 1, delimiter = ' | ')
    ref_wavelengths_nm = np.array([float(elem) for elem in PD_data[0]])
    #ref_wavelengths_nm = np.array(peak_wavelengths)
    PD_dark_currs = [[[float(elem) for elem in sequence.split(',')] for sequence in PD_data[1]], [[float(elem) for elem in sequence.split(',')] for sequence in PD_data[3]]]
    PD_bright_currs = [[float(elem) for elem in sequence.split(',')] for sequence in PD_data[2]]
    n_PD_data_subfigs = int(np.ceil(np.sqrt(len(PD_bright_currs))))
    f, bright_axarr = plt.subplots(n_PD_data_subfigs, n_PD_data_subfigs, figsize = (current_plot_arr_elem_size * n_PD_data_subfigs, current_plot_arr_elem_size * n_PD_data_subfigs))

    for i in range(len(PD_bright_currs)):
        ax = bright_axarr[i // n_PD_data_subfigs, i % n_PD_data_subfigs]
        ax.plot(range(len(PD_dark_currs[0][i]) + 1, len(PD_dark_currs[0][i]) + 1 + len(PD_bright_currs[i]) ), PD_bright_currs[i], c = 'k')
        ax.text(0.1, 0.9, r'$\lambda = $' + str(ref_wavelengths_nm[i]) + 'nm', c = 'g', transform=ax.transAxes)
        ax.set_xlabel('N in Sequence')
        ax.set_ylabel('PD current (A)')
    n_PD_data_subfigs = int(np.ceil(np.sqrt(len(PD_bright_currs))))
    plt.savefig(target_dir + 'BrightCurrentDataStream.pdf')
    plt.close('all')
    f, dark_axarr = plt.subplots(n_PD_data_subfigs, n_PD_data_subfigs, figsize = (current_plot_arr_elem_size * n_PD_data_subfigs, current_plot_arr_elem_size * n_PD_data_subfigs))
    for i in range(len(PD_dark_currs[0])):
        ax = dark_axarr[i // n_PD_data_subfigs, i % n_PD_data_subfigs]
        ax.plot(range(1, len(PD_dark_currs[0][i]) + 1), PD_dark_currs[0][i], c = 'k')
        ax.plot(range(len(PD_dark_currs[0][i]) + 1 + len(PD_bright_currs[i]) , len(PD_dark_currs[0][i]) + 1 + len(PD_bright_currs[i])  + len(PD_dark_currs[1][i])), PD_dark_currs[1][i], c = 'k')
        ax.text(0.1, 0.9, r'$\lambda = $' + str(ref_wavelengths_nm[i]) + 'nm', c = 'g', transform=ax.transAxes)
        ax.set_xlabel('N in Sequence')
        ax.set_ylabel('PD current (A)')
    plt.savefig(target_dir + 'DarkCurrentDataStream.pdf')
    plt.close('all')
    PD_bright_means, PD_bright_uncertainties = [[ can.sigClipMean(curr_arr, sig_clip = 3) for curr_arr in PD_bright_currs], [ can.sigClipStd(curr_arr, sig_clip = 3, standard_error = 1) for curr_arr in PD_bright_currs] ]
    PD_dark_means, PD_dark_uncertainties = [ [[ can.sigClipMean(curr_arr, sig_clip = 3) for curr_arr in PD_dark_currs[0]], [ can.sigClipMean(curr_arr, sig_clip = 3) for curr_arr in PD_dark_currs[1]]],
                      [ [ can.sigClipStd(curr_arr, sig_clip = 3, standard_error = 1) for curr_arr in PD_dark_currs[0]], [ can.sigClipStd(curr_arr, sig_clip = 3, standard_error = 1) for curr_arr in PD_dark_currs[1]] ] ]
    dark_drift_uncertainty = can.sigClipStd(PD_dark_means[0] + PD_dark_means[1], sig_clip = 3, standard_error = 0)
    PD_currents_A = np.abs(np.array(PD_bright_means) - (np.array(PD_dark_means[0]) + np.array(PD_dark_means[1])) / 2)
    PD_currents_A_uncertainty = np.sqrt( np.array(PD_bright_uncertainties) ** 2.0 + np.array(dark_drift_uncertainty) ** 2.0 + (np.array(PD_dark_uncertainties[0]) ** 2.0 + np.array(PD_dark_uncertainties[1]) ** 2.0) / 2 )
    plt.scatter(ref_wavelengths_nm, PD_currents_A,  c = 'k')
    plt.errorbar(ref_wavelengths_nm, PD_currents_A,  yerr = PD_currents_A_uncertainty, ecolor = 'k', fmt = 'none')
    plt.savefig(target_dir + 'RefPDPhotocurrent.pdf')
    plt.close('all')

    #collecting_diameter_cm = 4
    #collecting_area_cmSqr = np.pi * (collecting_diameter_cm / 2.0) ** 2.0
    int_sphere_diam_m = 1.0
    PD_QE_APerW = np.array([ float(PD_QE_interp(wave)) for wave in ref_wavelengths_nm ])
    PD_energies_W, PD_energies_W_uncertainty = [arr / PD_QE_APerW for arr in [PD_currents_A, PD_currents_A_uncertainty]]

    PD_pupil_diam_m = 5 * 10.0 ** -3.0
    PD_tube_length_m = 2 * 2.54 * 10.0 ** -2.0
    PD_pupil_area = (PD_pupil_diam_m / 2.0) ** 2.0 * np.pi
    PD_solid_angle = 4.0 * np.pi * ((PD_pupil_diam_m / 2) ** 2.0 / ((PD_pupil_diam_m / 2) ** 2.0 + PD_tube_length_m ** 2.0))
    PD_integral_correction = 1.0

    sphere_ergs_steridian_per_mSqr, sphere_ergs_steridian_per_mSqr_uncertainty = [arr * 10.0 ** 7.0 / (PD_pupil_area * PD_solid_angle * PD_integral_correction) for arr in [PD_energies_W, PD_energies_W_uncertainty]]

    #ster_to_asecSqr = astro_arch.getSteridanToSquareArcsec()
    print ('The energy angular flux, in ergs/m^2/steridian, for the measured wavelengths was: ' )
    for i in range(len(ref_wavelengths_nm)):
        print (str(ref_wavelengths_nm[i]) + ' nm => ' + str(sphere_ergs_steridian_per_mSqr[i]) + ' +/- ' + str(sphere_ergs_steridian_per_mSqr_uncertainty [i]) )

    ref_line_wavelengths_km = ref_wavelengths_nm * 10.0 ** -9.0 * 10.0 ** -3.0
    watts_per_photon = astro_arch.getc() / ref_line_wavelengths_km  * astro_arch.getPlancksConstant()
    ergs_per_photon = watts_per_photon * 10.0 ** 7.0

    ang_surf_fluxes_Ry, ang_surf_fluxes_Ry_uncertainty = [ arr / ergs_per_photon * 10.0 ** (-10.0) * 4.0 * np.pi for arr in [sphere_ergs_steridian_per_mSqr, sphere_ergs_steridian_per_mSqr_uncertainty] ]


    #energy_flux_to_rayleigh = astro_arch.getRayleighToSurfaceBrightnessAtLambda(ref_wavelength_nm)
    #sphere_brightness_Ry = ang_surf_flux_erg_cmSqr_arcsec_sqr * energy_flux_to_rayleigh
    print ('That is, in Ry: ')
    for i in range(len(ref_wavelengths_nm)):
        print ( str(ref_wavelengths_nm[i]) + ' nm => ' + str(ang_surf_fluxes_Ry[i]) + ' +/- ' + str(ang_surf_fluxes_Ry_uncertainty[i]) )

    system_throughput_Ry_per_ADU = np.array(ang_surf_fluxes_Ry) / np.array(ADU_rates)
    system_throughput_Ry_per_ADU_uncertainty = np.sqrt((np.array(ang_surf_fluxes_Ry_uncertainty) / np.array(ADU_rates)) ** 2.0 + (np.array(ang_surf_fluxes_Ry) * ADU_rate_uncertainties / np.array(ADU_rates) ** 2.0) ** 2.0 )
    print ('ADU_rates = ' + str(ADU_rates))
    print ('So system throughputs are: ')
    for i in range(len(ref_wavelengths_nm)):
        print (str(ref_wavelengths_nm[i]) + ' nm => ' + str(system_throughput_Ry_per_ADU[i]) + ' +/- ' + str( system_throughput_Ry_per_ADU_uncertainty [i] ) + ' Ry/ADU')

    #plt.scatter(ref_wavelengths_nm, system_throughput_Ry_per_ADU, c = 'k')
    #plt.errorbar(ref_wavelengths_nm, system_throughput_Ry_per_ADU, yerr = system_throughput_Ry_per_ADU_uncertainty, c = 'k')
    #plt.show()
    good_points = [i for i in range(len(system_throughput_Ry_per_ADU)) if np.abs(system_throughput_Ry_per_ADU[i]) > system_throughput_Ry_per_ADU_uncertainty[i] and system_throughput_Ry_per_ADU[i] > 0]
    can.saveListsToColumns([ref_wavelengths_nm, system_throughput_Ry_per_ADU, system_throughput_Ry_per_ADU_uncertainty], 'OSELOTS_throughput_' + '_'.join(date_strs) + '.txt', save_throughput_dir, header = 'Wavelength (nm), Throughput (Ry/ADU), Throughput Uncertainty (Ry/ADU)', sep = ',')
    f, axarr = plt.subplots(1,1, figsize = [10, 6])
    axarr.scatter(ref_wavelengths_nm, 1.0 / system_throughput_Ry_per_ADU, c = 'k')
    axarr.errorbar(ref_wavelengths_nm, 1.0 / system_throughput_Ry_per_ADU, yerr = system_throughput_Ry_per_ADU_uncertainty / system_throughput_Ry_per_ADU ** 2.0, c = 'k')
    axarr.set_ylim([-np.max([1.0 / system_throughput_Ry_per_ADU[i] for i in good_points]) * 0.05, np.max([1.0 / system_throughput_Ry_per_ADU[i] for i in good_points]) * 1.05])
    #axarr.set_yscale('log')
    axarr.set_xlabel('Calibration wavelength (nm)', fontsize= labelsize)
    axarr.set_ylabel(r'Throughput (ADU Ry $^{-1}$ s $^{-1}$)', fontsize= labelsize)
    axarr.tick_params(axis='both', which='major', labelsize=ticksize)
    axarr.tick_params(axis='both', which='major', labelsize=ticksize)
    plt.savefig(save_throughput_dir + 'OSELOTS_throughput_' + '_'.join(date_strs) + '.pdf')
    print ('Done. ')
