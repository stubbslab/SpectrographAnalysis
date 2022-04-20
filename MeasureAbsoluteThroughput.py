import cantrips as can
import numpy as np
import matplotlib.pyplot as plt
import AstronomicalParameterArchive as apa

if __name__=="__main__":
    astro_arch = apa.AstronomicalParameterArchive()
    target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/ut20210929/'
    biases = ['Bias_2021_05_29_' + str(i) + '.fits' for i in range(14, 34)]
    laser_images = ['VLM635_f23p7_2021_05_29_' + str(i) + '.fits' for i in range(4, 13)]
    save_stacked_image_name = 'VLM635_f23p7_2021_05_29_MEDIAN.fits'
    sum_box = [[866, 882], [450, 704]]
    exp_time = 1.0

    collecting_diameter_cm = 4
    collecting_area_cmSqr = np.pi * (collecting_diameter_cm / 2.0) ** 2.0
    ref_wavelength_nm = 637.0
    int_sphere_diam_m = 1.0
    PD_QE_APerW = 0.3317
    PD_current_A = 0.760 * 10.0 ** -9.0
    PD_energy_W = PD_current_A / PD_QE_APerW

    PD_pupil_diam_m = 5 * 10.0 ** -3.0
    PD_tube_length_m = 2 * 2.54 * 10.0 ** -2.0
    PD_pupil_area = (PD_pupil_diam_m / 2.0) ** 2.0 * np.pi
    PD_solid_angle = 4.0 * np.pi * ((PD_pupil_diam_m / 2) ** 2.0 / ((PD_pupil_diam_m / 2) ** 2.0 + PD_tube_length_m ** 2.0))
    PD_integral_correction = 1.0

    sphere_ergs_steridian_per_mSqr = PD_energy_W * 10.0 ** 7.0 / (PD_pupil_area * PD_solid_angle * PD_integral_correction)

    #ster_to_asecSqr = astro_arch.getSteridanToSquareArcsec()
    print ('The energy angular flux, in ergs/m^2/steridian, is: ' + str(sphere_ergs_steridian_per_mSqr ))

    ref_line_wavelength_km = ref_wavelength_nm * 10.0 ** -9.0 * 10.0 ** -3.0
    watts_per_photon = astro_arch.getc() / ref_line_wavelength_km  * astro_arch.getPlancksConstant()
    ergs_per_photon = watts_per_photon * 10.0 ** 7.0

    ang_surf_flux_Ry = sphere_ergs_steridian_per_mSqr / ergs_per_photon * 10.0 ** (-10.0) * 4.0 * np.pi


    #energy_flux_to_rayleigh = astro_arch.getRayleighToSurfaceBrightnessAtLambda(ref_wavelength_nm)
    #sphere_brightness_Ry = ang_surf_flux_erg_cmSqr_arcsec_sqr * energy_flux_to_rayleigh
    print ('That is, ' + str(ang_surf_flux_Ry) + 'Ry')

    stacking_x_partitions = 2
    stacking_y_partitions = 2
    stacked_bias_data, stacked_bias_header = can.smartMedianFitsFiles(biases, target_dir, stacking_x_partitions, stacking_y_partitions, ref_index = 0, n_mosaic_image_extensions = 0, scalings = [1])

    laser_data_arrays = []
    for i in range(len(laser_images)):
        laser_image = laser_images[i]
        laser_data, laser_header = can.readInDataFromFitsFile(laser_image, target_dir)
        bias_sub_laser_data = laser_data - stacked_bias_data
        laser_data_arrays = laser_data_arrays + [bias_sub_laser_data]

    stacked_laser_header = laser_header
    stacked_laser_header['NSTACK'] = len(laser_images)
    stacked_laser_header['BIASSUB'] = 'TRUE'
    stacked_laser_data = np.median(laser_data_arrays, axis = 0)
    can.saveDataToFitsFile(np.transpose(stacked_laser_data), save_stacked_image_name, target_dir, header = stacked_laser_header)
    clipped_sum_region = stacked_laser_data[sum_box[1][0]:sum_box[1][1], sum_box[0][0]:sum_box[0][1]]
    plt.imshow(clipped_sum_region)
    plt.show()
    summed_ADU = np.sum(clipped_sum_region)
    ADU_rate=  summed_ADU / exp_time
    print ('ADU_rate = ' + str(ADU_rate))
    print ('So system throughput at ' + str(ref_wavelength_nm) + 'nm is: ' + str(ang_surf_flux_Ry / ADU_rate) + ' Ry/ADU')

    print ('Done. ')
