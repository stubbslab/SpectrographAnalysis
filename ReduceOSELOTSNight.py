import matplotlib.pyplot as plt
import numpy as np
import cantrips as can
import sys
import SpectroscopyReferenceParamsObject as ref_param
import os
import ProcessRawSpectrumClass as prsc
import shutil

def getCalibrationDir(dir_root, image_type_print_string):
    cal_data_dir_suffix = bad_img = can.getUserInputWithDefault('Enter the observation night (formatted as YYYY_MM_DD) with the '  + image_type_print_string + ' data you want to use: ', '')
    cal_data_dir = dir_root + cal_data_dir_suffix + '/'
    while not(os.path.isdir(cal_data_dir)):
        print ('Directory ' + str(cal_data_dir) + ' does not exist. ')
        cal_data_dir = can.getUserInputWithDefault('Please enter the full path to the directory with the ' + image_type_print_string + ' data you want to use: ', dir_root)
    return cal_data_dir

def getImagesList(image_type_print_string, possible_images, indeces, good_indeces = 0):
    still_imgs_to_add = 1
    indeces_to_add = []
    selected_images = []
    selected_indeces = []
    while still_imgs_to_add:
        index = int(can.getUserInputWithDefault('Enter the index (trailing number) of the next ' + ('GOOD' if good_indeces else 'BAD') + ' ' + image_type_print_string + ' image.  Just hit [RETURN] when done: ', -1))
        if index < 0:
            still_imgs_to_add = 0
        else:
            indeces_to_add = indeces_to_add + [index]
    indeces_to_add = can.safeSortOneListByAnother(indeces_to_add, [indeces_to_add])[0]
    for index in indeces_to_add:
        if index in indeces:
            img = possible_images[indeces.index(index)]
            selected_images = selected_images + [img]
            selected_indeces = selected_indeces + [index]
        else:
            print ('Image index ' + str(index) + ' does not appear to match to a viable calibration image.  I will ignore that one.')
    return selected_images, selected_indeces

def trimImagesList(image_type_print_string, imgs, indeces, good_indeces = 0):
    bad_imgs_exist = 1
    bad_imgs = []
    while bad_imgs_exist:
        bad_img = int(can.getUserInputWithDefault('Enter the index (trailing number) of the next ' + ('GOOD' if good_indeces else 'BAD') + ' ' + 'image.  Just hit [RETURN] when done: ', -1))
        if bad_img < 0:
            bad_imgs_exist = 0
        else:
            bad_imgs = bad_imgs + [bad_img]
    for bad_img in bad_imgs:
        if bad_img in indeces:
            bad_spot = indeces.index(bad_img)
            print ('bad_spot = ' + str(bad_spot))
            imgs = can.removeListElement(imgs, bad_spot)
            indeces = can.removeListElement(indeces, bad_spot)
    return imgs, indeces

def getListOfImages(target_dir, dir_root, image_type_print_string, default_prefix, data_image_suffix, select_all_images = 0, check_target_dir = 0):
    prefix = can.getUserInputWithDefault('What prefix should I use to look for ' + image_type_print_string + ' images (default ' + default_prefix + '): ', default_prefix)
    if check_target_dir:
         use_default = can.getUserInputWithDefault('Should I use ' + image_type_print_string + ' calibration data from the same night? (y, Y, yes, Yes, YES, or 1 for "yes"; default 1): ', '1')
         use_default = (use_default in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
         if use_default:
             target_dir = target_dir
         else:
              target_dir = getCalibrationDir(dir_root, image_type_print_string)
    else:
        target_dir = target_dir
    all_files = os.listdir(target_dir)
    imgs = [i for i in all_files if prefix == i[0:len(prefix)] and data_image_suffix and (i.split('.')[0].split('_')[-1]).isdigit() ]
    indeces = [int(img.split('_')[-1][0:-len(data_image_suffix)]) for img in imgs]
    imgs, indeces = can.safeSortOneListByAnother(indeces, [imgs, indeces])
    print ('I found the following ' + image_type_print_string + (' possible' if select_all_images else '' ) + ' images: ')
    print (imgs)
    if select_all_images:
        print ('Because we typically want to use only a few of the ' + image_type_print_string + ' images, we will have the user specify the images they want.')
        imgs, indeces = getImagesList(image_type_print_string, imgs, indeces, good_indeces = 1)
    else:
        bad_imgs_exist = can.getUserInputWithDefault('Should I ignore any of these ' + image_type_print_string + ' images in the analysis? (y, Y, yes, Yes, YES, or 1 for "yes"; default 0): ', '0')
        bad_imgs_exist = (bad_imgs_exist in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
        if bad_imgs_exist:
            imgs, indeces = trimImagesList(image_type_print_string, imgs, indeces)
            print ('Final list of ' + data_image_suffix + ' images is: ')
            print (imgs)

    return prefix, imgs, indeces, target_dir



if __name__ == "__main__":
    """
    Processes a night of data taken with OSELOTS.  User should pass:
        a date string, formatted as YYYY_MM_DD
    """
    sys_args = sys.argv[1:]
    date_str = sys_args[0]

    #NEW USER: UPDATE THIS VARIABLE!!!!
    root_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/'
    data_subdir = 'data/'
    data_dir = root_dir + data_subdir

    if len(date_str) != 10:
        print ('Did not get properly formatted date for the night to analyze.  I expect a date formatted like: YYYY_MM_DD, passed as 1st argument.')
        sys.exit()

    ref_param_holder = ref_param.CommandHolder(spectrograph_file = 'OSELOTSDefaults.txt', defaults_dir = './')
    ref_spec_solution_file = ref_param_holder.getRefSpecSolutionFile()


    data_image_suffix = ref_param_holder.getImageSuffix() # .fits
    master_bias_list = ref_param_holder.getBiasList()
    master_dark_list = ref_param_holder.getDarkList()

    target_dir = data_dir + date_str + '/'
    get_new_target_dir = can.getUserInputWithDefault('I am going to analyze data in this directory: ' + target_dir + '  Is that okay (y, Y, yes, Yes, YES, or 1 for "yes"; default 1): ', '1')
    if not(get_new_target_dir in ['y', 'Y', 'yes', 'Yes', 'YES', '1']):
        target_dir = input('Please enter the FULL PATH to the directory with the night of data you would like to analyze: ' )
    if not(os.path.isdir(target_dir)):
        print ('Directory ' + str(target_dir) + ' does not exist.  Please double check the path and try again. ')
        sys.exit()

    overwrite = can.getUserInputWithDefault('Should I redo and overwrite previous calibration calculations (Master bias, master dark, wavelength solution, etc) (y, Y, yes, Yes, YES, or 1 for "yes"; default 0): ', '0')
    overwrite = (overwrite in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
    if overwrite:
        print ('We WILL overwrite.')
    else:
        print ('We will NOT overwrite.')

    #bias_from_tonight = can.getUserInputWithDefault('Should I do determ (Master bias, master dark, wavelength solution, etc) (y, Y, yes, Yes, YES, or 1 for "yes"; default 0): ', '0')
    bias_prefix, bias_imgs, bias_indeces, bias_dir = getListOfImages(target_dir, data_dir, 'Bias', 'Bias', data_image_suffix, check_target_dir = 1)
    print ('bias_dir = ' + str(bias_dir))
    if overwrite or not(os.path.exists(target_dir + master_bias_list)):
        can.saveListsToColumns(bias_imgs, master_bias_list, target_dir)


    dark_prefix, dark_imgs, dark_indeces, dark_dir = getListOfImages(target_dir, data_dir, 'Dark', 'Dark', data_image_suffix, check_target_dir = 1)
    print ('dark_dir = ' + str(dark_dir)) # 2022_05_25
    if overwrite or not(os.path.exists(target_dir + master_dark_list)):
        can.saveListsToColumns(dark_imgs, master_dark_list, target_dir)

    do_wavelength_from_this_night = can.getUserInputWithDefault('Should we use a wavelength solution determined on THIS night? (y, Y, yes, Yes, YES, or 1 for "yes"; default 1): ', '1')
    do_wavelength_from_this_night = (do_wavelength_from_this_night in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
    if do_wavelength_from_this_night:
        arc_lamp_prefix, arc_lamp_imgs, arc_lamp_indeces, arc_lamp_dir = getListOfImages(target_dir, data_dir, 'spectral calibration', 'HG2', data_image_suffix, select_all_images = 1, check_target_dir = 0)
        master_arclamp_list = arc_lamp_prefix + ref_param_holder.getListSuffix()
        if overwrite or not(os.path.exists(target_dir + master_arclamp_list)):
             can.saveListsToColumns(arc_lamp_imgs, master_arclamp_list, target_dir)
    else:
        arc_lamp_prefix = None
        wavelength_night = can.getUserInputWithDefault('Enter the date string for the night from which we should correct the data (formatted as YYYY_MM_DD; default ' + date_str + '): ', 'date_str')
        wavelength_correction_dir = data_dir + wavelength_night + '/'
        shutil.copy2(wavelength_correction_dir + ref_spec_solution_file, target_dir)

    sky_prefix, sky_imgs, sky_indeces, sky_dir = getListOfImages(target_dir, data_dir, 'Sky', 'Sky', data_image_suffix, check_target_dir = 0)
    dark_sky_start_index = int(can.getUserInputWithDefault('Enter first sky image number when sky was dark (used to identify sky lines, over sun continuum); default ' + str(sky_indeces[0]) + '): ', str(sky_indeces[0])))
    dark_sky_end_index = int(can.getUserInputWithDefault('Enter last sky image number when sky was dark (used to identify sky lines, over sun continuum); default ' + str(sky_indeces[-1]) + '): ', str(sky_indeces[-1])))
    focus_positions = [float(can.readInDataFromFitsFile(sky_img, target_dir)[1][ref_param_holder.getFocusKeyword()]) for sky_img in sky_imgs]
    unique_focus_positions = np.unique(focus_positions)
    if len(focus_positions) > 0:
        print ('I have detected images with the following focus positions: ' + str(unique_focus_positions))
        process_by_focus = can.getUserInputWithDefault('Should I process different images with different focus positions as their own sets? (y, Y, yes, Yes, YES, or 1 for "yes"; default 1): ', '1')
    else:
        process_by_focus = '0'
    process_by_focus = (process_by_focus in ['y', 'Y', 'yes', 'Yes', 'YES', '1'])
    if process_by_focus:
        print ('focus_postions = ' + str(focus_positions))
        print ('unique_focus_positions = ' + str(unique_focus_positions))
        sky_imgs_set, sky_indeces_set = [ [[arr[i] for i in range(len(focus_positions)) if focus_positions[i] == unique_focus_positions[j]] for j in range(len(unique_focus_positions))] for arr in [sky_imgs, sky_indeces]]
    else:
        sky_imgs_set, sky_indeces_set = [[sky_imgs], [sky_indeces]]

    dark_sky_imgs_set = [[sky_imgs_set[j][i] for i in range(len(sky_imgs_set[j])) if (sky_indeces[i] >= dark_sky_start_index and sky_indeces[i] <= dark_sky_end_index)] for j in range(len(sky_imgs_set))]
    dark_sky_indeces_set = [[sky_indeces_set[j][i] for i in range(len(sky_indeces_set[j])) if (sky_indeces[i] >= dark_sky_start_index and sky_indeces[i] <= dark_sky_end_index)] for j in range(len(sky_imgs_set))]



    processor = prsc.SpectrumProcessor(target_dir, show_fits = 0, date = date_str.split('_'), ref_spec = arc_lamp_prefix, redo_master_bias = overwrite, redo_master_dark = overwrite, bias_dir = bias_dir, dark_dir = dark_dir )

    redo_spectrum = overwrite
    if (redo_spectrum and len(arc_lamp_imgs) > 0) or not(os.path.isfile(target_dir + ref_spec_solution_file)):
        processor.getWavelengthSolution(ref_spec_images_list = master_arclamp_list, ref_spec_solution_file = ref_spec_solution_file, save_new_reference_spectrum = 1, ref_spec = None, show_fits = 0)
        #As a check, process the reference wavelength images:
        processor.pullCombinedSpectrumFromImages(arc_lamp_imgs, show_fits = 0, analyze_spec_of_ind_images = 1, line_dict_id = None, plot_title = 'Stacked ' + arc_lamp_prefix + ' Spectrum', save_intermediate_images = 0, stacked_image_name = 'Stacked' + arc_lamp_prefix + 'Image_img' + str(arc_lamp_indeces[0]) + 'To' + str(arc_lamp_indeces[-1]), apply_background_correction = 0, apply_scatter_correction = 0)
        for i in range(len(arc_lamp_imgs)):
            img = arc_lamp_imgs[i]
            img_num = arc_lamp_indeces[i]
            processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)
        print ('arclamp idenfitied wavelengths: ')
        for line in processor.identified_lines_dict['FULL']['LINES'].keys():
            line_pix = processor.identified_lines_dict['FULL']['LINES'][line][1]
            print ('line_pix = ' + str(line_pix))
            print ('processor.wavelength_of_mu_solution(line_pix) = ' + str(processor.wavelength_of_mu_solution(line_pix)))
            wave = float(processor.wavelength_of_mu_solution(line_pix))
            print ('wave = ' + str(wave))
            print (str(can.round_to_n(wave, 4)) + ' nm')
    #Now we should reinitiate the processor so that we don't try to match reference and sky lines.  Do this once for each of the different sets of images.
    for j in range(len(sky_imgs_set)):
        if process_by_focus:
            focus_position = focus_positions[j]
            print ('Processing images with focus position ' + str(focus_position))
            extra_save_str = '_' + 'p'.join(str(float(focus_position)).split('.'))
        else:
            extra_save_str = ''
        sky_imgs = sky_imgs_set[j]
        dark_sky_imgs = dark_sky_imgs_set[j]
        dark_sky_indeces = dark_sky_indeces_set[j]
        sky_indeces = sky_indeces_set[j]
        processor = prsc.SpectrumProcessor(target_dir, show_fits = 0, date = date_str.split('_'), ref_spec = arc_lamp_prefix, dark_dir = dark_dir, bias_dir = bias_dir)

        processor.pullCombinedSpectrumFromImages(dark_sky_imgs, analyze_spec_of_ind_images = 0, line_dict_id = None, plot_title = 'Stacked Spectrum', save_intermediate_images = 0, stacked_image_name = 'StackedSkyImage_img' + str(dark_sky_indeces[0]) + 'To' + str(dark_sky_indeces[-1]), apply_scatter_correction = 0, )
        for i in range(len(sky_imgs)):
            print ('!!! ' + str(i) + ' !!!')
            img = sky_imgs[i]
            img_num = sky_indeces[i]
            processor.measureStrengthOfLinesInImage(img, show_fits = 0, line_dict_id = img_num, redetermine_spec_range = 0)

        processor_python_obj_save_file = 'FullNight_' + date_str + extra_save_str + '.prsc'
        processor.saveSpecProcessor(processor_python_obj_save_file, save_dir = None, )
        print ("processor_reloaded.loadSpecProcessor(processor_python_obj_save_file, load_dir = None)")
        processor.plotScaledLineProfilesInTime(line_variation_image_name = 'skyLineChangesOverTimeScaled' + extra_save_str + '.pdf')
        processor.plotLineProfilesInTime(line_variation_image_name = 'skyLineChangesOverTime' + extra_save_str + '.pdf')
        print ('You can reload the saved spectrum processor using the following (in the Python environment): ')
        print ('import ProcessRawSpectrumClass as prsc')
        print ("date_str = '" + str(date_str) + "'")
        print ("target_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/data/' + date_str + '/' " )
        print ('processor_reloaded = prsc.SpectrumProcessor(target_dir, show_fits = 0)')
        print ("processor_python_obj_save_file = '" +   str(processor_python_obj_save_file) + "'" )

    #arc_lamp_images = getListOfRefSpectrumImages()
    print ('Done.')
