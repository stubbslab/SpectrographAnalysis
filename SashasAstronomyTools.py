import cantrips as can
import astropy.stats as astrostats
from astropy.io import fits
from photutils import DAOStarFinder
import numpy as np
from photutils import find_peaks
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.coordinates import Angle
from astropy import units as u
from astropy import wcs
from datetime import datetime

def divideSkyCircleIntoNRings(n_sky_rings, maximum_angle_rad, n_start_angles = 16, n_angle_windings = 1):
    sky_ring_radii = np.linspace(0.0, maximum_angle_rad, n_sky_rings+1)[1:]
    sky_angle_starts = np.linspace(0.0, 2.0 * np.pi, n_sky_rings) * n_angle_windings
    sky_angle_starts = [0.0 for ring in range(n_sky_rings)]
    for i in range(1, len(sky_angle_starts)):
        sky_angle_starts[i] = (sky_angle_starts[i-1] + np.pi + (i % 2) * 2.0 * np.pi / n_start_angles) % (2.0 * np.pi)
    inner_solid_angle = 4.0 * np.pi * np.sin(sky_ring_radii[0]) ** 2.0
    solid_angle_per_pix = inner_solid_angle
    n_pixels_per_radius = [1 for rad in sky_ring_radii]
    solid_angles_per_radius = [solid_angle_per_pix for rad in sky_ring_radii]
    prev_solid_angle = inner_solid_angle
    for i in range(1, len(sky_ring_radii)):
        sky_ring_radius = sky_ring_radii[i]
        current_solid_angle = 4.0 * np.pi * np.sin(sky_ring_radius) ** 2.0
        diff_solid_angle = current_solid_angle - prev_solid_angle
        prev_solid_angle = current_solid_angle
        n_pixels = int(round(diff_solid_angle / solid_angle_per_pix))
        n_pixels_per_radius[i] = n_pixels
        solid_angles_per_radius[i] = diff_solid_angle

    #print ('[sky_ring_radii, n_pixels_per_radius, solid_angles_per_radius] = ' + str([sky_ring_radii, n_pixels_per_radius, solid_angles_per_radius]))

    sky_pixel_centers = [[[0.0, 0.0] for pix in range(n_pix)] for n_pix in n_pixels_per_radius]
    sky_pixel_solid_angles = [[solid_angles_per_radius[i] / n_pixels_per_radius[i] for pix in range(n_pixels_per_radius[i])] for i in range(len(n_pixels_per_radius))]

    prev_circle_radius = sky_ring_radii[0]
    sky_pixel_centers[0] = [[sky_ring_radii[0] / 2.0, 0.0]]
    for i in range(1, len(sky_ring_radii)):
        current_circle_radius = sky_ring_radii[i]
        average_radius = (current_circle_radius + prev_circle_radius) / 2.0
        prev_circle_radius = current_circle_radius
        n_pixels = n_pixels_per_radius[i]
        angles_around_circle = np.linspace(sky_angle_starts[i], 2.0 * np.pi + sky_angle_starts[i], n_pixels+1)[0:-1] % (np.pi * 2.0)
        sky_pixel_centers[i] = [[average_radius, angle_around_circle] for angle_around_circle in angles_around_circle]

    sky_pixel_centers = can.flattenListOfLists(sky_pixel_centers)
    sky_pixel_solid_angles = can.flattenListOfLists(sky_pixel_solid_angles)

    return [sky_pixel_centers, sky_pixel_solid_angles]



#Break a circular region of the sky into pixels of roughly equal area
def computeSkyPixels(target_n_sky_pixels, maximum_angle_rad, n_angle_windings = 1, n_start_angles = 16):
    n_rings = -1
    prev_n_sky_pixels = 0.0
    test_n_rings = 0
    while n_rings < 0:
        test_n_rings = test_n_rings + 1
        test_n_pixels = len(divideSkyCircleIntoNRings(test_n_rings, maximum_angle_rad)[0])
        if test_n_pixels >= target_n_sky_pixels:
            n_rings = test_n_rings
        else:
            prev_n_sky_pixels = test_n_pixels
    if abs(prev_n_sky_pixels - target_n_sky_pixels) < abs(test_n_pixels - target_n_sky_pixels):
        best_n_rings = n_rings - 1
    else:
        best_n_rings = n_rings
    best_sky_pixel_centers, best_sky_pixel_solid_angles = divideSkyCircleIntoNRings(best_n_rings, maximum_angle_rad, n_start_angles = n_start_angles, n_angle_windings = n_angle_windings)
    return [best_sky_pixel_centers, best_sky_pixel_solid_angles]

def getRADecFromPixelAndWCS(header, x_y_positions):
    wcs_object = wcs.WCS(header)
    RA_Dec_positions = wcs_object.wcs_pix2world(x_y_positions, 1)

    return RA_Dec_positions

def getPixelFromWCSHeader(header, ra_dec, ra_dec_in_deg = 1):
    wcs_object = wcs.WCS(header)
    #If ra_dec are given in sexegesimal, then we need to convert
    if ra_dec_in_deg:
        ra_dec_in_deg = ra_dec
    else:
        ra, dec = [Angle(ra_dec[0]), Angle(ra_dec[1])]
        ra_dec_in_deg = [ra.degree, dec.degree]
    RA_Dec_positions = wcs_object.wcs_world2pix(*ra_dec_in_deg, 1)
    RA_Dec_positions = [float(RA_Dec_positions[0]), float(RA_Dec_positions[1])]

    return RA_Dec_positions

def getPixelFromWCSFitsHeader(header):
    wcs_object = wcs.WCS(header)
    RA_Dec_positions = wcs_object.wcs_pix2world(x_y_positions, 1)

def getDegreeFromSkyCoordStr(coord_str, hour_angle = 0):
    elements = [float(elem) for elem in coord_str.split(':')]
    decimal_rep = abs(elements[0]) + (elements[1] + elements[2] / 60.0) / 60.0
    if coord_str[0] == '-':
        decimal_rep = decimal_rep * -1
    if hour_angle:
        decimal_rep = decimal_rep * 360.0 / 24.0
    return decimal_rep


def fwhmToSig(fwhm):
    sig = fwhm / (np.sqrt(np.log(2.0) * 2.0) * 2.0)
    return sig

def sigToFwhm(sig):
    fwhm = sig * (np.sqrt(np.log(2.0) * 2.0) * 2.0)
    return fwhm

#Do the 1d measuring, but do it until some convergence
def measure1dPSFOfStar(init_x, init_y, img_file, target_dir, init_fwhm_guess_arcsec,
                                   fit_radius_scaling = 5.0, bg_radii_scalings = [6.0, 7.0],
                                   init_fit_guess = None, radial_fit_funct = lambda rsqr, A, sig: A * np.exp(- rsqr / (2.0 * sig ** 2.0)),
                                   show_fit = 1, max_iters = 5, fwhm_convergence_arcsec = 0.01, pixel_scaling = 0.1,
                                   max_search_rad_in_pix = 100, min_n_iters_before_completion = 2, verbose = 0):

    init_fwhm_guess = init_fwhm_guess_arcsec / pixel_scaling
    if verbose: print ('init_fwhm_guess in pixels = ' + str(init_fwhm_guess))
    img, header = can.readInDataFromFitsFile(img_file, target_dir)
    init_sig_guess = fwhmToSig(init_fwhm_guess)
    current_fit_params =[init_x, init_y, np.nan, init_sig_guess]
    current_fit_guess = init_fit_guess
    convergence_satisfied = 0
    n_iters = 0
    fit_params_sequence = []

    #No convergence criterion yet defined
    while not(convergence_satisfied):
        try:
            fit_params_sequence = fit_params_sequence + [current_fit_params[:]]
            current_x, current_y, current_height, current_sig = current_fit_params
            if verbose: print ('current_fit_params = ' + str(current_fit_params))
            if current_sig * bg_radii_scalings[1] > max_search_rad_in_pix: current_sig = max_search_rad_in_pix // bg_radii_scalings [1]

            current_fit = measure1DPSFOfStarFixedRadius(current_x, current_y, img, round(fit_radius_scaling * current_sig), [round(scaling * current_sig) for scaling in bg_radii_scalings],
                                                        fit_guess = current_fit_guess, radial_fit_funct = radial_fit_funct,
                                                        show_fit = show_fit, verbose = verbose,
                                                        display_title = 'Centroiding at ' + str([c.round_to_n(init_x, 5), can.round_to_n(init_y, 5)]) + ' in file ' + img_file)
            current_fit_params = current_fit[0:-1]
            current_fit_minimized_val = current_fit[-1]
            current_fit_guess = [0.0, 0.0, current_fit_params[2], current_fit_params[3]]
            n_iters = n_iters + 1
            if n_iters > max(min_n_iters_before_completion-1, 1):
                if sigToFwhm(abs(fit_params_sequence[-1][0] - fit_params_sequence[-2][0])) <= fwhm_convergence_arcsec:
                    if verbose: print('Converged after ' + str(n_iters) + ' iterations. ')
                    break
            if max_iters < n_iters:
                print ('Maximum iterations reached.  Returning current fit.')
                break
        except ValueError as e:
            print ('Fitting to that star failed with ValueError: "' + str(e) + '".  Returning "None"')
            return None
    return current_fit_params


def measure1DPSFOfStarFixedRadius(x, y, img, star_fit_radius, bg_radii,
                                  fit_guess = None, radial_fit_funct = lambda rsqr, A, sig: A * np.exp(- rsqr / (2.0 * sig ** 2.0)) ,
                                  show_fit = 0, pixel_scale = 0.1, verbose = 0, display_title = '', title_size = 10):
    max_y, max_x = np.shape(img)
    subimage_section = [[max(int(round(y - bg_radii[1])), 0), min(int(round(y + bg_radii[1])) + 1, max_y)],
                        [max(int(round(x - bg_radii[1])), 0), min(int(round(x + bg_radii[1])) + 1, max_x)]]

    subimage = img[subimage_section[0][0]:subimage_section[0][1] , subimage_section[1][0]:subimage_section[1][1]]
    ys, xs = [np.arange(subimage_section[0][0], subimage_section[0][1]) - y, np.arange(subimage_section[1][0], subimage_section[1][1] ) - x]
    #print ('[x, y] = ' + str())
    #if verbose: print ('[xs, ys] = ' + str([xs, ys]))
    #y_mesh, x_mesh = np.meshgrid(ys, xs)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    #print ('[np.shape(x_mesh), np.shape(subimage)] = ' + str([np.shape(x_mesh), np.shape(subimage)]))
    r_mesh = np.sqrt(x_mesh ** 2.0 + y_mesh ** 2.0)
    vals = can.flattenListOfLists(subimage)
    flat_xs = can.flattenListOfLists(x_mesh)
    flat_ys = can.flattenListOfLists(y_mesh)
    rs =  can.flattenListOfLists(r_mesh)
    #print ('[np.argmin(rs), np.argmax(vals)] = ' + str([np.argmin(rs), np.argmax(vals)] ))


    star_vals = [vals[i] for i in range(len(vals)) if rs[i] <= star_fit_radius]
    star_xs = np.array([flat_xs[i] for i in range(len(rs)) if rs[i] <= star_fit_radius])
    star_ys = np.array([flat_ys[i] for i in range(len(rs)) if rs[i] <= star_fit_radius])
    star_rs = [rs[i] for i in range(len(rs)) if rs[i] <= star_fit_radius]
    mean_star = np.mean(star_vals)
    bg_vals = [vals[i] for i in range(len(vals)) if rs[i] > bg_radii[0] and rs[i] <= bg_radii[1]]
    med_bg = np.median(bg_vals)
    bg_sub_star_vals = [val - med_bg for val in star_vals]

    if fit_guess is None:
        #fit_guess = [np.mean(xs), np.mean(ys), np.max(bg_sub_star_vals), star_fit_radius / 2]
        fit_guess = [0.0, 0.0, np.max(bg_sub_star_vals), star_fit_radius / 2]

    guess_rsqrs = (np.array(star_xs) - fit_guess[0]) ** 2.0 + (np.array(star_ys) - fit_guess[1]) ** 2.0
    #print ('[star_xs, star_ys] = ' + str([star_xs, star_ys] ))
    loaded_funct = lambda args: np.sum((radial_fit_funct((star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0, args[2] * fit_guess[2], args[3] * fit_guess[3]) - bg_sub_star_vals) ** 2.0)  #args = [x_mean, y_mean, amplitude, width]
    #
    loaded_funct = lambda args, star_xs = star_xs, star_ys = star_ys, fit_guess = fit_guess: sumOfSqrsFunct(args, star_xs, star_ys, fit_guess, bg_sub_star_vals, radial_fit_funct)

    best_fit = optimize.minimize(loaded_funct, [fit_guess[0], fit_guess[1], 1, 1], bounds = [(-star_fit_radius,star_fit_radius), (-star_fit_radius, star_fit_radius), (0.0, np.inf), (pixel_scale, np.inf) ])

    best_fit_params = best_fit['x']
    if verbose: print ('best_fit_params are ' + str(best_fit_params))
    best_fit_minimized_val = best_fit['fun']
    best_fit_rsqrs = (star_xs - best_fit_params[0]) ** 2.0 + (star_ys - best_fit_params[1]) ** 2.0
    fitting_rsqrs = np.linspace(min(best_fit_rsqrs), max(best_fit_rsqrs), 101)
    if show_fit:
        #print ('np.shape(r_mesh)= '+ str(np.shape(r_mesh)))
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(subimage, norm = LogNorm() )
        new_scat = axarr[1].scatter(np.sqrt(best_fit_rsqrs), bg_sub_star_vals, marker = 'x', c = 'r')
        guess_scat = axarr[1].scatter(np.sqrt(guess_rsqrs), bg_sub_star_vals, marker = 'x', c = 'k')
        best_fit_curve, = axarr[1].plot(np.linspace(0.0, np.sqrt(np.max(best_fit_rsqrs)) , 101),
                                       radial_fit_funct(np.linspace(0.0, np.sqrt(np.max(best_fit_rsqrs)) , 101)** 2.0, best_fit_params[2] * fit_guess[2], best_fit_params[3] * fit_guess[3]) )
        axarr[1].legend([guess_scat, new_scat, best_fit_curve], ['Data with old centroid', 'Data with new centroid', 'Best-fit of new centroid'])
        plt.suptitle(display_title, fontsize = title_size )
        plt.draw()
        plt.pause(0.5)
        plt.close()

    best_fit_x, best_fit_y, best_fit_height, best_fit_sigma = [x + best_fit_params[0], y + best_fit_params[1], best_fit_params[2] * fit_guess[2], best_fit_params[3] * fit_guess[3]]

    if verbose: print ('best_fit params are: *** ' + str([best_fit_x, best_fit_y, best_fit_height, best_fit_sigma]) + ' *** and give a sum_of_sqrs of ' + str(best_fit_minimized_val))
    return [best_fit_x, best_fit_y, best_fit_height, best_fit_sigma, best_fit_minimized_val]


def sumOfSqrsFunct(args, star_xs, star_ys, fit_guess, bg_sub_star_vals, radial_fit_funct):
    #print ('(star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0 = ' + str((star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0) )
    #print ('radial_fit_funct((star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0, args[2] * fit_guess[2], args[3] * fit_guess[3]) = ' + str(radial_fit_funct((star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0, args[2] * fit_guess[2], args[3] * fit_guess[3])))
    #print ('bg_sub_star_vals = ' + str(bg_sub_star_vals))
    radial_fit_vals = radial_fit_funct((star_xs - args[0]) ** 2.0 + (star_ys - args[1]) ** 2.0, args[2] * fit_guess[2], args[3] * fit_guess[3])
    #f, axarr = plt.subplots(1,2)
    #axarr[0].scatter(radial_fit_vals, bg_sub_star_vals)
    #axarr[1].scatter(radial_fit_vals, (radial_fit_vals - bg_sub_star_vals) ** 2.0)
    sum_of_sqrs = np.sum((radial_fit_vals - bg_sub_star_vals) ** 2.0)
    #print ('np.sum(bg_sub_star_vals) = ' + str(np.sum(bg_sub_star_vals)))
    #axarr[0].text(np.mean([max(radial_fit_vals), min(radial_fit_vals)]), np.mean([max(bg_sub_star_vals), min(bg_sub_star_vals)]), 'Sum of squares = ' + str(sum_of_sqrs))
    #plt.show( )
    #print('for minimization params, ' + str(args) + ', sum_of_sqrs = ' + str(sum_of_sqrs))
    return (sum_of_sqrs)

#Returns the median seeing of stars in a specified image
def getSeeingInImage(img_file, target_dir, kernel_in_arcsec,
                     pixel_scale = 0.1, sig_clipping_for_stats = 3.0, star_find_n_sig_threshold = 4.0,
                     fit_radius_scaling = 3.0, bg_radii_scalings = [5.0, 6.0], verbose = 0,
                     init_fit_guess = None, radial_fit_funct = lambda rsqr, A, sig: A * np.exp(- rsqr / (2.0 * sig ** 2.0)),
                     show_fit = 1, max_iters = 5, min_n_iters_before_completion = 2,
                     good_pix_thresh = 60000, desired_n_stars = 10, fwhm_convergence_arcsec = 0.01):
    kernel_in_pix = kernel_in_arcsec / pixel_scale
    star_stats = getDAOStarFinderStats(img_file, target_dir, kernel_in_pix,
                                     sig_clipping_for_stats = sig_clipping_for_stats, star_find_n_sig_threshold = star_find_n_sig_threshold ,
                                     fit_radius_scaling = fit_radius_scaling, bg_radii_scalings = bg_radii_scalings)
    xs = star_stats[0]
    ys = star_stats[1]
    fluxes = star_stats[2]
    peaks = star_stats[3]
    good_stars = [1 if peak < good_pix_thresh else 0 for peak in peaks ]
    xs = [xs[i] for i in range(len(xs)) if good_stars[i]]
    ys = [ys[i] for i in range(len(ys)) if good_stars[i]]
    fluxes = [ fluxes[i] for i in range(len(fluxes)) if good_stars[i] ]
    peaks = [ peaks[i] for i in range(len(peaks)) if good_stars[i] ]

    print ('Found ' + str(len(xs)) + ' stars.   Picking ' + str(desired_n_stars) + ' brightest stars with no pixels above ' + str(good_pix_thresh) + 'ADU...')

    xs, ys, fluxes, peaks = can.safeSortOneListByAnother(fluxes, [xs, ys, fluxes, peaks])
    xs.reverse()
    ys.reverse()
    fluxes.reverse()
    peaks.reverse()
    good_positions = [[xs[i], ys[i]] for i in range(len(xs))]
    positions = np.transpose((xs, ys))

    stellar_fits = [[] for star in range(desired_n_stars)]

    n_measured_stars = 0
    n_failed_stars = 0
    current_star_index = 0
    while n_measured_stars < desired_n_stars:
        pos = good_positions[current_star_index]
        if verbose: print ('Working on star at position [x,y] = ' + str(pos))
        stellar_fit = measure1dPSFOfStar(*pos, img_file, target_dir, kernel_in_arcsec,
                                         fit_radius_scaling = fit_radius_scaling, bg_radii_scalings = bg_radii_scalings,
                                         init_fit_guess = init_fit_guess, radial_fit_funct =radial_fit_funct, min_n_iters_before_completion = min_n_iters_before_completion,
                                         show_fit = show_fit, max_iters = max_iters, fwhm_convergence_arcsec = fwhm_convergence_arcsec  )
        if stellar_fit is None:
            #Failed to fit the star well.
            n_failed_stars = n_failed_stars + 1
        else:
            stellar_fits[n_measured_stars] =  stellar_fit
            n_measured_stars = n_measured_stars + 1
        current_star_index = current_star_index + 1

    sigs = [fit[3] for fit in stellar_fits]
    fwhms = [sigToFwhm(sig) for sig in sigs]

    return fwhms

def getDAOStarFinderStats(img_file, target_dir, kernel_in_pix,
                          sig_clipping_for_stats = 3.0, star_find_n_sig_threshold = 4.0,
                          fit_radius_scaling = 5.0, bg_radii_scalings = [7.0, 8.0]):

    data, header = can.readInDataFromFitsFile(img_file, target_dir)
    mean, median, std = astrostats.sigma_clipped_stats(data, sigma=sig_clipping_for_stats)
    print ('Finding stars in image: ' + target_dir + img_file)
    daofind = DAOStarFinder(fwhm = kernel_in_pix, threshold = star_find_n_sig_threshold * std)
    sources = daofind(data - median)
    results = {'xcentroid':sources['xcentroid'].data, 'ycentroid':sources['ycentroid'].data, 'sharpness':sources['sharpness'].data,
               'roundness1':sources['roundness1'].data, 'roundness2':sources['roundness2'].data, 'npix':sources['npix'].data, 'sky':sources['sky'].data,
               'peak':sources['peak'].data, 'flux':sources['flux'].data, 'mag':sources['mag'].data}

    return results

def measureStatisticsOfFitsImages(img_list, data_dir = None, n_mosaic_image_extensions = 0,
                                    #img_indeces_to_id = None, img_ids = None,
                                    time_header_key = 'STARTEXP', time_header_formatting = '%Y-%m-%dT%H:%M:%SZ',
                                    stat_type = 'median', show_plot = 1, n_std_lims = 3.5,
                                    save_plot = 0, save_plot_name = None, ax = None, data_sect = None,
                                    xlabel = r'$\Delta t$ (sec)', ylabel = None, labelsize = 16, title = '', titlesize = 20, color = 'k'):
    if ylabel == None:
        ylabel = stat_type + ' of counts in images'

    stats = []
    exp_time_strs = []
    for i in range(len(img_list)):
        img = img_list[i]
        data, header = can.readInDataFromFitsFile(img, data_dir, n_mosaic_image_extensions = n_mosaic_image_extensions)
        if data_sect != None:
            data = data[data_sect[0][0]:data_sect[0][1], data_sect[1][0]:data_sect[1][1]]
        if stat_type == 'median':
            stat = np.median(data)
        elif stat_type == 'mean':
            stat = np.mean(data)
        elif stat_type == 'std':
            stat = np.std(data)

        stats = stats + [stat]
        exp_time_strs = exp_time_strs + [header[time_header_key]]
    stats_mean = can.sigClipMean(stats, sig_clip = n_std_lims)
    stats_std = can.sigClipStd(stats, sig_clip = n_std_lims)
    exp_times_datetimes = [datetime.strptime(exp_time_str, time_header_formatting) for exp_time_str in exp_time_strs]
    exp_times = [exp_time.timestamp() for exp_time in exp_times_datetimes]
    min_time = min(exp_times)
    delta_ts = [exp_time - min_time for exp_time in exp_times]
    if ax == None:
        f, axarr = plt.subplots(1,1)
        ax = axarr
    ax.plot(delta_ts, stats, marker = '.', c = color)
    ax.set_ylim(stats_mean - stats_std * n_std_lims, stats_mean + stats_std * n_std_lims)
    ax.set_xlabel(xlabel, fontsize = labelsize)
    ax.set_ylabel(ylabel, fontsize = labelsize)
    ax.set_title(title, fontsize = titlesize)
    if save_plot and save_plot_name != None:
        plt.savefig(save_plot_name)
    if show_plot:
        plt.show()

    return exp_times, stats
