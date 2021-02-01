#We want to fit a function like: pix(wave) = a + b * wave + c * wave ** 2.0 to the data
# I can take a guess the initial values of a, b, and c or determine initial values of 
#  a, b, and c by an initial guess of the positions of the lines for which I am looking.
#I then need to do this fit based on the positions of lines.  How do I determine the positions
# of lines?  I think running a fit around each of the points where I believe a line to be
# would be a good way to start.
#So strategy is: try to fit each of the guessed lines.  If we manage to fit them, add them
# to the list of well fitted lines.  If not, catch the error and move on.
#This should give us a list of lines that we can use to determine a wavelength solution
# and a seeing function.  So, actually, we should be able to do both in one go...

import matplotlib.pyplot as plt
import numpy as np
import math
from cantrips import readInDataFromFitsFile
import SimulatedCCDData as SCD
import scipy.optimize as optimize
from cantrips import safeSortOneListByAnother

def getOneDSpectrumFromFile(file_name, file_dir = '', summing_method = 'median', size = [1024, 1024], binning = [1,1], spectrum_box = [[0, 1023], [417, 652]], pix_to_wavelength_funct = None ):
    data_array, header = readInDataFromFitsFile(file_name, file_dir)
    print ('np.shape(data_array) = ' + str(np.shape(data_array))) 
    one_d_spectrum = SCD.measureSpectrum(spectrum_box, data_array, summing_method = summing_method, binning = binning, pix_to_wavelength_funct = pix_to_wavelength_funct )
    return one_d_spectrum

def measureWavelengthSolution(one_d_spectrum_to_fit, start_spectrum = 'KR1', wavelength_fit_width = 5.0, show_line_fits = 1, init_guess_width = 2.0, 
                                                     guess_wave_to_pix_scale = 4.0/3.0, binning = 1.0, single_function_to_fit = 'single_hump',
                                                     wavelength_solution_order = 2, seeing_function = 'gen_power', seeing_function_init_guess = [] ):

    if single_function_to_fit in ['single_hump']:
        single_function_to_fit = lambda pix, A, mu, sig: A * np.exp(-(pix - mu) ** 2.0 / (2.0 * sig ** 2.0))
    
    spectrum_pixels = np.array(one_d_spectrum_to_fit[0])
    spectrum_flux = np.array(one_d_spectrum_to_fit[1])
    if start_spectrum in ['KR1', 'kr1', 'kr', 'KR', 'KR-1', 'kr-1', 'Krypton', 'krypton', 'KRYPTON']:
        #start_spectrum = {431.958:None, 436.264:None, 437.612:None, 439.997:None, 445.392:None, 446.369:None, 450.235:None, 556.222:653.8, 557.029:653.8,
        #                  587.096:780.5, 758.741:1507.4, 760.155:1513, 768.525:1549.6, 769.454:1552.9, 785.482:1618.6, 791.343:1622.7, 805.950:1709.8, 810.436:1732.0,
        #                  819.006:1766.3, 826.324:1798.9, 829.811:1813.3, 877.675:2020.1, 892.869:2085.5, 1181.938:None, 1220.353:None, 1317.741:None, 1363.422:None,
        #                  1442.679:None, 1473.444:None, 1537.204:None, 1689.676:None, 1800.223:None, 1816.733:None, 2190.851:None}
        start_spectrum = {431.958:None, 436.264:None, 437.612:None, 439.997:None, 445.392:None, 446.369:None, 450.235:None, 556.222:None, 557.029:224.0,
                          587.096:261.0, 758.741:465.0, 760.155:468.0, 768.525:481.0, 769.454:None, 785.482:501.0, 791.343:None, 805.950:526.0, 810.436:530.0, 
                          819.006:540.0, 826.324: 549.0, 829.811:553.0, 877.675:611.0, 892.869:629.0, 1181.938:None, 1220.353:None, 1317.741:None, 1363.422:None,
                          1442.679:None, 1473.444:None, 1537.204:None, 1689.676:None, 1800.223:None, 1816.733:None, 2190.851:None}
        
    
    fitting_wavelengths = sorted([key for key in start_spectrum.keys() if not(start_spectrum[key] is None) ])
    print ('fitting_wavelengths = ' + str(fitting_wavelengths) )
    
    #single_function_to_fit = lambda pix, A, mu, sig: A * np.exp(-(pix - mu) ** 2.0 / (2.0 * sig ** 2.0))

    
    pix_fit_width = wavelength_fit_width * guess_wave_to_pix_scale / binning 
    sig = 0.5
    A = 10.0

    grouped_fitting_wavelengths = [[fitting_wavelengths[0]]]
    for wave in fitting_wavelengths[1:]:
        print ('current_wavelength = ' + str(grouped_fitting_wavelengths[-1]))
        if wave - wavelength_fit_width <= grouped_fitting_wavelengths[-1][-1]:
            print ('Appending wave = ' + str(wave) + ' to current bin. ' )
            grouped_fitting_wavelengths[-1] = grouped_fitting_wavelengths[-1] + [wave]
        else:
            print ('Using wave = ' + str(wave) + ' to start new bin. ' )
            grouped_fitting_wavelengths = grouped_fitting_wavelengths + [[wave]]

    fluxes = []
    centers = []
    seeings = []
    backgrounds = []
    used_waves = [] 

    for wave_set in grouped_fitting_wavelengths:
        print ('wave_set = ' + str(wave_set))
        data_indeces = [np.argmin(abs(start_spectrum[wave_set[0]] - pix_fit_width - spectrum_pixels)),
                        np.argmin(abs(start_spectrum[wave_set[-1]] + pix_fit_width - spectrum_pixels)) ]
        print ('data_indeces = ' + str(data_indeces)) 
        
        x_data = spectrum_pixels[data_indeces[0]:data_indeces[1]]
        y_data = spectrum_flux[data_indeces[0]:data_indeces[1]]
        full_fitting_funct = lambda pix, *single_line_params_with_shift: sum([single_function_to_fit(pix, *(single_line_params_with_shift[i*3:i*3+3]))
                                                                             for i in range(len(wave_set))]) + single_line_params_with_shift[-1]

        init_param_guesses = []
        param_bounds = [[], []]
        max_val = max(y_data)
        median_val = np.median(y_data)
        for wave in wave_set:
            param_bounds[0] = param_bounds[0] + [0.0, x_data[0] - pix_fit_width, 0.1]
            param_bounds[1] = param_bounds[1] + [np.inf, x_data[-1] + pix_fit_width, x_data[-1] - x_data[0]]
            init_param_guesses = init_param_guesses + [max_val - median_val, start_spectrum[wave], init_guess_width]
        init_param_guesses = init_param_guesses + [median_val]
        param_bounds[0] = param_bounds[0] + [-np.inf]
        param_bounds[1] = param_bounds[1] + [np.inf]
        param_bounds = tuple(param_bounds) 

        print ('param_bounds = ' + str(param_bounds))
        print ('init_param_guesses = ' + str(init_param_guesses))
        print ('fluxes = ' + str(fluxes))
        print ('centers = ' + str(centers))
        print ('seeings = ' + str(seeings))
        print ('backgrounds = ' + str(backgrounds) )
        
            
        try: 
            fitting_results = optimize.curve_fit(full_fitting_funct, x_data, y_data, p0 = init_param_guesses, bounds = param_bounds)[0].tolist()
            new_fluxes = []
            new_centers = []
            new_seeings = []
            for i in range(len(wave_set)):
                new_fluxes = new_fluxes + [fitting_results[i*3]]
                new_centers = new_centers + [fitting_results[i*3 + 1]]
                new_seeings = new_seeings + [fitting_results[i*3 + 2]]
            new_fluxes, new_centers, new_seeings = safeSortOneListByAnother(new_centers, [new_fluxes, new_centers, new_seeings])
            new_background = fitting_results[-1] 
            print ('fitting_results = ' + str(fitting_results))
            fluxes = fluxes + new_fluxes
            centers = centers + new_centers
            seeings = seeings + new_seeings
            backgrounds = backgrounds + [new_background]
            used_waves = used_waves + wave_set
            if show_line_fits:
                legend_components = [plt.scatter(x_data, y_data), plt.plot(x_data, full_fitting_funct(x_data, *fitting_results))[0]]
                plt.xlabel('pixel value')
                plt.ylabel('Summed column intensity')
                plt.legend(legend_components, ['data','line fit(s)'])
            
        except RuntimeError:
            fitting_results = init_param_guesses
            print ('Failed to fit lines for these waves. ')
        
        #single_line_params = []
        #for wave in wave_set:
        #    single_line_params = single_line_params + [A, start_spectrum[wave], sig]
        if show_line_fits:
            legend_components = [plt.scatter(x_data, y_data), plt.plot(x_data, full_fitting_funct(x_data, *fitting_results))[0]]
            plt.xlabel('pixel value')
            plt.ylabel('Summed column intensity')
            plt.legend(legend_components, ['data','line fits'])
            plt.show() 

    if seeing_function in ['gen_power']:
        include_shift = 1
        if include_shift: 
            seeing_function = lambda wavelength, wavelength0, scaling, power, shift: scaling * (abs(np.array(wavelength) - wavelength0)) ** power + shift
            seeing_function_init_guess = [np.mean(used_waves), 0.001, 0.0, np.min(seeings)]
        else: 
            seeing_function = lambda wavelength, wavelength0, scaling, power: scaling * (abs(np.array(wavelength) - wavelength0)) ** power + 1.0
            seeing_function_init_guess = [np.mean(used_waves), 0.001, 0.0]
        
    pix_to_wave_poly_params = np.polyfit(centers, used_waves, wavelength_solution_order )
    wave_to_pix_poly_params = np.polyfit(used_waves, centers, wavelength_solution_order )
    legend_components = [plt.scatter(centers, used_waves),
                         plt.plot(centers, [pix_to_wave_poly_params[0] * x ** 2.0 + pix_to_wave_poly_params[1] * x + pix_to_wave_poly_params[2] for x in centers], c = 'r')[0]]
    plt.legend(legend_components, ['Measured pixel-to-wavelength points', 'quadratic wavelength solution'])
    plt.xlabel('pixel position (pix)')
    plt.ylabel('wavelength (nm)') 
    plt.show()
    pix_to_wavelength_funct = lambda xs: sum([pix_to_wave_poly_params[i] * xs **  (wavelength_solution_order - i) for i in range( wavelength_solution_order + 1)])
    wavelength_to_pix_funct = lambda xs: sum([wave_to_pix_poly_params[i] * xs **  (wavelength_solution_order - i) for i in range( wavelength_solution_order + 1)])
    print ('pix_to_wave_poly_params = ' + str(pix_to_wave_poly_params))
    try: 
        seeing_fit_params = optimize.curve_fit(seeing_function, used_waves, seeings, p0 = seeing_function_init_guess)[0] 
        print ('seeing_fit_params = ' + str(seeing_fit_params))
    except RuntimeError:
        seeing_fit_params = seeing_function_init_guess
        print ('Seeing function failed to converge.  Providing default of mean seeing value. ')
    wavelength_to_seeing_funct = lambda xs: seeing_function(xs, *seeing_fit_params)
            
    legend_components = [plt.scatter(used_waves, seeings),
                         plt.plot(np.linspace(min(used_waves), max(used_waves), 100), wavelength_to_seeing_funct(np.linspace(min(used_waves), max(used_waves), 100)), c = 'r')[0]]
    plt.legend(legend_components, ['Measured pixel-to-seeing points', 'power law seeing solution'])
    plt.xlabel('wavelength (nm)')
    plt.ylabel('width of lines (pix)') 
    plt.show()

    return [fluxes, centers, seeings, backgrounds, wavelength_to_pix_funct, pix_to_wavelength_funct, wavelength_to_seeing_funct] 
