import cantrips as c 
import numpy as np
import matplotlib.pyplot as plt 



if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    data_file_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/spectrographElementsSpecifications/'

    ind_throughput_files = ['EdmundOptics_VIS-NIR35Vis_NIR_transmission.txt', 'NIRTransGratings.txt', 'edmundOpticsLongpass84756.txt', 'SWIR-50_Transmission_byHand.txt', 'BRCCDQE_atm70C.txt']
    n_ignore = [2, 1, 1, 1, 1]
    delimiters = [' ', ' ', ' ', ' ', ' ']
    wavelength_scalings = [1.0, 1.0, 1.0, 1000.0, 1.0]
    throughput_scalings = [1.0, 0.01, 0.01, 1.0, 1.0]

    ind_throughput_curves = [ c.readInColumnsToList(ind_throughput_files[i], file_dir = data_file_dir, n_ignore = n_ignore[i], delimiter = delimiters[i], convert_to_float = 1, convert_to_int = 0) for i in range(len(ind_throughput_files)) ]
    ind_throughput_curves = [[[wavelength * wavelength_scalings[i] for wavelength in ind_throughput_curves[i][0]], [throughput * throughput_scalings[i] for throughput in ind_throughput_curves[i][1]]] for i in range(len(ind_throughput_curves)) ]
    ind_throughput_fits = [c.safeInterp1d(curve[0], curve[1], out_of_bounds_val = 0.0)  for curve in ind_throughput_curves]

    wavelength_range = [450, 1100]
    wavelengths_to_calc = np.linspace(450, 1100, 651)

    total_interp = lambda wavelengths: c.productSum([ind_fit(wavelengths) for ind_fit in ind_throughput_fits]) 
    

    ind_lines = [i for i in range(len(ind_throughput_curves))]
    for i in range(len(ind_throughput_curves)):
        curve = ind_throughput_curves[i]
        ind_lines[i] = plt.plot(curve[0], curve[1])[0]
    total_line =  plt.plot(wavelengths_to_calc, total_interp(wavelengths_to_calc)) [0]

    plt.xlabel('Incident Wavelength (nm)')
    plt.ylabel('Sensitivity') 
    plt.xlim(wavelength_range)
    plt.ylim(0.0, 1.0) 
    plt.legend(ind_lines + [total_line], ['Collimating Lens', 'Diffraction Grating', 'Longpass Filter', 'Reimaging Lens', 'Camera', 'Total'])
    plt.show()

    gemini_dir = '/Users/sasha/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/'
    gemini_file = 'GeminiSkyBrightness.txt'
    gemini_brightness = c.readInColumnsToList(gemini_file, file_dir = gemini_dir, n_ignore = 14, delimiter = ' ', convert_to_float = 1)
    gemini_interp = c.safeInterp1d(gemini_brightness[0], gemini_brightness[1]) 

    slit_area = 3.0 * 10.0 ** -3.0 * 10.0 ** -6.0
    solid_angle = np.pi * (np.arctan(0.5) * 180.0 / np.pi * 3600) ** 2.0

    theoretical_photon_rate = lambda wavelengths: solid_angle * slit_area * gemini_interp(wavelengths) * total_interp(wavelengths)

    plt.plot(wavelengths_to_calc, theoretical_photon_rate(wavelengths_to_calc))
    plt.xlabel('Incident Wavelength (nm)')
    plt.ylabel(r'Count rate per wavelength (photons s$^{-1}$ nm$^{-1}$)') 
    plt.xlim(wavelength_range)
    plt.ylim(0.0, 1.01 * max(theoretical_photon_rate(wavelengths_to_calc)) )
    plt.show() 
    
    
    
