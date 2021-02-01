import math 
import numpy as np 
import math 
import sys
sys.path.insert(0, 'C:\\Users\\labpc\\Documents\\sashasPythonScripts')
from manageBlackCometSpectra import readInBlackCometSpectraFile
from BlackCometSpectrum import BlackCometSpectrum

class BlackCometSpectrumAnalyzer: 

    def computePhotonTransferCurves(self, compute_by_wavelength = 1, 
                                    use_raw_darks = 0, use_bc_darks = 0, use_raw_continuum = 1, use_bc_continuum = 0) :
        #Each element in this set consists of an array of paired arrays:
        #  the first is a set of observed wavelengths (should be same for all if we are calculating continuum) and the second is the ADUs for that exposure
        frames_to_compute_stats = [] 
        if use_raw_darks: 
            for exp_time in self.dark_raw_spectra_by_exposure.keys():
                frames_to_compute_stats = frames_to_compute_stats + [self.dark_raw_spectra_by_exposure[exp_time]]
        if use_bc_darks:
            for exp_time in self.dark_bc_spectra_by_exposure.keys():
                frames_to_compute_stats = frames_to_compute_stats + [self.dark_bc_spectra_by_exposure[exp_time]]
        if use_raw_continuum: 
            for exp_time in self.cont_raw_spectra_by_exposure.keys():
                frames_to_compute_stats = frames_to_compute_stats + [self.cont_raw_spectra_by_exposure[exp_time]]
        if use_bc_continuum:
            for exp_time in self.cont_bc_spectra_by_exposure.keys():
                frames_to_compute_stats = frames_to_compute_stats + [self.cont_bc_spectra_by_exposure[exp_time]]
        
        self.frames_to_compute_stats = frames_to_compute_stats 
        if compute_by_wavelength:
            arrays_to_measure_stats = [[], []]
            photon_transfer_curves = [[], [], []]
            wavelengths = frames_to_compute_stats[0][0][0]
            arrays_to_measure_stats[0] = wavelengths
            photon_transfer_curves[0] = wavelengths
            arrays_to_measure_stats[1] = [[] for wavelength in wavelengths]
            for i in range(len(wavelengths)): 
                for spectra_set in frames_to_compute_stats: 
                    stats_at_wavelength = [spectrum[1][i] for spectrum in spectra_set]
                    arrays_to_measure_stats[1][i] = arrays_to_measure_stats[1][i] + [stats_at_wavelength]
                #print 'len(arrays_to_measure_stats[1][i]) = ' + str(len(arrays_to_measure_stats[1][i]))
                #print 'len(arrays_to_measure_stats[1][i][0]) = ' + str(len(arrays_to_measure_stats[1][i][0]))
                photon_transfer_curves[1] = photon_transfer_curves[1] + [[float(np.median(same_condition_spectra_set)) for same_condition_spectra_set in arrays_to_measure_stats[1][i]]]
                photon_transfer_curves[2] = photon_transfer_curves[2] + [[float(np.std(same_condition_spectra_set)) for same_condition_spectra_set in arrays_to_measure_stats[1][i]]]     
            
        else:
            arrays_to_measure_stats = []
            print 'still_working_on_this' 


        return photon_transfer_curves

    def __init__(self, bias_exp_labels, dark_exp_lengths, dark_exp_labels, cont_exp_lengths, cont_exp_labels, 
                 bias_exp_length = 1, bias_prefix = 'dark', bias_midfix = 'ms_', bias_suffix = '.SSM',
                 biasDirectory = "C:/Users/labpc/Documents/testSpectraBlackComet/", 
                 dark_prefix = 'dark', dark_midfix = 'ms_', dark_suffix = '.SSM',
                 darkDirectory = "C:/Users/labpc/Documents/testSpectraBlackComet/", 
                 cont_prefix = 'cont', cont_midfix = 'ms_', cont_suffix = '.SSM',
                 contDirectory = "C:/Users/labpc/Documents/testSpectraBlackComet/",
                 subtract_indiv_bias = 1):
        self.contDirectory = contDirectory
        self.darkDirectory = darkDirectory
        self.biasDirectory = biasDirectory 
        #self.darkDirectory =  "C:/Users/labpc/Documents/testSpectraBlackComet/"
        #dark_prefix = 'dark'
        #dark_exp_lengths = [1, 10, 50, 100, 1000, 2000, 5000, 10000]
        first_data_line = 3
        header_line = 1
        #dark_midfix = 'ms_'
        #dark_exp_labels = [1, 2, 3, 4, 5]
        #dark_suffix = '.SSM'

        self.bias_files = []
        self.bias_spectra = []
        self.bias_headers = [] 

        self.dark_files_by_exposure = {} 
        self.dark_raw_spectra_by_exposure = {}
        self.dark_headers_by_exposure = {}

        self.cont_files_by_exposure = {} 
        self.cont_raw_spectra_by_exposure = {}
        self.cont_headers_by_exposure = {}
 

        self.bias_files = [] 
        for label in bias_exp_labels:
            bias_file = self.biasDirectory + bias_prefix + str(bias_exp_length) + bias_midfix + str(label) + bias_suffix 
            self.bias_files = self.bias_files + [bias_file]
            bias_spectrum = BlackCometSpectrum(bias_file)
            self.bias_spectra = self.bias_spectra + [bias_spectrum]

        for exp_time in dark_exp_lengths:
            print 'dark exp_time = ' + str(exp_time) 
            dark_files_set = []
            dark_spectrum_set = []
            dark_headers_set = []
            for label in dark_exp_labels: 
                dark_spectrum_file = self.darkDirectory + dark_prefix + str(exp_time) + dark_midfix + str(label) + dark_suffix
                dark_files_set = dark_files_set + [dark_spectrum_file]
                dark_spectrum = BlackCometSpectrum(dark_spectrum_file)
                dark_spectrum_set = dark_spectrum_set + [dark_spectrum]

            self.dark_raw_spectra_by_exposure[exp_time] = dark_spectrum_set
            self.dark_files_by_exposure[exp_time] = dark_files_set 

        for exp_time in cont_exp_lengths:
            print 'cont exp_time = ' + str(exp_time) 
            cont_files_set = []
            cont_spectrum_set = []
            cont_headers_set = []
            for label in cont_exp_labels: 
                cont_spectrum_file = self.contDirectory + cont_prefix + str(exp_time) + cont_midfix + str(label) + cont_suffix
                cont_spectrum = BlackCometSpectrum(cont_spectrum_file)
                cont_spectrum_set = cont_spectrum_set + [cont_spectrum]
                cont_files_set = cont_files_set + [cont_spectrum_file] 

            self.cont_raw_spectra_by_exposure[exp_time] = cont_spectrum_set
            self.cont_files_by_exposure[exp_time] = cont_files_set 

        med_bias = BlackCometSpectrum(None)
        med_bias.wavelengths = self.bias_spectra[0].wavelengths
        set_of_bias_adus = [spectrum.adus for spectrum in self.bias_spectra]
        med_bias.adus = np.median(set_of_bias_adus, axis = 0) 

        n_wavelengths = len(med_bias.wavelengths) 
    
        self.dark_bc_spectra_by_exposure = {}
        #print self.dark_raw_spectra_by_exposure
        for exp_time in self.dark_raw_spectra_by_exposure.keys():
            bc_darks = []
            for i in range(len(dark_exp_labels)):
                raw_dark = self.dark_raw_spectra_by_exposure[exp_time][i]
                if subtract_indiv_bias == 1 and len(bias_exp_labels) >= len(dark_exp_labels): 
                    bias = self.bias_spectra[i]
                else: 
                    if len(bias_exp_labels) < len(dark_exp_labels): print 'Too few bias frames to do individual dark subtractions.  Using median bias. '
                    bias = med_bias 
                bc_dark = BlackCometSpectrum(None)
                bc_dark.wavelengths = raw_dark.wavelengths
                bc_dark.adus = (np.array(raw_dark.adus) - np.array(bias.adus)).tolist() 
                bc_darks = bc_darks + [bc_dark]
            self.dark_bc_spectra_by_exposure[exp_time] = bc_darks

        self.cont_bc_spectra_by_exposure = {}
        #print self.dark_raw_spectra_by_exposure
        for exp_time in self.cont_raw_spectra_by_exposure.keys():
            bc_conts = []
            for i in range(len(cont_exp_labels)):
                raw_cont = self.cont_raw_spectra_by_exposure[exp_time][i]
                if subtract_indiv_bias == 1 and len(bias_exp_labels) >= len(cont_exp_labels): 
                    bias = self.bias_spectra[i]
                else: 
                    if len(bias_exp_labels) < len(cont_exp_labels): print 'Too few bias frames to do individual dark subtractions.  Using median bias. '
                    bias = med_bias 
                bc_cont = BlackCometSpectrum(None)
                bc_cont.wavelengths = raw_cont.wavelengths
                bc_cont.adus = (np.array(raw_cont.adus) - np.array(bias.adus)).tolist() 
                bc_conts = bc_conts + [bc_cont]
            self.cont_bc_spectra_by_exposure[exp_time] = bc_conts


        self.dark_med_raw_spectra_by_exposure = {}
        for exp_time in self.dark_raw_spectra_by_exposure.keys():
            n_exposures_to_avareage = len(self.dark_raw_spectra_by_exposure[exp_time])
            med_counts_by_wavelength = np.median([self.dark_raw_spectra_by_exposure[exp_time][j].adus for j in range(len(dark_exp_labels))], axis = 0)
            dark_med_spectrum = BlackCometSpectrum(None)
            dark_med_spectrum.wavelengths = self.dark_raw_spectra_by_exposure[exp_time][0].wavelengths
            dark_med_spectrum.adus = med_counts_by_wavelength 
            print 'Finished computing med_counts_by_wavelength for raw dark spectrum of exp_time = ' + str(exp_time) 
            
            self.dark_med_raw_spectra_by_exposure[exp_time] = dark_med_spectrum 

        self.cont_med_raw_spectra_by_exposure = {}
        for exp_time in self.cont_bc_spectra_by_exposure.keys():
            n_exposures_to_avareage = len(self.cont_bc_spectra_by_exposure[exp_time])
            med_counts_by_wavelength = np.median([self.cont_bc_spectra_by_exposure[exp_time][j].adus for j in range(len(cont_exp_labels))], axis = 0)
            cont_med_spectrum = BlackCometSpectrum(None)
            cont_med_spectrum.wavelengths = self.cont_raw_spectra_by_exposure[exp_time][0].wavelengths
            print 'Finished computing med_counts_by_wavelength for bc dark spectrum of exp_time = ' + str(exp_time) 
            cont_med_spectrum.adus = med_counts_by_wavelength 
            self.cont_med_raw_spectra_by_exposure[exp_time] = cont_med_spectrum

        dark_coefs_by_wavelength = []
        bias_coefs_by_wavelength = [] 
        #self.dark_med_raw_spectra_by_exposure = {}
        for i in range(n_wavelengths): 
            exp_times = []
            dark_current_by_exp_time = []
            for exp_time in self.dark_raw_spectra_by_exposure.keys():
                exp_times = exp_times + [exp_time]
                dark_current_by_exp_time = dark_current_by_exp_time + [self.dark_med_raw_spectra_by_exposure[exp_time].adus[i]]
            lin_coefs = np.polyfit(exp_times, dark_current_by_exp_time, 1)
            bias_coef = lin_coefs[0]
            dark_coef = lin_coefs[1]
            bias_coefs_by_wavelength = bias_coefs_by_wavelength + [bias_coef]
            dark_coefs_by_wavelength = dark_coefs_by_wavelength + [dark_coef]

        self.coefs_by_wavelength = [self.dark_med_raw_spectra_by_exposure[self.dark_med_raw_spectra_by_exposure.keys()[0]].wavelengths, bias_coefs_by_wavelength, dark_coefs_by_wavelength]
