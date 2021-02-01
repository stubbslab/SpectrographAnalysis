import numpy as np
from BlackCometSpectrum import BlackCometSpectrum  

#Expects set of BlackCometSpectrum Objects.  Each element in array is itself an set with some number of objects
# taken under same observing conditions (from which stdev and med will be computed and plotted on PTC) 
def computePhotonTransferCurve(raw_phot_transfer_spectra, bias_spectra, std_from_diff_of_pairs = 1, sub_mean_bias = 1, rigid_bias_match = 0, compute_ptc_by_wavelength = 0, compute_ptc_from_mean = 0):
    sub_rand_bias = 1
    if rigid_bias_match and (np.shape(raw_phot_transfer_spectra) == np.shape(bias_spectra)):
        sub_rand_bias = 0
    elif rigid_bias_match:
        print 'shapes of photon transfer spectrum and bias spectrum do not match.  Will subtract itemized bias. '
    if len(bias_spectra) >= len(raw_phot_transfer_spectra[0]):
        sub_itemized_bias = 1
    else:
        sub_itemized_bias = 0 
    if not(sub_itemized_bias): 
        bias_spectra = [spectrum for spectrum in bias_spectra]
    mean_bias = np.mean([bias.adus for bias in bias_spectra], axis = 0)
    print 'len(mean_bias) = ' + str(len(mean_bias)) 
    print 'mean_bias = ' + str(mean_bias)

    n_wavelengths = len(raw_phot_transfer_spectra[0][0].wavelengths)

    bc_phot_transfer_spectra = [[] for spectrum_set in raw_phot_transfer_spectra] #bias corrected spectra for computing photon transfer curve
    bc_means = [[] for spectrum_set in raw_phot_transfer_spectra]
    bc_totals = [[] for spectrum_set in raw_phot_transfer_spectra]
    bc_meds_by_wavelength = [[] for spectrum_set in raw_phot_transfer_spectra]
    bc_stds_by_wavelength = [[] for spectrum_set in raw_phot_transfer_spectra]
    bc_meds_of_means = []
    bc_meds_of_totals = []
    bc_stds_of_means = []
    bc_stds_of_totals = []
    
    if std_from_diff_of_pairs: 
        bc_ps_phot_transfer_spectra = [[] for i in range(len(raw_phot_transfer_spectra[0]))] #pair subtracted , bias corrected spectra for computing photon transfer curve
        bc_ps_stds_by_wavelength = [[] for i in range(len(raw_phot_transfer_spectra))]
        bc_ps_means = [[] for i in range(len(raw_phot_transfer_spectra))]
        bc_ps_totals = [[] for i in range(len(raw_phot_transfer_spectra))]
        bc_ps_stds_of_means = []
        bc_ps_stds_of_totals = []


    for i in range(len(raw_phot_transfer_spectra)):
        print 'Working on i = ' + str(i) + ' of ' + str(len(raw_phot_transfer_spectra)) 
        for j in range(len(raw_phot_transfer_spectra[i])):
            bc_phot_transfer_spectra[i] = bc_phot_transfer_spectra[i] + [BlackCometSpectrum(None)]
            bc_phot_transfer_spectra[i][j].wavelengths = raw_phot_transfer_spectra[i][j].wavelengths[:]
            bc_phot_transfer_spectra[i][j].header = raw_phot_transfer_spectra[i][j].header[:]
            if sub_mean_bias:
                bc_phot_transfer_spectra[i][j].adus = (np.array(raw_phot_transfer_spectra[i][j].adus) - np.array(mean_bias)).tolist()
            elif sub_itemized_bias:
                bc_phot_transfer_spectra[i][j].adus = (np.array(raw_phot_transfer_spectra[i][j].adus) - np.array(bias_spectra[j].adus)).tolist()
            else:
                print 'You do not have enough bias spectra (the number of observation spectra for a single type of exposure is greater than number of bias spectra).  '
                print 'Returning 0. '
                return 0
            bc_means[i] = bc_means[i] + [np.mean(bc_phot_transfer_spectra[i][j].adus)]
            bc_totals[i] = bc_totals[i] + [sum(bc_phot_transfer_spectra[i][j].adus)]
            if std_from_diff_of_pairs:
                if j % 2 == 1:
                    #print 'Here on j = ' + str(j) 
                    bc_ps_phot_transfer_spectra[i] = bc_ps_phot_transfer_spectra[i] + [np.array(bc_phot_transfer_spectra[i][j].adus) - np.array(bc_phot_transfer_spectra[i][j-1].adus)]
                    #print 'np.shape(bc_ps_phot_transfer_spectra) = ' + str(np.shape(bc_ps_phot_transfer_spectra))
                    #print 'np.shape(bc_ps_phot_transfer_spectra[0]) = ' + str(np.shape(bc_ps_phot_transfer_spectra[0]))
                    bc_ps_means[i] = bc_ps_means[i] + [np.mean(bc_ps_phot_transfer_spectra[i][-1])]
                    bc_ps_totals[i] = bc_ps_totals[i] + [sum(bc_ps_phot_transfer_spectra[i][-1])]
                
                                                   

        bc_meds_of_means = bc_meds_of_means + [np.median(bc_means[i])]
        bc_stds_of_means = bc_stds_of_means + [np.std(bc_means[i])]
        bc_ps_stds_of_means = bc_ps_stds_of_means + [np.std(bc_ps_means[i])]
        bc_meds_of_totals = bc_meds_of_totals + [np.median(bc_totals[i])]
        bc_stds_of_totals = bc_stds_of_totals + [np.std(bc_totals[i])]
        bc_ps_stds_of_totals = bc_ps_stds_of_totals + [np.std(bc_ps_totals[i])] 
        #print 'bc_meds_of_means = ' + str(bc_meds_of_means )
        bc_meds_by_wavelength[i] = [np.median([bc_phot_transfer_spectra[i][j].adus[k] for j in range(len(raw_phot_transfer_spectra[i]))], axis = 0)
                                     for k in range(n_wavelengths)]
        bc_stds_by_wavelength[i] = [np.std([bc_phot_transfer_spectra[i][j].adus[k] for j in range(len(raw_phot_transfer_spectra[i]))], axis = 0)
                                     for k in range(n_wavelengths)]
        print 'np.shape(np.array(bc_ps_phot_transfer_spectra)) = ' + str(np.array(np.shape(bc_ps_phot_transfer_spectra))) 
        bc_ps_stds_by_wavelength[i] = [np.std([bc_ps_phot_transfer_spectra[i][j][k] for j in range(len(bc_ps_phot_transfer_spectra[i]))], axis = 0)
                                        for k in range(n_wavelengths)]
    if std_from_diff_of_pairs:
        stds_of_means_for_ptc = bc_ps_stds_of_means
        stds_of_totals_for_ptc = bc_ps_stds_of_totals
        stds_by_wavelength_for_ptc = bc_ps_stds_by_wavelength
    else:
        stds_of_means_for_ptc = bc_stds_of_means
        stds_of_totals_for_ptc = bc_stds_of_totals
        stds_by_wavelength_for_ptc = bc_stds_by_wavelength

    if std_from_diff_of_pairs:
        ptc_by_mean = np.polyfit(bc_meds_of_means, np.array(stds_of_means_for_ptc) ** 2.0 / 2.0, 1)
        ptc_by_total = np.polyfit(bc_meds_of_totals, np.array(stds_of_totals_for_ptc) ** 2.0 / 2.0, 1)
        ptcs_by_wavelength = [np.polyfit([bc_meds_by_wavelength[i][k] for i in range(len(bc_meds_by_wavelength))], [stds_by_wavelength_for_ptc[i][k] ** 2.0 / 2.0 for i in range(len(stds_by_wavelength_for_ptc))], 1)
                              for k in range(len(bc_meds_by_wavelength[0])) ]
    else:
        ptc_by_mean = np.polyfit(bc_meds_of_means, np.array(stds_of_means_for_ptc) ** 2.0, 1)
        ptc_by_total = np.polyfit(bc_meds_of_totals, np.array(stds_of_totals_for_ptc) ** 2.0, 1)
        ptcs_by_wavelength = [np.polyfit([bc_meds_by_wavelength[i][k] for i in range(len(bc_meds_by_wavelength))], [stds_by_wavelength_for_ptc[i][k] ** 2.0 for i in range(len(stds_by_wavelength_for_ptc))], 1)
                              for k in range(len(bc_meds_by_wavelength[0])) ]


    if compute_ptc_by_wavelength:
        return bc_meds_by_wavelength, stds_by_wavelength_for_ptc, ptcs_by_wavelength
    elif compute_ptc_from_mean:
        return bc_meds_of_means, stds_of_means_for_ptc, ptc_by_mean
    else:
        return bc_meds_of_totals, stds_of_totals_for_ptc, ptc_by_total

    
