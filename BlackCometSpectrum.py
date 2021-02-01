#The BlackCometSpectrum Object reads in a Black Comet Spectrum file (.SSM).  
# It stores the header as a string in a list.
# It stores the wavelengths and measured ADUs at each wavelength each as a single list. 


def readInBlackCometSpectraFile(spectrum_file, first_data_line, header_lines): 
    spectrum_opened_object = open(spectrum_file)
    first_data_line = 3
    header_lines = [1]
    spectrum = [ [], [] ]
    header = [] 
    line_num = 0
    for line in spectrum_opened_object:
        if line_num >= first_data_line: 
            spectrum[0] = spectrum[0] + [float(line.split()[0])]
            spectrum[1] = spectrum[1] + [float(line.split()[1])]
        if line_num in header_lines:
            #print 'HEADER = ' 
            #print str(line)                        
            header = header + [str(line)]
        line_num = line_num + 1
    return header, spectrum

class BlackCometSpectrum: 

    def __init__ (self, spec_file, first_data_line = 3, header_lines = [1]):
        self.spectrum_file = spec_file

        if self.spectrum_file is None:
            self.header = []
            self.wavelengths = []
            self.adus = []
        else: 
            self.header, spectrum = readInBlackCometSpectraFile(self.spectrum_file, first_data_line, header_lines)
            self.wavelengths = spectrum[0]
            self.adus = spectrum[1] 
