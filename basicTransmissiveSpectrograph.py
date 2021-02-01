#A class of a spectorgraph, that has:
# slit -> collimating lens -> transmissive diffraction grating -> focusing lens -> CCD chip.
#Because I want to be able to recalculate things in the this spectrograph, you can provide
# a new (nominally) fixed parameter into any of its functions, and it immediately updates.
#Just be careful that you are aware of any changes that you make!

#All angles should be given in radians!!!
#All distances should be given in mm!!! 
import math 

class TransmissiveSpectrograph:

    def computeNPixelsOfSlit(self, wavelength, dif_order, 
                             slit_width = None, col_focal_length = None,
                             grating_angle = None, grating_slit_sep = None,
                             focusing_focal_length = None,
                             pixel_width = None):
        if not(pixel_width is None):
            self.pixel_width = None
        projected_positions = self.computeProjectedPositions(wavelength, dif_order, 
                                                             slit_width = slit_width, col_focal_length = col_focal_length,
                                                             grating_angle = grating_angle, grating_slit_sep = grating_slit_sep,
                                                             focusing_focal_length = focusing_focal_length)
        if any([var is None
                for var in [self.slit_width,  self.col_focal_length,  self.grating_angle, self.grating_slit_sep, self.focusing_focal_length,  self.pixel_width]]):
            print 'Not enough information given to compute number of subtended pixels by single wavelength slit image. ' 
            n_pixels = None
        else:
            n_pixels = abs(projected_positions[0] - projected_positions[1]) / self.pixel_width
        return n_pixels  

    def computeProjectedPositions(self, wavelength, dif_order, 
                                  slit_width = None, col_focal_length = None,
                                  grating_angle = None, grating_slit_sep = None,
                                  focusing_focal_length = None):
        if not(focusing_focal_length is None):
            self.focusing_focal_length = focusing_focal_length
        
        angles_of_diffracted_column =  self.computeAnglesOfDiffractedSlit(wavelength, dif_order, 
                                                                          slit_width = slit_width, col_focal_length = col_focal_length,
                                                                          grating_angle = grating_angle, grating_slit_sep = grating_slit_sep)
        if any([var is None for var in [self.slit_width,  self.col_focal_length,  self.grating_angle, self.grating_slit_sep, self.focusing_focal_length]]):
            print 'Not enough information given to compute positions of single wavelength slit image. ' 
            projected_positions = [None, None]
        else:
            projected_positions = [self.focusing_focal_length * math.tan(angle - self.grating_angle) for angle in angles_of_diffracted_column]
            print 'projected_positions = ' + str(projected_positions) 
        return projected_positions 
            
        

    def computeAnglesOfDiffractedSlit(self, wavelength, dif_order, 
                                      slit_width = None, col_focal_length = None,
                                      grating_angle = None, grating_slit_sep = None):

        if not(grating_angle is None):
            self.grating_angle = grating_angle
        if not(grating_slit_sep is None):
            self.grating_slit_sep = grating_slit_sep

        init_slit_angle_sep = self.computeInitSlitAngleSep(slit_width = slit_width, col_focal_length = col_focal_length)

        if any([var is None for var in [self.slit_width,  self.col_focal_length,  self.grating_angle, self.grating_slit_sep]]):
            print 'Not enough information given to compute single diffracted wavelength column angles. ' 
            angles_of_diffracted_column = [None, None]
        else:
            angles_of_diffracted_column = [(math.asin((dif_order * float(wavelength)) / self.grating_slit_sep
                                                     + math.sin(self.grating_angle + side * init_slit_angle_sep/ 2.0 ))
                                           + self.grating_angle)
                                          for side in [-1.0, 1.0]]
            print 'angles_of_diffracted_column = ' + str(angles_of_diffracted_column) 
        return angles_of_diffracted_column 
    

    def computeInitSlitAngleSep(self, slit_width = None, col_focal_length = None):
        if not(slit_width is None):
            self.slit_width = slit_width
        if not(col_focal_length is None):
            self.col_focal_length = col_focal_length
        if (not (self.slit_width is None) and not (self.col_focal_length is None)):
            init_slit_angle_sep = math.atan(self.slit_width / 2.0 / self.col_focal_length) * 2.0
            print 'init_slit_angle_sep = ' + str(init_slit_angle_sep)
        else:
            print 'Not enough information given to compute initial slit opening angle. ' 
            init_slit_angle_sep = None
        return init_slit_angle_sep
            
    
    #Values that are None are assumed to be not ad-priori specified by the user
    #I've preloaded some values that are pretty fixed for our current effort 
    def __init__(self, slit_width = None, col_focal_length = 50.0,
                  grating_angle = None, grating_slit_sep = 1.0 / 300.0,
                  focusing_focal_length = None, pixel_width = 0.0054):
        self.slit_width = slit_width
        self.col_focal_length = col_focal_length
        self.grating_angle = grating_angle
        self.grating_slit_sep = grating_slit_sep
        self.focusing_focal_length = focusing_focal_length
        self.pixel_width = 0.0054 

        
        
