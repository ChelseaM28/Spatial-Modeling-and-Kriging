# Filename: semivariogram.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script contains classes and methods for 
#               computing and plotting the empirical semivariogram.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from common.base import Plotting, Preprocessing
import numpy as np

# Not all of these methods will be performed, as the scope of the project will need to be limited.   


class EmpiricalSemivariogram:
    def __init__(self, filtered_data_object): 
        self.data_object = filtered_data_object  # Filtered to chosen earhquake only + outliers removed.
        # I am using peak ground acceleration as my parameter of interest with the 
        # assumption that professionals in industry would find this metric more useful.
        # It is "a natural simple design parameter since it can be related 
        # to a force and for simple design" - USGS Earthquake Hazards 201
        self.PGA_true = {} 
        self.log_PGA_trues = []
        
        self.initial_PGA = []
        self.residuals_sum = []
        self.outlier_treated_PGA = []
        self.anisotropy_treated_PGA = []

        self.location_pairs = []

        self.semivariogram = []  # This is the experimental variogram referenced in literature.
        self.station_variance = {}

        self.cov_model = None

    def construct_location_pairs(self):
        return self.location_pairs

    def construct_GMM(self):
        # Model construction belongs here.
        # self.PGA_predicted for each station is updated --> given by GMM models
        self.PGA_predicted = None 
        # self.station_variance for each station is updated --> given by GMM models
        self.station_variance = None 
        self.initial_PGA = None  # Here we update self.initial_HMM so we don't have to use a non-local variable
        return self.initial_PGA  

    def sample_size(self):
        pass
        
    def lag_interval_and_bin_width(self):
        pass

    def marginal_distribution(self):
        pass

    def outliers(self):
        # It looks odd, but the PRIOR location pairs are used to UPDATED
        self.outlier_treated_PGA, self.location_pairs = Preprocessing.detect_outliers(
            self.data_object, self.location_pairs, self.initial_PGA
            )
        # NOTE: This is already a function in the preprocessing class. 
        return self.outlier_treated_PGA, self.location_pairs

    def anisotropy(self):
        # Be sure to use outlier_treated_PGA
        return self.anisotropy_treated_PGA

    def trend(self):
        pass

    def compute_empirical_semivariogram(self): 
        # Be sure to use anisotropy_treated_PGA values

        # Now I need ε˜, "the sum of the intra-event residual (εi) and inter-event residual (η) 
        # normalized by the standard deviation of the intra-event residual (σi).
        for station_1, station_2 in self.location_pairs:
            log_PGA_predicted_station1 = np.log(self.anisotropy_treated_PGA[station_1])
            log_PGA_predicted_station2 = np.log(self.anisotropy_treated_PGA[station_2])
            
            self.residuals_sum.append(
                ((self.log_PGA_true[station_1]-log_PGA_predicted_station1)/self.station_variance[station_1]), 
                ((self.log_PGA_true[station_2]-log_PGA_predicted_station2)/self.station_variance[station_2])
                )
        # I need to check when this one is supposed to be used.
        self.semivariogram = None
        return self.semivariogram

    # methods for fitting a covariance model to the semivariogram via WLS.
    def spherical_model(self):
        pass
    
    def exponential_model(self):
        pass

    def power_empirical_model(self):
        pass

    def choose_covariance_model(self, semivar_values):
        self.cov_model = None
        pass


class PlotEmpiricalSemivariogram:  # I will figure this out later.
    def __init__(self, data_object):
        self.data_object = data_object
        self.passed_to_plotting_class = None

    def plot_empirical_semivariogram(self):
        #self.passed_to_plotting_class = some operation
        Plotting(self.passed_to_plotting_class).plot_passed_data()
        print("Empirical Semivariogram plotted. Check output folder > graphics for the graph.")
        pass
