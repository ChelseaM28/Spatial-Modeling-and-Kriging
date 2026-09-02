# Filename: main.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script will run each step of the kriging process.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from base.data_handling import DataHandler
from semivariogram import EmpiricalSemivariogram
from kriging import OrdKriging

class Main:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.datahandler = DataHandler(self.filepath)

    def main(self):
        cleaned_file = self.datahandler.clean_filter_and_save()
        semivariogram_obj = EmpiricalSemivariogram(cleaned_file)

        # This also updates self.location_pairs
        station_coords = semivariogram_obj.construct_location_pairs()

        # TODO: Later, I will ensure I can have multiple objects, 
        # likely one for each semivariogram model. This means I'll need to 
        # either pass in the model type I want or set it up in the config.
        # Right now I'm only constructing the exponential model.
        semivariogram_obj.construct_GMM()  # This updates initial_PGA
        
        # These 2 methods may not be removed, as the code depends on the values of these
        # functions which update in this expected order.
        semivariogram_obj.outliers()  # This updates outlier_treated_PGA, self.updated_location_pairs
        pga_values = semivariogram_obj.anisotropy()  # Updates PGA information once again

        semivariogram_obj.compute_empirical_semivariogram()

        # Fit the exponential model's sill (a) and range (b) via WLS.
        a_fit, b_fit = semivariogram_obj.sill_and_range()

        # choose_covariance_model() (the hand-built C matrix) isn't needed here since
        # PyKrige builds its own internal covariance from a_fit/b_fit directly.
        # cov_model = semivariogram_obj.choose_covariance_model()

        krig = OrdKriging(a_fit, b_fit, station_coords, pga_values)

        predictions, actuals = krig.LOO_cross_validation()

        print(f"LOOCV Result [PREDICTION | ACTUAL]:\n{list(zip(predictions, actuals))}")
        return predictions, actuals

    
        
