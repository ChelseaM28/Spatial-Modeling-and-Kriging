# Filename: semivariogram.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script will run each step of the kriging process.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from data_scripts.data_handling import DataHandler
from semivariogram import EmpiricalSemivariogram
from kriging import OrdinaryKriging

class Main:
    def __init__(self, filepath):
        self.filepath = filepath
        self.datahandler = DataHandler(self.filepath)

    def main(self):
        cleaned_file = self.datahandler.clean_filter_and_save()
        semivariogram_obj = EmpiricalSemivariogram(cleaned_file)

        semivariogram_obj.construct_location_pairs  # This updates self.location_pairs
        semivariogram_obj.construct_GMM()  # This updates intial_PGA
        semivariogram_obj.outliers()  # This updates outlier_treated_PGA, self.updated_location_pairs
        semivariogram_obj.anisotropy()  # Updates PGA information once again

        semivariogram_obj.compute_empirical_semivariogram() 
       
        cov_model = semivariogram_obj.choose_covariance_model() 

        krig = OrdinaryKriging(cov_model)
        
        LOOCV_result = krig.LOO_cross_validation
        
        # Add a print statement for the LOOCV result.
        return LOOCV_result

    
        
