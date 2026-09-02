# Filename : base.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script contains different scripts common 
# to the kriging and tail extremes processes.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

import numpy as np

# Preporcessing's data object should be a filtered dataframe including only the chosen earthquake.
class Preprocessing:
    def __init__(self, filtered_data_object, location_pairs, GMM_predictions):
        self.dataframe = filtered_data_object
        self.site_pairs = location_pairs
        self.site_pair_lag_distances = {}
        self.GMM_predictions = []

    def detect_outliers(self):
        # I will not use a box plot/histogram to CONFIRM outliers. 
        # Such tools should only be used as visual aids. Instead, I will perform an IQR or Z-Score test.
        # Calculate Q1, Q3, and IQR
        Q1 = np.percentile(self.GMM_predictions, 25)
        Q3 = np.percentile(self.GMM_predictions, 75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Detect outliers
        outliers = self.GMM_predictions[(self.GMM_predictions < lower_bound) | (self.GMM_predictions > upper_bound)]
        print(f"Outliers based on IQR test: {outliers}.\nRemoving...")
        # I need to decide HOW to decide whether to remove outliers. 
        # I can remove them by default for now, though 'remove by default' is not best practice.
        self.GMM_predictions = self.GMM_predictions.pop(outliers)
        # Unfortunately, the data structure is likely incorrect. Lists.. dicts... loops. Not correct yet.
        
        location_pairs = None  # Will need to update the location pairs
        return self.GMM_predictions, location_pairs  



