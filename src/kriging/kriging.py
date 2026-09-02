# Filename: kriging.py
# Author : Chelsea Momoh
# Date : 2026-08-06
# Version : 1.0
# Description : This script contains classes and methods for 
#               implementing ordinary kriging and Leave-One-Out-Cross-Validation.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import LeaveOneOut
import numpy as np

class OrdKriging:
    def __init__(self, a, b, station_coords, PGA_values):
        self.a = a
        self.b = b
        self.station_coords = station_coords  
        self.PGA_values = PGA_values  

    # Now that I've decided to use the PyKrige package, I no longer will build 
    # kriging_neighborhood() and block_kriging() from the literature. (Thank goodness) 

    def punctual_kriging(self, target_x, target_y):
        station_ids = list(self.station_coords.keys())
        x = [self.station_coords[sid][0] for sid in station_ids]
        y = [self.station_coords[sid][1] for sid in station_ids]
        z = [self.PGA_values[sid] for sid in station_ids]

        OK = OrdinaryKriging(
            x, y, z,
            variogram_model="exponential",
            variogram_parameters={'sill': self.a, 'range': self.b, 'nugget': 0},
            verbose=False,
            enable_plotting=False
        )

        z_pred, ss_pred = OK.execute("points", target_x, target_y)
        return z_pred, ss_pred

    def LOO_cross_validation(self):
        # Will need to perform punctual kriging here.
        station_ids = list(self.station_coords.keys())
        x = np.array([self.station_coords[sid][0] for sid in station_ids])
        y = np.array([self.station_coords[sid][1] for sid in station_ids])
        z = np.array([self.PGA_values[sid] for sid in station_ids])
        loo = LeaveOneOut()
        predictions = []
        actuals = []

        for train_idx, test_idx in loo.split(x):
            x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]
            x_test, y_test, z_test = x[test_idx], y[test_idx], z[test_idx]

            OK = OrdinaryKriging(x_train, y_train, z_train,
                                variogram_model="exponential",
                                variogram_parameters={'sill': self.a, 'range': self.b, 'nugget': 0})
            z_pred, ss_pred = OK.execute("points", x_test, y_test)

            predictions.append(z_pred[0])
            actuals.append(z_test[0])
        return predictions, actuals

# Kriging Docs: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html
