# Filename: kriging.py
# Author : Chelsea Momoh
# Date : 2026-08-06
# Version : 1.0
# Description : This script contains classes and methods for 
#               implementing ordinary kriging and Leave-One-Out-Cross-Validation.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from pykrige.ok import OrdinaryKriging


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
        # Add a print statement or diagram for the LOOCV result.
        pass

# Kriging Docs: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html
