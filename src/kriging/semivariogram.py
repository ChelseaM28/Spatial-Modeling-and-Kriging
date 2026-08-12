# Filename: semivariogram.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script contains classes and methods for 
#               computing and plotting the empirical semivariogram.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from common.base import Plotting, Preprocessing
import numpy as np
import yaml
import itertools
from pyproj import Transformer
from scipy.spatial.distance import pdist, squareform

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# I am using peak ground acceleration as my parameter of interest with the 
# assumption that professionals in industry would find this metric more useful.
# It is "a natural simple design parameter since it can be related 
# to a force and for simple design" - USGS Earthquake Hazards 201

# Not all of these methods will be performed, as the scope of the project will need to be limited.   
class EmpiricalSemivariogram:
    def __init__(self, filtered_data_object): 
        self.data_object = filtered_data_object  # Filtered to chosen earhquake only + outliers removed.

        self.lag_interval = config['lag_interval']
        self.bin_width = config['bin_width']

        self.PGA_true: dict[str, float] = {} 
        self.log_PGA_trues: dict[str, float] = {}
        
        self.initial_PGA: dict[str, float] = {}
        self.residuals_sum = []
        self.outlier_treated_PGA: dict[str, float] = {}
        self.anisotropy_treated_PGA: dict[str, float] = {}

        self.location_pairs: list[tuple[int, int]] = []
        # This is a dict that looks like:  
        # (stationpair tuple): distance
        self.pairwise_distances: dict[tuple[int,int],float] = {}

        self.semivariogram = []  # This is the experimental variogram referenced in literature.
        self.station_variance = {}

        self.cov_model = None


    # @Brief: This function will take the cleaned dataframe and return location pairs and distances.
    # I enter this function with only the dataframe filtered to our target earthquake. 
    # To determine my station pairs, I need to know their lag distances. I will determine each station's 
    # distance from the earthquake (and their distances from one another) and store that information.
    # Based on the lag interval and bin width set in the config file, I may then construct my location pairs.
    def construct_location_pairs(self) -> list[tuple[int, int]]:

        # Firstly, I'll create a dictionary of information for each station.
        station_zip: list[tuple[int, str, float, float, float]] = list(
        zip(self.data_object['station_id_no'], self.data_object['station_name'], 
        self.data_object['station_latitude'],self.data_object['station_longitude'], 
        self.data_object['pga_(g)'], self.data_object['epid_(km)']))

        station_information: dict[int, tuple[str, float, float, float, float]] = {}
        
        for entry in station_zip:
            station_information[entry[0]] = entry[1:]
        
        # Next, I will group all station pairs. There should be no self pairs
        station_ids = self.data_object['station_id_no']
        for station_1, station_2 in itertools.combinations(station_ids, 2):
            self.location_pairs.append((station_1, station_2))

        # I will also create pairwise distances
        # I'm admittedly not very familiar with these sorts of distance calculations, so this
        # next pairwise distance section here is moreso copy paste at the moment.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
 
        station_coords: dict[int, tuple[float, float]] = {}
        for station_id, info in station_information.items():
            # info = (station_name, station_latitude, station_longitude, pga_(g), epid_(km))
            lat, lon = info[1], info[2]
            easting, northing = transformer.transform(lon, lat)
            station_coords[station_id] = (easting, northing)
 
        def pairwise_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
            return float(np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2))
       
        for station1, station2 in self.location_pairs:
            distance = pairwise_distance(station_coords[station1], station_coords[station2])
            self.pairwise_distances[(station1, station2)] = distance
        return self.location_pairs

    def construct_GMM(self) -> dict[str, float]:
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

    def outliers(self) -> tuple[dict[str, float], list[tuple[str, str]]]:
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
