# Filename: semivariogram.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script contains classes and methods for 
#               computing and plotting the empirical semivariogram.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from common.base import Plotting, Preprocessing
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import yaml
#import math
import itertools
from pyproj import Transformer
import matplotlib.pyplot as plt  # Is required for the pyGMM install 
import pygmm  # Ground motion model package
# from scipy.spatial.distance import pdist, squareform


config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"

with open(config_path, "r") as f:
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
        self.PGA_predicted: dict[tuple[int, int], list[float, float]] = {}
        
        self.initial_PGA: dict[str, float] = {}
        self.residuals_sum = []
        self.outlier_treated_PGA: dict[str, float] = {}
        self.anisotropy_treated_PGA: dict[str, float] = {}

        self.location_pairs: list[tuple[str]] = []
        self.station_coords = None
        # This is a dict that looks like:  
        # (stationpair tuple): distance
        self.pairwise_distances: dict[tuple[str, str], float] = {}
        self.station_ids = []

        self.semivariogram = []  # This is the experimental variogram referenced in literature.
        self.station_variance: dict[tuple[str, str], list[float, float]] = {}

        self.cov_model = None
        self.C = None

    # @Brief: This function will take the cleaned dataframe and return location pairs and distances.
    # I enter this function with only the dataframe filtered to our target earthquake. 
    # I will determine each station's distance from the earthquake (and their distances from one another)
    # and store that information. Based on the lag interval and bin width set in the config file, I may 
    # then construct my location pairs.
   
    def construct_location_pairs(self) -> list[tuple[int, int]]:
        # Firstly, I'll create a dictionary of information for each station.
        station_zip: list[tuple[int, str, float, float, float]] = list(
            zip(self.data_object['station_id__no.'], self.data_object['station_name'], 
                self.data_object['station_latitude'], self.data_object['station_longitude'], 
                self.data_object['pga_(g)'], self.data_object['epid_(km)']))

        station_information: dict[int, tuple[str, float, float, float, float]] = {}
        
        for entry in station_zip:
            station_information[entry[0]] = entry[1:]
        
        # Next, I will group all station pairs. There should be no self pairs
        self.station_ids = list(self.data_object['station_id__no.'])
        for station_1, station_2 in itertools.combinations(self.station_ids, 2):
            self.location_pairs.append((station_1, station_2))

        # I will also create pairwise distances
        # I'm admittedly not very familiar with these sorts of distance calculations, so this
        # next pairwise distance section here is moreso copy paste/tutorial at the moment.
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
 
        station_coords: dict[int, tuple[float, float]] = {}
        for station_id, info in station_information.items():
            # TODO: Need to update this.
            # info = (station_name, station_latitude, station_longitude, pga_(g), epid_(km))
            lat, lon = info[1], info[2]
            easting, northing = transformer.transform(lon, lat)
            station_coords[station_id] = (easting, northing)
 
        def pairwise_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
            return float(np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2))
       
        for station1, station2 in self.location_pairs:
            distance = pairwise_distance(station_coords[station1], station_coords[station2])
            self.pairwise_distances[(station1, station2)] = distance
        print(f"Some location Pairs: {self.location_pairs[:5]}")
        self.station_coords = station_coords
        return self.station_coords

    def construct_GMM(self) -> dict[str, float]:
        # I chose to use the ChiouYoungs2014 model based on a quick trade-study-esque process
        station_params = {}
        vs30 = 'vs30_(m/s)_selected_for_analysis'
        for station in self.station_ids: 
            station_params[station] = [
                self.data_object.loc[self.data_object['station_id__no.'] == station, 'earthquake_magnitude'].iloc[0], 
                self.data_object.loc[self.data_object['station_id__no.'] == station, 'joyner-boore_dist._(km)'].iloc[0], 
                self.data_object.loc[self.data_object['station_id__no.'] == station, 'rx'].iloc[0], 
                # distance from rupture plane chosen as ClstD_(km), i assumed this was what it meant
                self.data_object.loc[self.data_object['station_id__no.'] == station, 'clstd_(km)'].iloc[0],   
                self.data_object.loc[self.data_object['station_id__no.'] == station, 'dip_(deg)'].iloc[0], 
                self.data_object.loc[self.data_object['station_id__no.'] == station, vs30].iloc[0]
                ]
       
        # Save one fitted model per station, since the same station appears in multiple pairs
        # and the scenario/model construction is otherwise repeated unnecessarily. This fixes the todo.
        station_models: dict[str, pygmm.ChiouYoungs2014] = {}

        def get_model(station):
            if station not in station_models:
                scenario = pygmm.Scenario(  # See the docs. PyGMM now takes a single 'scenario' parameter 
                    mag=station_params[station][0],
                    dist_jb=station_params[station][1],
                    dist_x=station_params[station][2],
                    dist_rup=station_params[station][3],
                    dip=station_params[station][4],
                    v_s30=station_params[station][5])
                station_models[station] = pygmm.ChiouYoungs2014(scenario)
            return station_models[station]

        log_station_std = []
        for station1, station2 in self.location_pairs:
            model1 = get_model(station1)
            model2 = get_model(station2)
            self.initial_PGA[(station1, station2)] = [model1.pga, model2.pga]
            # I use a list instead of a tuple because I don't think ndarray will handl tuples properly.
            log_station_std.append([model1.ln_std_pga, model2.ln_std_pga])

        print("\n\nCOMPLETED RUNNING GMM MODEL.")

        log_station_var = np.array(log_station_std) ** 2
        
        for index, (station1, station2) in enumerate(self.location_pairs):
            self.station_variance[(station1, station2)] = log_station_var[index]
            '''
            the enumerate transforms the list to this format:
            0 [var1, var2]
            1 [var1, var2]
            2 [var1, var2]
            So i can reference each pair by its index to get
            0 [var1^2, var2^2], etc.
            '''

        print(f"Initial PGA Keys:\n{list(self.initial_PGA.keys())[:4]}")
        print(f"Initial PGA:\n{list(self.initial_PGA.values())[:4]}")
        return self.initial_PGA  

    def sample_size(self):
        # This will not be checked in v1
        pass
        
    def lag_interval_and_bin_width(self):
        pass

    def marginal_distribution(self):
        # I will not be implementing this one in V1, 
        # though i leave this here for future improvements.
        pass

    def outliers(self) -> tuple[dict[str, float], list[tuple[str, str]]]:
        # It looks odd, but the PRIOR location pairs are used to UPDATED
        self.outlier_treated_PGA, self.location_pairs = Preprocessing.detect_outliers(
            self.data_object, self.location_pairs, self.initial_PGA
            )
        # NOTE: This is already a function in the preprocessing class. 
        # However, I might be moving the code here instead and getting rid of 
        # the preprocessing class. Need to think of downstream effects of this though.
        return self.outlier_treated_PGA, self.location_pairs

    def anisotropy(self):
        # Be sure to use outlier_treated_PGA
        # though the semivariograms in MY literature are assumed (and proven) to be isotropic
        self.anisotropy_treated_PGA = self.outlier_treated_PGA  # Placeholder
        return self.anisotropy_treated_PGA

    def trend(self):
        # This will also not be checked in v1
        pass

    def compute_empirical_semivariogram(self):
        # Be sure to use anisotropy_treated_PGA values
        # Bins are centered at multiples of self.lag_interval (h), each with
        # half-width self.bin_width / 2 (δh/2), per Baker eq. 4 / bin definition.

        # Assign each pair to the nearest lag center, if it falls within that bin.
        lag_groups: dict[float, list[tuple[str, str]]] = {}
        half_width = self.bin_width / 2

        for station_1, station_2 in self.location_pairs:
            distance = self.pairwise_distances[(station_1, station_2)]

            # how many lag intervals fit into this distance
            nearest_multiple = round(distance / self.lag_interval)
            # assign this pair to a lag distance by snapping to the rounded value
            lag_center = nearest_multiple * self.lag_interval  # will = 2 in config. from lit.

            # if the station is within specified width
            if abs(distance - lag_center) <= half_width:
                ''' What setdefault does:
                if lag_center not in lag_groups:
                    lag_groups[lag_center] = []
                    lag_groups[lag_center].append((station_1, station_2))
                '''
                lag_groups.setdefault(lag_center, []).append((station_1, station_2))
            # else: distance falls in the gap between bins (possible when bin_width < lag_interval)
            # and is simply not used in any semivariogram point, per the paper's definition.

        # Step 2: compute semivari(h) for each lag bin independently.
        self.semivariogram = []
        for lag_distance, location_pairs_lagged in sorted(lag_groups.items()):
            residuals_sum = []  # local to this lag, not self.residuals_sum
            for station_1, station_2 in location_pairs_lagged:
                log_PGA_predicted_station1 = np.log(self.anisotropy_treated_PGA[station_1])
                log_PGA_predicted_station2 = np.log(self.anisotropy_treated_PGA[station_2])
                
                pair_variance = self.station_variance[(station_1, station_2)]
                residuals_sum.append((
                    (self.log_PGA_trues[station_1] - log_PGA_predicted_station1) / pair_variance[0],
                    (self.log_PGA_trues[station_2] - log_PGA_predicted_station2) / pair_variance[1]
                ))

            sqrd_differences = [
                (residual_1 - residual_2) ** 2
                for residual_1, residual_2 in residuals_sum
            ]

            # Baker eq. 4: semivari(h) = 1 / (2*N(h)) * sum[z_u - z_u+h]^2
            N = len(location_pairs_lagged)  # number of pairs AT THIS LAG DISTANCE
            if N == 0:
                continue
            sum_sqrd_differences = sum(sqrd_differences)
            gamma_h = (1 / (2 * N)) * sum_sqrd_differences

            self.semivariogram.append((lag_distance, gamma_h, N))

        return self.semivariogram

    def sill_and_range(self):  # Section admittedly needs more study work. 
        lags = np.array([point[0] for point in self.semivariogram])
        gammas = np.array([point[1] for point in self.semivariogram])
        Ns = np.array([point[2] for point in self.semivariogram])

        weights = Ns / lags**2          # Cressie-style WLS weights
        sigma = 1 / np.sqrt(weights)    # curve_fit wants sigma, not weight, so invert

        # initial guesses: a ~ sill ~ variance of Z_u; b ~ range ~ some fraction of max lag
        # a0: rough guess at Var(Z_u), used to seed the sill.
        # b0: rough guess at decay distance, used to seed the range before curve fit.
        a0 = np.var(list(self.anisotropy_treated_PGA.values()))
        b0 = lags.max() / 3

        (a_fit, b_fit), _ = curve_fit(
            self.exponential_model, lags, gammas,
            p0=[a0, b0], sigma=sigma, absolute_sigma=False
        )

        self.cov_model = {'a': a_fit, 'b': b_fit}
        return a_fit, b_fit

    # semivariogram functional forms to create "a CONTINUOUS function... fitted [on the finite]
    # experimental values in order to deduce variogram values for any possible separation h."
    # The literature notes that the best choice can be chosen visually.
    # For now, I will start with one model only.

    def exponential_model(self, h, a, b):
        # ISOTROPIC CASE:
        # semivari(h) = a[1 - exp(-3h/b)]
        # a - sill (also known as the variance of Z_u, aka station 1 (NOT Z_u+h))
        # b - range (also the h at which semivari = .95 times sill of exponential semivari)
        return a * (1 - np.exp(-3 * h / b))

    def power_empirical_model(self):
        # Will be skipped.
        pass
    
    def spherical_model(self):
        # Used for isotropic case (direction has no effect on semivari)
        #Will be skipped for now.
        pass
    # methods for fitting a covariance model to the semivariogram via WLS.
    
    # TODO: Think about how i will need to eventually restructure main.py. Should
    # I even have a choose_cov_model separate from teh sill_and_range() functions?
    # Maybe sillandrange should instead be called choose_cov_model? 
    # And how will future users have optionality in the config in terms of choosing the 
    # functional form of the model? Perhaps I need a structure where the config also allows
    # the functional form to be chosen, not just lag distance and bin width!!

    def choose_covariance_model(self, semivar_values):
        # "The covariance structure of Z (location realizations) is completely specified by the 
        # semivariogram function and the sill and the range of the semivariogram." Baker Pg 8
        # semivari = a(1 - p(h))
        # a - sill
        # p(h) - correlation coefficient between Z_u and Z_u+h (location 1 and location 2) 
        # Note, the correlation coefficient, p(h) = COV(Z_u , Z_u+h) / (sqrt(VAR(Z_u))*sqrt(VAR(Z_u+h)))
        # So to isolate the covariance model for pairs at this lag, isolate COV
        # COV = p(h) * (sqrt(VAR(Z_u)) * sqrt(VAR(Z_u+h)))
        # Recal VAR(Z_u) is a (as I defined it before).
        # Under second order stationarity (intrinsic stationarity) (which the text assumes), 
        # VAR doesn't depend on location
        # We have COV = p(h) * (sqrt(a) * sqrt(a))
        # We have COV = p(h) * a

        # 1 - (semivari / a) = p(h)
        # a(1 - (semivari / a)) = COV
        if self.cov_model is None:
            self.sill_and_range()

        a = self.cov_model['a']
        b = self.cov_model['b']

        n = len(self.station_ids)
        C = np.zeros((n, n))

        for i, station_i in enumerate(self.station_ids):
            for j, station_j in enumerate(self.station_ids):
                if i == j:
                    C[i, j] = a  # h = 0 -> Cov = a
                    continue
                
                # Determining if this pair ordering is in the dict.
                pair = (station_i, station_j) if (station_i, station_j) in self.pairwise_distances \
                    else (station_j, station_i) 
                h = self.pairwise_distances[pair]  # If not, try reverse order
                C[i, j] = a * np.exp(-3 * h / b)

        self.cov_model['C'] = C
        self.C = C
        print(f"C, the cov model:\n{C}")
        return C
      

class PlotEmpiricalSemivariogram:  # I will figure this out later.
    def __init__(self, data_object):
        self.data_object = data_object
        self.passed_to_plotting_class = None

    def plot_empirical_semivariogram(self):
        #self.passed_to_plotting_class = some operation
        Plotting(self.passed_to_plotting_class).plot_passed_data()
        # This fig, ax thing will be deleted. 
        # It is to silence the flake8 linting concerns about plt not being used.
        fig, ax = plt.subplots() 
        print("Empirical Semivariogram plotted. Check output folder > graphics for the graph.")
        pass
