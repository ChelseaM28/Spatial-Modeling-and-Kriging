# Filename: test_semivariogram.py
# Author : Chelsea Momoh
# Date : 2026-08-05
# Version : 1.0
# Description : Test script for empirical semivariogram.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

# to lint locally:
# conda activate project-environment
# flake8 src
# to test: 
# pytest
# pytest -s

import unittest
from pathlib import Path
from kriging.semivariogram import EmpiricalSemivariogram
from data_scripts.data_handling import DataHandler

# ========= Data Cleaning Testing ========

# TODO: Need to restructure to determine which individual asserts are passing/failing.

data_path = Path(__file__).resolve().parent.parent / "data" / "original.csv"
columns = ['earthquake_name', 'station_name', 'station_id__no.', 
            'station_latitude', 'station_longitude', 'joyner-boore_dist._(km)',
            'rx', 'clstd_(km)', 'dip_(deg)', 'earthquake_magnitude', 'magnitude_type',
            'vs30_(m/s)_selected_for_analysis', 'epid_(km)', 'pga_(g)'] 

class DataHandlingCheck(DataHandler):
    def __init__(self):
        super().__init__(data_path)  

def test_clean_data_has_expected_columns():
    checker = DataHandlingCheck()
    columns_present = set(columns).issubset(checker.test_clean_data().columns)
    assert columns_present


# ========= Semivariogram Testing =========

data_obj = DataHandlingCheck()
clean_data = data_obj.test_clean_data()

class SemivariogramCheck(EmpiricalSemivariogram):
    def __init__(self):
        super().__init__(clean_data) 

def test_semivariogram():
    test_obj = SemivariogramCheck()
    assert len(test_obj.construct_location_pairs()) > 4
    # or maybe test for no negative PGA values or some other value checker
    assert len(test_obj.construct_GMM()) > 4
    
    outlier_treated_PGA, location_pairs = test_obj.outliers()
    surviving_stations = set(outlier_treated_PGA.keys())
    pairs_stations = set(s for pair in location_pairs for s in pair)
    assert pairs_stations.issubset(surviving_stations)
    
    pga_values_tested = test_obj.anisotropy()
    # Once this function works, will need to test it here
    
    assert len(test_obj.compute_empirical_semivariogram()) > 4

