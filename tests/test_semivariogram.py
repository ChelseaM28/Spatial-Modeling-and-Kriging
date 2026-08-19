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

data_path = Path(__file__).resolve().parent.parent / "data" / "original.csv"
columns = ['earthquake_name', 'station_name', 'station_id__no.', 
            'station_latitude', 'station_longitude', 'joyner-boore_dist._(km)',
            'rx', 'dip_(deg)', 'earthquake_magnitude', 'magnitude_type',
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

def test_construct_location_pairs():
    location_checker = SemivariogramCheck()
    assert len(location_checker.construct_location_pairs()) > 4
    assert len(location_checker.construct_GMM()) > 4

'''def test_construct_GMM():
    gmm_checker = SemivariogramCheck()
    initial_PGA_keys = gmm_checker.construct_GMM()
    assert len(list(initial_PGA_keys.keys())) > 4'''


