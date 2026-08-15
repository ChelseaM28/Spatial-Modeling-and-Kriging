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

import unittest
from kriging.semivariogram import EmpiricalSemivariogram
from data_scripts.data_handling import DataHandler

data_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "original.csv"

class TestDataHandling(DataHandler):
    def test_test_clean_data(self):
        obj = DataHandler(filepath)
        
        columns = ['earthquake_name', 'station_name', 'station_id_no.', 
            'station_latitude', 'station_longitude', 'joyner-boore_dist._(km)',
            'rx', 'dip_(deg)', 'earthquake_magnitude', 'magnitude_type',
            'vs30_(m/s)_selected_for_analysis', 'epid_(km)', 'pga_(g)']
        
        columns_present = columns in obj.tested_cleaned_data().columns()
        assert columns_present

data_tester = TestDataHandling(data_path)

data_tester.test_test_clean_data()

'''class test_Semivariogram(EmpiricalSemivariogram):
    def test_construct_location_pairs():
        pass

    def test_construct_GMM():
        pass
'''
def test_print_statement():
    test_finished = True
    assert test_finished
