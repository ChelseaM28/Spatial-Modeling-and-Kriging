# Filename : data_load.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script contains the data class for loading and cleaning data.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

import pandas as pd

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataframe = pd.read_csv(filepath)
        self.mini_dataframe = self.dataframe.head(10)
        self.filtered = None
        self.output_path = "/workspaces/Spatial-Geostatistics-Analysis/data/raw/processed"

    def clean_data(self, dataframe):
        # Figure how to handle missing data and data types. No outlier detection in this step.
        self.dataframe = self.dataframe.dropna()
        self.dataframe.columns = self.dataframe.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)   
        self.dataframe = self.dataframe.astype(
            {'earthquake_name': str, 'station_name': str, 'station_id_no.': str, 
            'station_latitude': float, 'station_longitude': float,
            'epid_(km)': float, 'pga_(g)': float}
            )  # EpiD is 'epicenter distance.' 
        # I believe these are all the columns I need, though I may update it.
        return self.dataframe

    def test_clean_data(self):  # I should probably somehow write this into my pytest instead. 
        self.tested_cleaned_data = self.clean_data(self.mini_dataframe)
        # This will print, in terminal, about 10 rows of the dataframe after a test cleaning process.
        print(f"Cleaned First Ten Rows of Dataset. \noriginal dataset is unmodified:\n{self.tested_cleaned_data}")
    
    def save_cleaned_to_json(self):
        self.dataframe.to_json(self.output_path + "/cleaned_data.json", orient='records', lines=True)
    
    # @Brief: This function will find the earthquake that has been captured by the most stations.
    # It will return the earthquake ID and the number of stations reporting data.
    def filter_to_earthquake(self):
        counts = self.dataframe.groupby('earthquake_name')['station_id_no.'].count()
        # If multiple values share the maximum count, idxmax() returns only the first occurrence.
        # So to ensure reproducibility, I sort the indices.
        best_earthquake = counts.sort_index().idxmax()  # I use station IDs because they are simple to parse and unique.
        self.dataframe = self.dataframe[(self.dataframe['earthquake_name'] == str(best_earthquake))]
        self.filtered = self.dataframe
        return self.filtered

    def save_filtered_to_json(self):
        self.filtered.to_json(self.output_path + "/filtered_data.json", orient='records', lines=True)

    def clean_filter_and_save(self):
        self.dataframe = self.clean_data(self.dataframe)
        self.filtered = self.filter_to_earthquake()
        self.save_filtered_to_json()
        return self.filtered
