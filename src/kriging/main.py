# Filename: main.py
# Author : Chelsea Momoh
# Date : 2026-08-03
# Version : 1.0
# Description : This script will run each step of the kriging process.
# * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -

from pathlib import Path
from data_scripts.data_handling import DataHandler
from kriging.semivariogram import EmpiricalSemivariogram
from kriging.kriging import OrdKriging, PlotKrigingResults
import csv

class Main:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.datahandler = DataHandler(self.filepath)

    def main(self):
        cleaned_file = self.datahandler.clean_filter_and_save()
        semivariogram_obj = EmpiricalSemivariogram(cleaned_file)

        # This also updates self.location_pairs
        station_coords = semivariogram_obj.construct_location_pairs()

        # TODO: Later, I will ensure I can have multiple objects, 
        # likely one for each semivariogram model. This means I'll need to 
        # either pass in the model type I want or set it up in the config.
        # Right now I'm only constructing the exponential model.
        semivariogram_obj.construct_GMM()  # This updates initial_PGA
        
        # These 2 methods may not be removed, as the code depends on the values of these
        # functions which update in this expected order.
        # This updates outlier_treated_PGA
        outlier_treated_PGA, _ = semivariogram_obj.outliers()      
        
        semivariogram_obj.anisotropy()  # Updates PGA information once again
        # initially i erroneously kriged the PGA values rather than the residuals the 
        # semivariogram is built upon.
        residuals = semivariogram_obj.compute_station_residuals()

        # I also need to update the location coordinates (not just location pairs) 
        station_coords = {sid: coord for sid, coord in station_coords.items() if sid in residuals}

        semivariogram_obj.compute_empirical_semivariogram()

        # Fit the exponential model's sill (a) and range (b) via WLS.
        a_fit, b_fit = semivariogram_obj.sill_and_range()

        # choose_covariance_model() (the hand-built C matrix) isn't needed here since
        # PyKrige builds its own internal covariance from a_fit/b_fit directly.
        # cov_model = semivariogram_obj.choose_covariance_model()

        krig = OrdKriging(a_fit, b_fit, station_coords, residuals)

        predictions, actuals, variances = krig.LOO_cross_validation()

        results = list(zip(predictions, actuals, variances))
        #print(f"LOOCV Result [PREDICTION | ACTUAL | VARIANCE]:\n{results}")
        print(f"Length of results: {len(results)}")
        output_csv_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "loocv_results.csv"
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prediction", "actual", "variance"])
            writer.writerows(results)
        LOOCV_message = f"LOOCV results saved to {output_csv_path}"

        plotter = PlotKrigingResults(predictions, actuals, variances)
        plotter.plot_loocv_calibration()
        plot_message = "LOOCV calibration plot saved to outputs/loocv_calibration_plot.png"
        return LOOCV_message, plot_message

def run():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "original.csv"
    runner = Main(filepath=str(data_path))
    return runner.main()  # returns predictions, actuals, & variances


if __name__ == "__main__":
    run()
