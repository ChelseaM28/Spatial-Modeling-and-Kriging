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
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

class OrdKriging:
    def __init__(self, a, b, station_coords, PGA_values):
        self.sill = a
        self.range = b
        self.station_coords = station_coords  
        self.PGA_values = PGA_values  

    # Now that I've decided to use the PyKrige package, I no longer will build 
    # kriging_neighborhood() and block_kriging() from the literature. (Thank goodness) 

    def punctual_kriging(self, target_x, target_y):
        print('\n\nPerforming punctual kriging...')
        station_ids = list(self.station_coords.keys())
        x = [self.station_coords[sid][0] for sid in station_ids]
        y = [self.station_coords[sid][1] for sid in station_ids]
        z = [self.PGA_values[sid] for sid in station_ids]

        OK = OrdinaryKriging(
            x, y, z,
            variogram_model="exponential",
            variogram_parameters={'sill': self.sill, 'range': self.range, 'nugget': 0},
            verbose=False,
            enable_plotting=False
        )
        print(f"sill: {self.sill}, range: {self.range}")
        z_pred, ss_pred = OK.execute("points", target_x, target_y)
        return z_pred, ss_pred

    def LOO_cross_validation(self):
        print('\n\nPerforming Leave-One-Out Cross-Validation...')
        # Will need to perform punctual kriging here.
        station_ids = list(self.station_coords.keys())
        x = np.array([self.station_coords[sid][0] for sid in station_ids])
        y = np.array([self.station_coords[sid][1] for sid in station_ids])
        z = np.array([self.PGA_values[sid] for sid in station_ids])
        loo = LeaveOneOut()
        predictions = []
        actuals = []
        variances = []

        for train_idx, test_idx in loo.split(x):
            x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]
            x_test, y_test, z_test = x[test_idx], y[test_idx], z[test_idx]

            OK = OrdinaryKriging(x_train, y_train, z_train,
                                variogram_model="exponential",
                                # I need to work out the nugget
                                variogram_parameters={'sill': self.sill, 'range': self.range, 'nugget': 0})
            z_pred, ss_pred = OK.execute("points", x_test, y_test)

            predictions.append(z_pred[0])
            actuals.append(z_test[0])
            variances.append(ss_pred[0])
        return predictions, actuals, variances


class PlotKrigingResults:
    def __init__(self, predictions, actuals, variances):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.variances = np.array(variances)
        self.std = np.sqrt(self.variances)
        # Standardized (studentized) residual: (actual - prediction) / sqrt(kriging variance)
        self.z = (self.actuals - self.predictions) / self.std

    def plot_loocv_calibration(self, output_dir=None):
        print("\n\nPlotting LOOCV calibration diagnostics...")
        pred, actual, std, z = self.predictions, self.actuals, self.std, self.z

        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        corr = np.corrcoef(pred, actual)[0, 1]
        z_mean, z_std = np.mean(z), np.std(z, ddof=1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Plot 1: Predicted vs Actual, with 1:1 line and 95% CI error bars
        ax = axes[0]
        lims = [min(pred.min(), actual.min()) - 0.5, max(pred.max(), actual.max()) + 0.5]
        ax.plot(lims, lims, "k--", lw=1.5, label="1:1 (perfect prediction)")
        ax.errorbar(actual, pred, yerr=1.96 * std, fmt="o", ecolor="gray",
                    elinewidth=1, capsize=3, alpha=0.7, color="#2b6cb0",
                    label="LOOCV prediction ±95% CI")
        ax.set_xlabel("Actual (standardized log-residual)")
        ax.set_ylabel("Kriged prediction")
        ax.set_title(f"Predicted vs. Actual\nRMSE = {rmse:.3f}, r = {corr:.3f}")
        ax.legend(fontsize=8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        # Plot 2: standardized residual histogram vs N(0,1)
        ax = axes[1]
        x_grid = np.linspace(-4, 4, 200)
        ax.hist(z, bins=8, density=True, alpha=0.6, color="#2b6cb0",
                edgecolor="white", label="Standardized residuals")
        ax.plot(x_grid, stats.norm.pdf(x_grid, 0, 1), "k-", lw=2, label="N(0,1)")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("(actual - prediction) / sqrt(variance)")
        ax.set_ylabel("Density")
        ax.set_title(f"Calibration Check\nmean={z_mean:.2f}, std={z_std:.2f} (ideal: 0, 1)")
        ax.legend(fontsize=8)

        # Plot 3: Q-Q plot of standardized residuals vs Normal
        ax = axes[2]
        stats.probplot(z, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot: Standardized Residuals vs. Normal")
        ax.get_lines()[0].set_markerfacecolor("#2b6cb0")
        ax.get_lines()[0].set_markeredgecolor("#2b6cb0")
        ax.get_lines()[1].set_color("black")

        plt.tight_layout()

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "graphics"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "loocv_calibration.png"
        plt.savefig(output_path, dpi=200)
        plt.close(fig)

        print(f"LOOCV calibration plot saved to {output_path}")
        print(f"RMSE={rmse:.4f}, corr={corr:.4f}, z_mean={z_mean:.4f}, z_std={z_std:.4f}")
        return output_path

# Kriging Docs: https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html
