# This class defines key analytical routines for calculating wake losses for an operating
# wind plant using SCADA data. For each SCADA time step, freestream wind turbines are
# identified using the turbine coordinates and a reference wind direction signal. The mean
# power production for all turbines in the wind plant is summed over all time steps and
# compared to the mean power of the freestream turbines summed over all time steps to
# estimate wake losses during the period of record. Methods for calclating the long-term
# wake losses using reanalaysis data and quantifying uncertaitny are provided as well.

import numpy as np
import pandas as pd

from operational_analysis import logging, logged_method_call
from operational_analysis.types import asset


logger = logging.getLogger(__name__)


class WakeLosses(object):
    """
    A serial (Pandas-driven) implementation of a method for estimating wake losses from SCADA data.

    TODO: Add more details.
    """

    @logged_method_call
    def __init__(self, plant, wind_direction_col, wind_direction_turbine_ids=None):
        """
        Initialize wake loss analysis object with data and parameters.

        Args:
            plant (:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data
            wind_direction_col (:obj:`string`): SCADA column to use for wind direction
            wind_direction_turbine_ids (:obj:`list`, optional): List of turbine IDs used to calculate the average wind direction at each time step. If None, all turbines will be used. Defaults to None.
        """
        logger.info("Initializing WakeLosses analysis object")

        self._plant = plant
        self._plant.asset.prepare()
        self._wind_direction_col = wind_direction_col

        if wind_direction_turbine_ids is not None:
            self._wind_direction_turbine_ids = wind_direction_turbine_ids
        else:
            self._wind_direction_turbine_ids = list(self._plant.scada.df.id.unique())

        # Run preprocessing steps
        self._calculate_aggregate_dataframe()

    @logged_method_call
    def _calculate_aggregate_dataframe(self):
        """
        Creates a data frame with relevant scada columns and plant-level columns to be used for the wake loss analysis.
        The reference mean wind direction is then added to the data frame.

        Args:
            (None)

        Returns:
            (None)
        """

        # keep relevant SCADA columns, create a unique time index and two-level turbine variable columns
        # (variable name and turbine ID)
        self._aggregate_df = (
            self._plant.scada.df[["id", "wmet_wdspd_avg", self._wind_direction_col, "energy_kwh"]]
            .reset_index()
            .set_index(["time", "id"])
            .unstack()
        )

        # remove times with any missing turbine IDs or data
        # TODO: revisit because this may remove too many samples
        self._aggregate_df = self._aggregate_df.dropna(how="any")

        # Calculate reference mean wind direction
        self._calculate_mean_wind_direction()

        # Drop turbine-level wind direction column
        self._aggregate_df = self._aggregate_df.drop(columns=[self._wind_direction_col])

    @logged_method_call
    def _calculate_mean_wind_direction(self):
        """
        Calculates the mean wind direction at each time step using the specified SCADA wind direction column for the
        specified subset of turbines. This reference mean wind direction is added to the plant-level data frame.

        Args:
            (None)
        Returns:
            (None)
        """

        self._aggregate_df["wmet_HorWdDir_ref"] = (
            np.degrees(
                np.arctan2(
                    np.sin(
                        np.radians(
                            self._aggregate_df["wmet_HorWdDir_avg"][
                                self._wind_direction_turbine_ids
                            ]
                        )
                    ).mean(axis=1),
                    np.cos(
                        np.radians(
                            self._aggregate_df["wmet_HorWdDir_avg"][
                                self._wind_direction_turbine_ids
                            ]
                        )
                    ).mean(axis=1),
                )
            )
            % 360.0
        )

    @logged_method_call
    def run(self, wd_bin_width=1.0, freestream_sector_width=90.0):
        """
        Estimates wake losses by comparing wind plant energy production to energy production of turbines identified as operating in freestream conditions. Wake losses are expressed as a fractional loss (e.g., 0.05 indicates a wake loss values of 5%).

        Args:
            wd_bin_width (:obj:`float`, optional): Wind diretion bin size when identifying freestream wind turbines
                (degrees). Defaults to 1 degree.
            freestream_sector_width (:obj:`float`, optional): Wind diretion sector size to use when identifying
                freestream wind turbines (degrees). If no turbines are located upstream of a particular turbine within
                the sector, the turbine will be classified as a freestream turbine. Defaults to 90 degrees.
        Returns:
            (None)
        """

        # For 1-degree wind direction bins, identify freestream turbines and calculate mean energy production
        self._aggregate_df["energy_kwh_mean_freestream"] = np.nan

        # Use 1-degree bins
        wd_bins = np.arange(0.0, 360.0, wd_bin_width)

        for wd in wd_bins:

            # identify freestream turbines
            freestream_turbine_ids = self._plant.asset.get_freestream_turbines(
                wd, sector_width=freestream_sector_width
            )

            if wd > 0.0:
                wd_bin_flag = (
                    self._aggregate_df["wmet_HorWdDir_ref"] >= (wd - 0.5 * wd_bin_width)
                ) & (self._aggregate_df["wmet_HorWdDir_ref"] < (wd + 0.5 * wd_bin_width))
            else:
                # Handle wind direction wrapping between 0 and 360 degrees for first bin
                wd_bin_flag = (
                    self._aggregate_df["wmet_HorWdDir_ref"] >= (360.0 - 0.5 * wd_bin_width)
                ) | (self._aggregate_df["wmet_HorWdDir_ref"] < (wd + 0.5 * wd_bin_width))

            # assign mean energy of freestrema turbines
            self._aggregate_df.loc[
                wd_bin_flag, "energy_kwh_mean_freestream"
            ] = self._aggregate_df.loc[wd_bin_flag, ("energy_kwh", freestream_turbine_ids)].mean(
                axis=1
            )

            # calculate wake losses during period of record
            self.wake_losses_por = (
                1
                - self._aggregate_df["energy_kwh"].mean(axis=1).sum()
                / self._aggregate_df["energy_kwh_mean_freestream"].sum()
            )
