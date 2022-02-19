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
        self.calculate_scada_plant_dataframes()

    @logged_method_call
    def calculate_scada_plant_dataframes(self):
        """
        Creates a multiindex scada data frame with relevant columns and a data frame for plant-level variables to be
        used for the wake loss analysis. The reference mean wind direction is then added to the plant-level data frame.

        Args:
            (None)

        Returns:
            (None)
        """

        # keep relevant SCADA columns and group data frame by time and turbine ID
        self._scada_df = (
            self._plant.scada.df[["id", "wmet_wdspd_avg", self._wind_direction_col, "energy_kwh"]]
            .reset_index()
            .set_index(["time", "id"])
        )

        # remove times with any missing turbine IDs or data
        # first drop rows with any NaNs
        self._scada_df = self._scada_df.dropna(how="any")

        # next drop times with any missing turbine IDs
        # TODO: revisit because this may remove too many samples
        self._scada_df = self._scada_df.loc[
            self._scada_df.groupby("time")[self._wind_direction_col]
            .transform("size")
            .eq(len(self._plant.scada.df.id.unique()))
        ]
        self._scada_df.index = self._scada_df.index.remove_unused_levels()

        # initialize an empty plant-level data frame with a single datetime index
        self._plant_df = pd.DataFrame(
            index=self._scada_df.index.levels[0],
            columns=["wmet_HorWdDir_ref", "energy_kw_mean", "energy_kw_mean_freestream"],
        )

        # Calculate reference mean wind direction
        self.calculate_mean_wind_direction()

        # Drop turbine-level wind direction column
        self._scada_df = self._scada_df.drop(columns=[self._wind_direction_col])

    @logged_method_call
    def calculate_mean_wind_direction(self):
        """
        Calculates the mean wind direction at each time step using the specified SCADA wind direction column for the
        specified subset of turbines. This reference mean wind direction is added to the plant-level data frame.

        Args:
            (None)
        Returns:
            (None)
        """

        self._plant_df["wmet_HorWdDir_ref"] = (
            np.degrees(
                np.arctan2(
                    np.sin(
                        np.radians(
                            self._scada_df.loc[
                                (slice(None), self._wind_direction_turbine_ids),
                                self._wind_direction_col,
                            ]
                        )
                    )
                    .groupby("time")
                    .mean(),
                    np.cos(
                        np.radians(
                            self._scada_df.loc[
                                (slice(None), self._wind_direction_turbine_ids),
                                self._wind_direction_col,
                            ]
                        )
                    )
                    .groupby("time")
                    .mean(),
                )
            )
            % 360.0
        )
