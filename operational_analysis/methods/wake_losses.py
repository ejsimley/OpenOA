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
from operational_analysis.toolkits import filters


logger = logging.getLogger(__name__)


class WakeLosses(object):
    """
    A serial (Pandas-driven) implementation of a method for estimating wake losses from SCADA data.

    TODO: Add more details.
    """

    @logged_method_call
    def __init__(
        self,
        plant,
        wind_direction_col,
        wind_direction_turbine_ids=None,
        start_date=None,
        end_date=None,
        correct_for_derating=True,
        derating_filter_wind_speed_start=4.5,
        max_power_filter=0.95,
        wind_bin_mad_thresh=7.0,
    ):
        """
        Initialize wake loss analysis object with data and parameters.

        Args:
            plant (:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data
            wind_direction_col (:obj:`string`): SCADA column to use for wind direction
            wind_direction_turbine_ids (:obj:`list`, optional): List of turbine IDs used to calculate the average wind
                direction at each time step. If None, all turbines will be used. Defaults to None.
            start_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Start datetime for wake loss analysis. If
                None, the earliest SCADA datetime will be used. Default is None.
            end_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): End datetime for wake loss analysis. If
                None, the latest SCADA datetime will be used. Default is None.
            correct_for_derating (:obj:`bool`, optional): Indicates whether derated, curtailed, or otherwise
                unavailable turbines should be flagged and excluded from the calculation of ideal freestream wind plant power production for a given time stamp. If True, ideal freestream power production will be calculated as the sum of the derated turbine powers added to the mean power of the freestream turbines in normal operation multiplied by the number of turbines operating normally in the wind plant. Defaults to True.
            derating_filter_wind_speed_start (:obj:`float`, optional): The wind speed above which turbines will be
                flagged as derated/curtailed/shutdown if power is less than 1% of rated power (m/s). Only used when correct_for_derating is True. Defaults to 4.5 m/s.
            max_power_filter (:obj:`float`, optional): Maximum power threshold, defined as a fraction of rated power,
                to which the power curve bin filter should be applied. Only used when correct_for_derating is True. Defaults to 0.95.
            wind_bin_mad_thresh (:obj:`bool`, float): The filter threshold for each power bin used to identify
                derated/curtailed/shutdown turbines, expressed as the number of median absolute deviations above the median wind speed. Only used when correct_for_derating is True. Defaults to 7.0.

        """
        logger.info("Initializing WakeLosses analysis object")

        self._plant = plant
        self._plant.asset.prepare()
        self._wind_direction_col = wind_direction_col
        self._correct_for_derating = correct_for_derating
        self._derating_filter_wind_speed_start = derating_filter_wind_speed_start
        self._max_power_filter = max_power_filter
        self._wind_bin_mad_thresh = wind_bin_mad_thresh

        # set default start and end dates if undefined
        if start_date is None:
            self._start_date = self._plant.scada.df.index.min()
        else:
            self._start_date = start_date

        if end_date is None:
            self._end_date = self._plant.scada.df.index.max()
        else:
            self._end_date = end_date

        self._turbine_ids = list(self._plant.scada.df.id.unique())

        if wind_direction_turbine_ids is not None:
            self._wind_direction_turbine_ids = wind_direction_turbine_ids
        else:
            self._wind_direction_turbine_ids = self._turbine_ids

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
        valid_times = (self._plant.scada.df.index >= self._start_date) & (
            self._plant.scada.df.index <= self._end_date
        )

        self._aggregate_df = (
            self._plant.scada.df.loc[
                valid_times, ["id", "wmet_wdspd_avg", self._wind_direction_col, "wtur_W_avg"]
            ]
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
                            self._aggregate_df[self._wind_direction_col][
                                self._wind_direction_turbine_ids
                            ]
                        )
                    ).mean(axis=1),
                    np.cos(
                        np.radians(
                            self._aggregate_df[self._wind_direction_col][
                                self._wind_direction_turbine_ids
                            ]
                        )
                    ).mean(axis=1),
                )
            )
            % 360.0
        )

    @logged_method_call
    def _identify_derating(self):
        """
        Estimates whether each turbine is derated, curtailed, or otherwise not operating for each time stamp based on
        power curve filtering. A derated flag is then added to the aggregate data frame for each turbine.

        Args:
            (None)

        Returns:
            (None)
        """

        turb_capac = self._plant._turbine_capacity * 1e6

        for t in self._turbine_ids:
            # Apply window range filter to flag samples for which wind speed is greater than a threshold and power is
            # below 1% of rated power
            flag_window = filters.window_range_flag(
                window_col=self._aggregate_df[("wmet_wdspd_avg", t)],
                window_start=self._derating_filter_wind_speed_start,
                window_end=40,
                value_col=self._aggregate_df[("wtur_W_avg", t)],
                value_min=0.01 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
            # wind speed in each power bin
            bin_width_frac = 0.02 * (self._max_power_filter - 0.01)  # split into 50 bins
            flag_bin = filters.bin_filter(
                bin_col=self._aggregate_df[("wtur_W_avg", t)],
                value_col=self._aggregate_df[("wmet_wdspd_avg", t)],
                bin_width=bin_width_frac * turb_capac,
                threshold=self._wind_bin_mad_thresh,  # wind bin thresh
                center_type="median",
                bin_min=0.01 * turb_capac,
                bin_max=self._max_power_filter * turb_capac,
                threshold_type="mad",
                direction="above",
            )

            self._aggregate_df[("derate_flag", t)] = flag_window | flag_bin

    @logged_method_call
    def run(self, wd_bin_width=1.0, freestream_sector_width=90.0):
        """
        Estimates wake losses by comparing wind plant energy production to energy production of turbines identified as
        operating in freestream conditions. Wake losses are expressed as a fractional loss (e.g., 0.05 indicates a wake
        loss values of 5%).

        Args:
            wd_bin_width (:obj:`float`, optional): Wind diretion bin size when identifying freestream wind turbines
                (degrees). Defaults to 1 degree.
            freestream_sector_width (:obj:`float`, optional): Wind diretion sector size to use when identifying
                freestream wind turbines (degrees). If no turbines are located upstream of a particular turbine within
                the sector, the turbine will be classified as a freestream turbine. Defaults to 90 degrees.
        Returns:
            (None)
        """

        # Estimate periods when each turbine is unavailable, derated, or curtailed, based on power curve filtering
        for t in self._turbine_ids:
            self._aggregate_df[("derate_flag", t)] = False

        if self._correct_for_derating:
            self._identify_derating()

        # For 1-degree wind direction bins, identify freestream turbines and calculate mean energy production
        self._aggregate_df["wtur_W_mean_freestream"] = np.nan

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

            # Assign mean energy of freestream turbines. If correct_for_derating is True, only freestream turbines
            # operating normally will be considered.
            self._aggregate_df.loc[wd_bin_flag, "wtur_W_mean_freestream"] = (
                self._aggregate_df.loc[wd_bin_flag, "wtur_W_avg"]
                * ~self._aggregate_df.loc[wd_bin_flag, "derate_flag"]
            )[freestream_turbine_ids].sum(axis=1) / (
                ~self._aggregate_df.loc[wd_bin_flag, "derate_flag"]
            )[
                freestream_turbine_ids
            ].sum(
                axis=1
            )

        # calculate total plant-level wake losses during period of record

        # Determine ideal wind plant energy, correcting for derated turbines if correct_for_derating is True. If
        # correct_for_derating is True, ideal energy is calculated as the sum of the power produced by derated turbines
        # and the mean power produced by freestream turbines operating normally multiplied by the total number of
        # turbines operating normally
        total_derated_turbine_power = (
            (self._aggregate_df["wtur_W_avg"] * self._aggregate_df["derate_flag"]).sum(axis=1).sum()
        )

        total_potential_freestream_power = (
            self._aggregate_df["wtur_W_mean_freestream"]
            * (~self._aggregate_df["derate_flag"]).sum(axis=1)
        ).sum()

        self.wake_losses_por = 1 - self._aggregate_df["wtur_W_avg"].sum(axis=1).sum() / (
            total_potential_freestream_power + total_derated_turbine_power
        )

        # calculate turbine-level wake losses during period of record
        self.turbine_wake_losses_por = len(self._turbine_ids) * [0.0]
        for i, t in enumerate(self._turbine_ids):
            # determine ideal turbine energy as sum of the power produced by the turbine when it is derated and the
            # mean power produced by all freestream turbines when the turbine is operating normally
            ideal_turbine_energy = (
                self._aggregate_df.loc[
                    ~self._aggregate_df[("derate_flag", t)], "wtur_W_mean_freestream"
                ].sum()
                + self._aggregate_df.loc[
                    self._aggregate_df[("derate_flag", t)], ("wtur_W_avg", t)
                ].sum()
            )

            self.turbine_wake_losses_por[i] = (
                1 - self._aggregate_df[("wtur_W_avg", t)].sum() / ideal_turbine_energy
            )
