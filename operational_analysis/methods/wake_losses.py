# This class defines key analytical routines for calculating wake losses for an operating
# wind plant using SCADA data. For each SCADA time step, freestream wind turbines are
# identified using the turbine coordinates and a reference wind direction signal. The mean
# power production for all turbines in the wind plant is summed over all time steps and
# compared to the mean power of the freestream turbines summed over all time steps to
# estimate wake losses during the period of record. Methods for calclating the long-term
# wake losses using reanalaysis data and quantifying uncertaitny are provided as well.

import time  # TODO: temp remove after debugging
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

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
        UQ=True,
        start_date=None,
        end_date=None,
        reanal_products=["merra2", "era5"],
        end_date_lt=None,
    ):
        """
        Initialize wake loss analysis object with data and parameters.

        Args:
            plant (:obj:`PlantData object`): PlantData object from which PlantAnalysis should draw data
            wind_direction_col (:obj:`string`): SCADA column to use for wind direction
            wind_direction_turbine_ids (:obj:`list`, optional): List of turbine IDs used to calculate the average wind
                direction at each time step. If None, all turbines will be used. Defaults to None.
            UQ (:obj:`bool`, optional): Dertermines whether to perform uncertainty quantification using Monte Carlo
                simulation (True) or provide a single wake loss estimate (False). Defaults to True.
            start_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): Start datetime for wake loss analysis. If
                None, the earliest SCADA datetime will be used. Default is None.
            end_date (:obj:`pandas.Timestamp` or :obj:`string`, optional): End datetime for wake loss analysis. If
                None, the latest SCADA datetime will be used. Default is None.
            reanal_products (:obj:`list`, optional): List of reanalysis products to use for long-term correction. If
                UQ = True, a single product will be selected form this list each Monte Carlo iteration. Defaults to
                ["merra2", "era5"].
            end_date_lt (:obj:`string` or :obj:`pandas.Timestamp`): The last date to use for the long-term correction.
                If None, the most recent date common to all reanalysis products will be used.

        """
        logger.info("Initializing WakeLosses analysis object")

        self._plant = plant
        self._plant.asset.prepare()
        self._wind_direction_col = wind_direction_col
        self._UQ = UQ
        self._reanal_products = reanal_products

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

        if end_date_lt is not None:
            # Set minutes to 30 to handle time indices on the hour and on the half hour
            self._end_date_lt = pd.to_datetime(end_date_lt).replace(minute=30)
        else:
            # Find most recent time common to all reanalysis products
            self._end_date_lt = min(
                [
                    self._plant._reanalysis._product[product].df.index.max()
                    for product in self._reanal_products
                ]
            ).replace(minute=30)

        # Run preprocessing steps
        self._calculate_aggregate_dataframe()

    @logged_method_call
    def run(
        self,
        num_sim=100,
        bootstrap_data=True,
        wd_bin_width=5.0,
        freestream_sector_width=None,
        freestream_power_method="mean",
        correct_for_derating=True,
        derating_filter_wind_speed_start=None,
        max_power_filter=None,
        wind_bin_mad_thresh=None,
        wd_bin_width_LT_corr=5.0,
        ws_bin_width_LT_corr=1.0,
        num_years_LT=None,
        assume_no_wakes_high_ws_LT_corr=True,
        no_wakes_ws_thresh_LT_corr=13.0,
    ):
        """
        Estimates wake losses by comparing wind plant energy production to energy production of turbines identified as
        operating in freestream conditions. Wake losses are expressed as a fractional loss (e.g., 0.05 indicates a wake
        loss values of 5%).

        Args:
            num_sim (:obj:`int`, optional): Number of Monte Carlo iterations to perform. Only used if UQ = True.
                Defaults to 100.
            bootstrap_data (:obj:`bool`, optional): Specifies whether SCADA data time series are randomly resampled
                with replacement each Monte Carlo iteration to help quantify uncertainty. Only used if UQ = True.
                Defaults to True.
            wd_bin_width (:obj:`float`, optional): Wind diretion bin size when identifying freestream wind turbines
                (degrees). Defaults to 5 degrees.
            freestream_sector_width (:obj:`tuple` or :obj:`float`, optional): Wind direction sector size to use when
                identifying freestream wind turbines (degrees). If no turbines are located upstream of a particular
                turbine within the sector, the turbine will be classified as a freestream turbine. This should be a
                tuple when UQ = True (values are Monte-Carlo sampled within the specified range) or a single value when
                UQ = False. If undefined (None), a value of 90 degrees will be used if UQ = False and values of (50,
                110) will be used if UQ = True. Defaults to None.
            freestream_power_method (:obj:`string`, optional): Method used to determine the representative power
                prouction of the freestream turbines ("mean", "median", "max"). Defaults to "mean".
            correct_for_derating (:obj:`bool`, optional): Indicates whether derated, curtailed, or otherwise
                unavailable turbines should be flagged and excluded from the calculation of ideal freestream wind plant
                power production for a given time stamp. If True, ideal freestream power production will be calculated
                as the sum of the derated turbine powers added to the mean power of the freestream turbines in normal
                operation multiplied by the number of turbines operating normally in the wind plant. Defaults to True.
            derating_filter_wind_speed_start (:obj:`tuple` or :obj:`float`, optional): The wind speed above which
                turbines will be flagged as derated/curtailed/shutdown if power is less than 1% of rated power (m/s).
                Only used when correct_for_derating is True. This should be a tuple when UQ = True (values are
                Monte-Carlo sampled within the specified range) or a single value when UQ = False. If undefined (None),
                a value of 4.5 m/s will be used if UQ = False and values of (4.0, 5.0) will be used if UQ = True.
                Defaults to None.
            max_power_filter (:obj:`tuple` or :obj:`float`, optional): Maximum power threshold, defined as a fraction
                of rated power, to which the power curve bin filter should be applied. Only used when
                correct_for_derating is True. This should be a tuple when UQ = True (values are Monte-Carlo sampled
                within the specified range) or a single value when UQ = False. If undefined (None), a value of 0.95 will
                be used if UQ = False and values of (0.92, 0.98) will be used if UQ = True. Defaults to None.
            wind_bin_mad_thresh (:obj:`tuple` or :obj:`float`, optional): The filter threshold for each power bin used
                to identify derated/curtailed/shutdown turbines, expressed as the number of median absolute deviations
                above the median wind speed. Only used when correct_for_derating is True. This should be a tuple when
                UQ = True (values are Monte-Carlo sampled within the specified range) or a single value when UQ =
                False. If undefined (None), a value of 7.0 will be used if UQ = False and values of (4.0,
                13.0) will be used if UQ = True. Defaults to None.
            wd_bin_width_LT_corr (:obj:`float`, optional): Size of wind direction bins used to calculate long-term
                frequencies from historical reanalysis data and correct wake losses during the period of record
                (degrees). Defaults to 5 degrees.
            ws_bin_width_LT_corr (:obj:`float`, optional): Size of wind speed bins used to calculate long-term
                frequencies from historical reanalysis data and correct wake losses during the period of record (m/s).
                Defaults to 1 m/s.
            num_years_LT (:obj:`tuple` or :obj:`int`, optional): Number of years of historical reanalysis data to use
                for long-term correction. This should be a tuple when UQ = True (values are Monte-Carlo sampled within
                the specified range) or a single value when UQ = False. If undefined (None), a value of 20 will be
                used if UQ = False and values of (10, 20) will be used if UQ = True. Defaults to None.
            assume_no_wakes_high_ws_LT_corr (:obj:`bool`, optional): If True, wind direction and wind speed bins for
                which operational data are missing above a certain wind speed threshold are corrected by assigning the
                wind turbines' rated power to both the actual and potential power production variables during the long
                term-correction process. This assumes there are no wake losses above the wind speed threshold. Defaults
                to True.
            no_wakes_ws_thresh_LT_corr (:obj:`float`, optional): The wind speed threshold (inclusive) above which rated
                power is assigned to both the actual and potential power production variables if operational data are
                missing for any wind direction and wind speed bin during the long term-correction process. Only used if
                assume_no_wakes_high_ws_LT_corr is True. Defaults to 13 m/s.
        Returns:
            (None)
        """

        self._num_sim = num_sim
        self._bootstrap_data = bootstrap_data
        self._correct_for_derating = correct_for_derating
        self._derating_filter_wind_speed_start = derating_filter_wind_speed_start
        self._max_power_filter = max_power_filter
        self._wd_bin_width_LT_corr = wd_bin_width_LT_corr
        self._ws_bin_width_LT_corr = ws_bin_width_LT_corr
        self._num_years_LT = num_years_LT
        self._assume_no_wakes_high_ws_LT_corr = assume_no_wakes_high_ws_LT_corr
        self._no_wakes_ws_thresh_LT_corr = no_wakes_ws_thresh_LT_corr

        # Assign default parameter values depending on whether UQ is performed
        if freestream_sector_width is not None:
            self._freestream_sector_width = freestream_sector_width
        elif self._UQ:
            self._freestream_sector_width = (50.0, 110.0)
        else:
            self._freestream_sector_width = 90.0

        if derating_filter_wind_speed_start is not None:
            self._derating_filter_wind_speed_start = derating_filter_wind_speed_start
        elif self._UQ:
            self._derating_filter_wind_speed_start = (4.0, 5.0)
        else:
            self._derating_filter_wind_speed_start = 4.5

        if max_power_filter is not None:
            self._max_power_filter = max_power_filter
        elif self._UQ:
            self._max_power_filter = (0.92, 0.98)
        else:
            self._max_power_filter = 0.95

        if wind_bin_mad_thresh is not None:
            self._wind_bin_mad_thresh = wind_bin_mad_thresh
        elif self._UQ:
            self._wind_bin_mad_thresh = (4.0, 13.0)
        else:
            self._wind_bin_mad_thresh = 7.0

        if num_years_LT is not None:
            self._num_years_LT = num_years_LT
        elif self._UQ:
            self._num_years_LT = (10, 20)
        else:
            self._num_years_LT = 20

        # Set up Monte Carlo simulation inputs if UQ = True or single simulation inputs if UQ = False.
        self._setup_monte_carlo_inputs()

        # TODO: temp for plotting, remove afterwards
        self.df_LT_bin = {}

        for n in tqdm(range(self._num_sim)):

            # TODO: temp, remove after debugging
            # start_time = time.time()

            self._run = self._inputs.loc[n]

            # Estimate periods when each turbine is unavailable, derated, or curtailed, based on power curve filtering
            for t in self._turbine_ids:
                self._aggregate_df[("derate_flag", t)] = False

            if self._correct_for_derating:
                self._identify_derating()

            # Randomly resample 10-minute periods for bootstrapping
            if self._bootstrap_data:
                self._aggregate_df_sample = self._aggregate_df.sample(frac=1.0, replace=True)
            else:
                self._aggregate_df_sample = self._aggregate_df.copy()

            # For a set of wind direction bins, identify freestream turbines and calculate mean energy production
            self._aggregate_df_sample["wtur_W_mean_freestream"] = np.nan

            wd_bins = np.arange(0.0, 360.0, wd_bin_width)

            for t in self._turbine_ids:
                self._aggregate_df_sample.loc[
                    ~self._aggregate_df_sample[("derate_flag", t)], ("wtur_W_avg_normal", t)
                ] = self._aggregate_df_sample.loc[
                    ~self._aggregate_df_sample[("derate_flag", t)], ("wtur_W_avg", t)
                ]

            # TODO: temp, remove after debugging
            # print("--- Starting wind direction loop %s seconds ---" % (time.time() - start_time))

            for wd in wd_bins:

                # identify freestream turbines
                freestream_turbine_ids = self._plant.asset.get_freestream_turbines(
                    wd, sector_width=self._run.freestream_sector_width
                )

                # if bootstrapping is enabled, randomly resample set of freestream turbines
                # TODO: add back in after UQ experiment!!!
                if self._bootstrap_data:
                    freestream_turbine_ids = random.choices(
                        freestream_turbine_ids, k=len(freestream_turbine_ids)
                    )

                if wd > 0.0:
                    wd_bin_flag = (
                        self._aggregate_df_sample["wmet_HorWdDir_ref"] >= (wd - 0.5 * wd_bin_width)
                    ) & (self._aggregate_df_sample["wmet_HorWdDir_ref"] < (wd + 0.5 * wd_bin_width))
                else:
                    # Handle wind direction wrapping between 0 and 360 degrees for first bin
                    wd_bin_flag = (
                        self._aggregate_df_sample["wmet_HorWdDir_ref"]
                        >= (360.0 - 0.5 * wd_bin_width)
                    ) | (self._aggregate_df_sample["wmet_HorWdDir_ref"] < (wd + 0.5 * wd_bin_width))

                # Assign representative energy and wind speed of freestream turbines. If correct_for_derating
                # is True, only freestream turbines operating normally will be considered.

                if freestream_power_method == "mean":
                    self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_mean_freestream"] = (
                        self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_avg_normal"]
                    )[freestream_turbine_ids].mean(axis=1)
                elif freestream_power_method == "median":
                    self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_mean_freestream"] = (
                        self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_avg_normal"]
                    )[freestream_turbine_ids].median(axis=1)
                elif freestream_power_method == "max":
                    self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_mean_freestream"] = (
                        self._aggregate_df_sample.loc[wd_bin_flag, "wtur_W_avg_normal"]
                    )[freestream_turbine_ids].max(axis=1)

                self._aggregate_df_sample.loc[wd_bin_flag, "wmet_wdspd_mean_freestream"] = (
                    self._aggregate_df_sample.loc[wd_bin_flag, "wmet_wdspd_avg"]
                    * ~self._aggregate_df_sample.loc[wd_bin_flag, "derate_flag"]
                )[freestream_turbine_ids].sum(axis=1) / (
                    ~self._aggregate_df_sample.loc[wd_bin_flag, "derate_flag"]
                )[
                    freestream_turbine_ids
                ].sum(
                    axis=1
                )

            # TODO: temp, remove after debugging
            # print("--- Finished wind direction loop %s seconds ---" % (time.time() - start_time))

            # calculate total plant-level wake losses during period of record

            # Determine ideal wind plant energy, correcting for derated turbines if correct_for_derating is True. If
            # correct_for_derating is True, ideal energy is calculated as the sum of the power produced by derated turbines
            # and the mean power produced by freestream turbines operating normally multiplied by the total number of
            # turbines operating normally
            total_derated_turbine_power = (
                self._aggregate_df_sample["wtur_W_avg"] * self._aggregate_df_sample["derate_flag"]
            ).sum(axis=1)

            total_potential_freestream_power = self._aggregate_df_sample[
                "wtur_W_mean_freestream"
            ] * (~self._aggregate_df_sample["derate_flag"]).sum(axis=1)

            # Assign total potential power
            self._aggregate_df_sample["potential_plant_power"] = (
                total_potential_freestream_power + total_derated_turbine_power
            )

            # Assign actual total power produced by wind plant
            self._aggregate_df_sample["actual_plant_power"] = self._aggregate_df_sample[
                "wtur_W_avg"
            ].sum(axis=1)

            wake_losses_por = (
                1
                - self._aggregate_df_sample["actual_plant_power"].sum()
                / self._aggregate_df_sample["potential_plant_power"].sum()
            )

            # calculate turbine-level wake losses during period of record
            turbine_wake_losses_por = len(self._turbine_ids) * [0.0]
            for i, t in enumerate(self._turbine_ids):
                # determine ideal turbine energy as sum of the power produced by the turbine when it is derated and the
                # mean power produced by all freestream turbines when the turbine is operating normally

                self._aggregate_df_sample.loc[
                    ~self._aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self._aggregate_df_sample.loc[
                    ~self._aggregate_df_sample[("derate_flag", t)], "wtur_W_mean_freestream"
                ]

                self._aggregate_df_sample.loc[
                    self._aggregate_df_sample[("derate_flag", t)], ("potential_turbine_power", t)
                ] = self._aggregate_df_sample.loc[
                    self._aggregate_df_sample[("derate_flag", t)], ("wtur_W_avg", t)
                ]

                turbine_wake_losses_por[i] = (
                    1
                    - self._aggregate_df_sample[("wtur_W_avg", t)].sum()
                    / self._aggregate_df_sample[("potential_turbine_power", t)].sum()
                )

            if self._UQ:
                self.wake_losses_por[n] = wake_losses_por
                self.turbine_wake_losses_por[n, :] = turbine_wake_losses_por

                # apply long-term correction to wake losses
                wake_losses_lt, turbine_wake_losses_lt = self._apply_LT_correction()

                self.wake_losses_lt[n] = wake_losses_lt
                self.turbine_wake_losses_lt[n, :] = turbine_wake_losses_lt

            # TODO: group wake losses by wind direction and save to attribute (for plant and turbine). Use wd_LT_bin_size to bin when grouping

        if not self._UQ:
            # apply long-term correction to wake losses and average results over all reanalysis products
            self.wake_losses_por = wake_losses_por
            self.turbine_wake_losses_por = turbine_wake_losses_por

            wake_losses_lt_all_products = np.empty([len(self._reanal_products), 1])
            turbine_wake_losses_lt_all_products = np.empty(
                [len(self._reanal_products), len(self._turbine_ids)]
            )

            wake_losses_vs_wdir_lt_all_products = np.empty(
                [len(self._reanal_products), int(360.0 / self._wd_bin_width_LT_corr)]
            )
            turbine_wake_losses_vs_wdir_lt_all_products = np.empty(
                [
                    len(self._reanal_products),
                    len(self._turbine_ids),
                    int(360.0 / self._wd_bin_width_LT_corr),
                ]
            )

            for i_rean, product in enumerate(self._reanal_products):
                self._run.reanalysis_product = product

                wake_losses_lt, turbine_wake_losses_lt = self._apply_LT_correction()

                wake_losses_lt_all_products[i_rean] = wake_losses_lt
                turbine_wake_losses_lt_all_products[i_rean] = turbine_wake_losses_lt

            self.wake_losses_lt = np.mean(wake_losses_lt_all_products)
            self.turbine_wake_losses_lt = np.mean(turbine_wake_losses_lt_all_products, axis=0)

            self._wake_losses_vs_wdir_lt = np.mean(wake_losses_vs_wdir_lt_all_products, axis=0)
            self._turbine_wake_losses_vs_wdir_lt = np.mean(
                turbine_wake_losses_vs_wdir_lt_all_products, axis=0
            )

        else:
            # Calculate mean and standard deviation of wake losses from Monte Carlo simulations
            self.wake_losses_lt_mean = np.mean(self.wake_losses_lt)
            self.turbine_wake_losses_lt_mean = np.mean(self.turbine_wake_losses_lt, axis=0)
            self.wake_losses_por_mean = np.mean(self.wake_losses_por)
            self.turbine_wake_losses_por_mean = np.mean(self.turbine_wake_losses_por, axis=0)

            self.wake_losses_lt_std = np.std(self.wake_losses_lt)
            self.turbine_wake_losses_lt_std = np.std(self.turbine_wake_losses_lt, axis=0)
            self.wake_losses_por_std = np.std(self.wake_losses_por)
            self.turbine_wake_losses_por_std = np.std(self.turbine_wake_losses_por, axis=0)

    def _setup_monte_carlo_inputs(self):
        """
        Create and populate the data frame defining the Monte Carlo simulation parameters. This data frame is stored as
        self._inputs.

        Args:
            (None)

        Returns:
            (None)
        """

        if self._UQ:
            inputs = {
                "reanalysis_product": random.choices(self._reanal_products, k=self._num_sim),
                "freestream_sector_width": np.random.randint(
                    self._freestream_sector_width[0],
                    self._freestream_sector_width[1] + 1,
                    self._num_sim,
                ),
                "wind_bin_mad_thresh": np.random.randint(
                    self._wind_bin_mad_thresh[0], self._wind_bin_mad_thresh[1] + 1, self._num_sim
                ),
                "derating_filter_wind_speed_start": np.random.randint(
                    self._derating_filter_wind_speed_start[0] * 10,
                    self._derating_filter_wind_speed_start[1] * 10 + 1,
                    self._num_sim,
                )
                / 10.0,
                "max_power_filter": np.random.randint(
                    self._max_power_filter[0] * 100,
                    self._max_power_filter[1] * 100 + 1,
                    self._num_sim,
                )
                / 100.0,
                "num_years_LT": np.random.randint(
                    self._num_years_LT[0], self._num_years_LT[1] + 1, self._num_sim
                ),
            }
            self._inputs = pd.DataFrame(inputs)

            self.wake_losses_por = np.empty([self._num_sim, 1])
            self.turbine_wake_losses_por = np.empty([self._num_sim, len(self._turbine_ids)])
            self.wake_losses_lt = np.empty([self._num_sim, 1])
            self.turbine_wake_losses_lt = np.empty([self._num_sim, len(self._turbine_ids)])

            self._wake_losses_vs_wdir_por = np.empty(
                [self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self._turbine_wake_losses_vs_wdir_por = np.empty(
                [self._num_sim, len(self._turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self._wake_losses_vs_wdir_lt = np.empty(
                [self._num_sim, int(360.0 / self._wd_bin_width_LT_corr)]
            )
            self._turbine_wake_losses_vs_wdir_lt = np.empty(
                [self._num_sim, len(self._turbine_ids), int(360.0 / self._wd_bin_width_LT_corr)]
            )

        elif not self._UQ:
            inputs = {
                "reanalysis_product": self._reanal_products,
                "freestream_sector_width": len(self._reanal_products)
                * [self._freestream_sector_width],
                "wind_bin_mad_thresh": len(self._reanal_products) * [self._wind_bin_mad_thresh],
                "derating_filter_wind_speed_start": len(self._reanal_products)
                * [self._derating_filter_wind_speed_start],
                "max_power_filter": len(self._reanal_products) * [self._max_power_filter],
                "num_years_LT": len(self._reanal_products) * [self._num_years_LT],
            }
            self._inputs = pd.DataFrame(inputs)

            self._num_sim = 1

    @logged_method_call
    def _calculate_aggregate_dataframe(self):
        """
        Creates a data frame with relevant scada columns, plant-level columns, and reanalysis variables to be used for
        the wake loss analysis. The reference mean wind direction is then added to the data frame.

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

        # Add reanalysis data to aggregate data frame
        self._include_reanal_data()

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
    def _include_reanal_data(self):
        """
        Combines reanalysis data columns with the aggregate data frame for use in long-term correction.

        Args:
            (None)
        Returns:
            (None)
        """

        # combine all reanalysis variables into aggregate data frame

        for product in self._reanal_products:

            df_rean = self._plant._reanalysis._product[product].df.copy()

            # Drop minute field
            df_rean.index = df_rean.index.floor("H")

            # Upsample to 10-minute samples to match SCADA data
            df_rean = df_rean.resample("10T").ffill()
            df_rean = df_rean.add_suffix(f"_{product}")
            df_rean = df_rean[df_rean.index.isin(self._aggregate_df.index)]

            self._aggregate_df[[col for col in df_rean.columns]] = df_rean

        #     # Upsample to 10-minute samples to match SCADA data
        #     rean_prods[product] = self._plant._reanalysis._product[product].df.resample('10T').ffill()

        # df_rean = pd.concat(rean_prods,axis=1)
        # df_rean.columns = df_rean.columns.swaplevel(0, 1)
        # df_rean = df_rean.sort_index(axis=1, level=0)

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
                window_start=self._run.derating_filter_wind_speed_start,
                window_end=40,
                value_col=self._aggregate_df[("wtur_W_avg", t)],
                value_min=0.01 * turb_capac,
                value_max=1.2 * turb_capac,
            )

            # # TODO: remove, adding temporarily because of prominent derating
            # flag_window2 = (
            #                 (self._aggregate_df[("wmet_wdspd_avg", t)] >= 11.25) &
            #                 (self._aggregate_df[("wtur_W_avg", t)] >= 1.65e6) &
            #                 (self._aggregate_df[("wtur_W_avg", t)] <= 1.75e6)
            #             )

            # flag_window = flag_window | flag_window2

            # Apply bin-based filter to flag samples for which wind speed is greater than a threshold from the median
            # wind speed in each power bin
            bin_width_frac = 0.04 * (
                self._run.max_power_filter - 0.01
            )  # split into 25 bins TODO: make this an optional argument?
            flag_bin = filters.bin_filter(
                bin_col=self._aggregate_df[("wtur_W_avg", t)],
                value_col=self._aggregate_df[("wmet_wdspd_avg", t)],
                bin_width=bin_width_frac * turb_capac,
                threshold=self._run.wind_bin_mad_thresh,  # wind bin thresh
                center_type="median",
                bin_min=0.01 * turb_capac,
                bin_max=self._run.max_power_filter * turb_capac,
                threshold_type="mad",
                direction="above",
            )

            self._aggregate_df[("derate_flag", t)] = flag_window | flag_bin

    @logged_method_call
    def _apply_LT_correction(self):
        """
        Estimates long term-corrected wake losses by binning wake losses by wind direction and wind speed and weighting
        by bin frequencies from long-term historical reanalysis data.

        Args:
            (None)

        Returns:
            (:obj:`float`, :obj:`numpy.ndarra`): The estimated long term-corrected wake losses and an array containing
                the estimated turbine-level long term-corrected wake losses
        """

        # TODO: make arguments?
        min_ws_bin_lin_reg = 3.0
        bin_count_thresh_lin_reg = 50

        # First, create hourly data frame for LT correction to match resolution of reanalysis data
        df_1hr = self._aggregate_df_sample[
            [
                ("wmet_HorWdDir_ref", ""),
                ("wmet_wdspd_mean_freestream", ""),
                ("actual_plant_power", ""),
                ("potential_plant_power", ""),
            ]
            + [("wtur_W_avg", t) for t in self._turbine_ids]
            + [("potential_turbine_power", t) for t in self._turbine_ids]
            + [(f"windspeed_ms_{self._run.reanalysis_product}", "")]
        ].copy()

        df_1hr = df_1hr.resample("H").mean().dropna(how="any")

        df_1hr["wmet_wdspd_mean_freestream_bin"] = df_1hr["wmet_wdspd_mean_freestream"].round()

        # Bin by integer wind speeds
        df_ws_bin = df_1hr.groupby(("wmet_wdspd_mean_freestream_bin", "")).mean()
        df_ws_bin_count = df_1hr.groupby(("wmet_wdspd_mean_freestream_bin", "")).count()

        valid_ws_bins = (df_ws_bin.index >= min_ws_bin_lin_reg) & (
            df_ws_bin_count["wmet_wdspd_mean_freestream"] >= bin_count_thresh_lin_reg
        )

        # Find linear regression mapping from SCADA freestream wind speed to reanalysis wind speeds and use to correct
        # SCADA freestream wind speeds
        reg = LinearRegression().fit(
            df_ws_bin.loc[valid_ws_bins].index.values.reshape(-1, 1),
            df_ws_bin.loc[valid_ws_bins, f"windspeed_ms_{self._run.reanalysis_product}"].values,
        )

        df_1hr[f"wmet_wdspd_mean_freestream_corr_{self._run.reanalysis_product}"] = reg.predict(
            df_1hr["wmet_wdspd_mean_freestream"].values.reshape(-1, 1)
        )

        # TODO: remove, temp for debugging
        self.df_1hr = df_1hr

        # Create data frame with long-term frequencies of wind direction and wind speed bins from reanalysis data
        df_reanal_freqs = pd.DataFrame()

        # get reanalysis data and limit date range
        df_reanal = self._plant._reanalysis._product[self._run.reanalysis_product].df.copy()
        df_reanal = df_reanal.loc[
            (df_reanal.index <= self._end_date_lt)
            & (
                df_reanal.index
                > self._end_date_lt - pd.offsets.DateOffset(years=self._run.num_years_LT)
            )
        ]
        df_reanal["windspeed_ms_bin"] = (
            self._ws_bin_width_LT_corr
            * (df_reanal["windspeed_ms"] / self._ws_bin_width_LT_corr).round()
        )
        df_reanal["winddirection_deg_bin"] = (
            self._wd_bin_width_LT_corr
            * (df_reanal["winddirection_deg"] / self._wd_bin_width_LT_corr).round()
        )
        df_reanal.loc[df_reanal["winddirection_deg_bin"] == 360.0, "winddirection_deg_bin"] = 0.0

        df_reanal["freq"] = 1.0
        df_reanal = df_reanal.groupby(["winddirection_deg_bin", "windspeed_ms_bin"]).count()["freq"]

        df_reanal_freqs = pd.DataFrame(df_reanal / df_reanal.sum())

        # Weight wake losses in each wind direction and wind speed bin by long-term frequencies to estimate long-term
        # wake losses
        df_1hr["windspeed_ms_bin"] = (
            self._ws_bin_width_LT_corr
            * (
                df_1hr[f"wmet_wdspd_mean_freestream_corr_{self._run.reanalysis_product}"]
                / self._ws_bin_width_LT_corr
            ).round()
        )
        df_1hr["winddirection_deg_bin"] = (
            self._wd_bin_width_LT_corr
            * (df_1hr["wmet_HorWdDir_ref"] / self._wd_bin_width_LT_corr).round()
        )
        df_1hr.loc[df_1hr["winddirection_deg_bin"] == 360.0, "winddirection_deg_bin"] = 0.0

        df_1hr_bin = df_1hr.groupby(
            [("winddirection_deg_bin", ""), ("windspeed_ms_bin", "")]
        ).mean()

        df_1hr_bin = pd.concat([df_reanal_freqs, df_1hr_bin], axis=1).rename(
            columns={
                ("potential_plant_power", ""): "potential_plant_power",
                ("actual_plant_power", ""): "actual_plant_power",
            }
        )

        # If specified, assume no wake losses at wind speeds above a given threshold for bins where data are
        # missing by assigning rated power to the actual and potential power production
        if self._assume_no_wakes_high_ws_LT_corr:
            fill_inds = (df_1hr_bin["actual_plant_power"].isna()) & (
                df_1hr_bin.index.get_level_values(1) >= self._no_wakes_ws_thresh_LT_corr
            )
            df_1hr_bin.loc[fill_inds, ["actual_plant_power", "potential_plant_power"]] = (
                len(self._turbine_ids) * self._plant._turbine_capacity * 1e6
            )
            df_1hr_bin.loc[
                fill_inds,
                [("wtur_W_avg", t) for t in self._turbine_ids]
                + [("potential_turbine_power", t) for t in self._turbine_ids],
            ] = (
                self._plant._turbine_capacity * 1e6
            )

        wake_losses_lt = 1 - (
            (df_1hr_bin["freq"] * df_1hr_bin["actual_plant_power"]).sum()
            / (df_1hr_bin["freq"] * df_1hr_bin["potential_plant_power"]).sum()
        )

        # Calculate long-term corrected turbine-level wake losses
        turbine_wake_losses_lt = len(self._turbine_ids) * [0.0]
        for i, t in enumerate(self._turbine_ids):
            # determine ideal turbine energy as sum of the power produced by the turbine when it is derated and the
            # mean power produced by all freestream turbines when the turbine is operating normally

            turbine_wake_losses_lt[i] = 1 - (
                (df_1hr_bin["freq"] * df_1hr_bin[("wtur_W_avg", t)]).sum()
                / (df_1hr_bin["freq"] * df_1hr_bin[("potential_turbine_power", t)]).sum()
            )

        # TODO: group wake losses by wind direction and return (for plant and turbine). First need to create actual and potential energy (product of freq and power), then groupby wind direction.

        # TODO: remove if this isn't needed later
        self.df_LT_bin[self._run.reanalysis_product] = df_1hr_bin

        return wake_losses_lt, turbine_wake_losses_lt

    def plot_wake_losses_by_wind_direction(self, wd_bin_width=5.0):
        """
        Plots wake losses during the period of record in the form of wind farm efficiency as a function of wind
        direction.

        Args:
            wd_bin_width (:obj:`float`, optional): Wind diretion bin size for wind farm efficiency plot (degrees).
                Defaults to 5 degrees.
        Returns:
            :obj:`matplotlib.pyplot.axes`: An axes object corresponding to the wake loss plot
        """

        # TODO: update to plot POR and LT wake losses by direction with uncertainty in shaded region

        import matplotlib.pyplot as plt

        df_sub = self._aggregate_df_sample[
            ["wmet_HorWdDir_ref", "actual_plant_power", "potential_plant_power"]
        ].copy()

        # group wind farm efficiency by wind direction bin
        df_sub["winddirection_bin"] = (
            wd_bin_width * (df_sub["wmet_HorWdDir_ref"] / wd_bin_width).round()
        )
        df_sub.loc[df_sub["winddirection_bin"] == 360.0, "winddirection_bin"] = 0.0
        df_sub_bin = df_sub.groupby("winddirection_bin").sum()

        _, ax = plt.subplots(figsize=(9, 5))
        ax.plot([0, 360.0 - wd_bin_width], [1, 1], "k", linewidth=1.5)
        ax.plot(df_sub_bin["actual_plant_power"] / df_sub_bin["potential_plant_power"])
        ax.set_xlim([0, 360.0 - wd_bin_width])
        ax.set_xlabel(
            "Wind Plant Average Wind Direction (deg)"
        )  # TODO: replace with $^\circ$ and commit
        ax.set_ylabel("Wind Plant Efficiency (-)")
        ax.grid()

        return ax
