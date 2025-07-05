from data_fields import SessionFields as SF, TrainingFields as TF
from parameters import Parameters

import datetime
import holidays
import time
import requests
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FeatureEngineeringTask(ABC):
    """Abstract base class for feature engineering tasks."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the engineering task to the DataFrame."""
        pass


class EnergyKwh(FeatureEngineeringTask):
    """Class computing the energy used in Kwh."""
    def apply(self, df):
        df[TF.ENERGY_KWH] = df[SF.ENERGY_WH] / 1000
        return df


class ChargingTime(FeatureEngineeringTask):
    """Adds charging durations in seconds to the dataset."""
    def apply(self, df):
        df[TF.CHARGING_DURATION] = (
                df[SF.LAST_ENERGY_CHANGE] - df[SF.STARTED]
        ).dt.total_seconds()
        return df


class StationAvgPower(FeatureEngineeringTask):
    """Calculates the average power supplied by the station in kW."""
    def apply(self, df):

        station_avg_power = df.groupby(SF.LOCATION).agg({
            TF.ENERGY_KWH: 'sum',
            TF.CHARGING_DURATION: 'sum'
        })
        df[TF.STATION_AVG_POWER] = df[SF.LOCATION].map(
            station_avg_power[TF.ENERGY_KWH] / (station_avg_power[TF.CHARGING_DURATION] / 3600)
        )
        return df


class PointPower(FeatureEngineeringTask):
    """Calculates the average power supplied by a given charging point in kW."""
    def apply(self, df):
        point_avg_power = df.groupby(SF.ID).agg({
            TF.ENERGY_KWH: 'sum',
            TF.CHARGING_DURATION: 'sum'
        })
        df[TF.POINT_AVG_POWER] = df[SF.ID].map(
            point_avg_power[TF.ENERGY_KWH] / (point_avg_power[TF.CHARGING_DURATION] / 3600)
        )
        return df


class ChargingPointNumber(FeatureEngineeringTask):
    """Calculates he number of charging points belonging to a station."""
    def apply(self, df):
        charging_points_per_station = df.groupby(SF.LOCATION)[SF.ID].nunique()
        df[TF.STATION_POINTS] = df[SF.LOCATION].map(charging_points_per_station).astype(int)
        return df


class TimeFeatures(FeatureEngineeringTask):
    """Adds season-related features like weekday, month, quarter, season, and year."""
    def __init__(self, mappings, date_col):
        """
        Initializes the TimeFeatures object.

        Args:
            mappings (dict): Dictionary of the month mappings used to create the seasons.
            date_col (str): The name of the date column used to extract starting dates.
        """
        self.mappings = mappings
        self.date_col = date_col

    def apply(self, df):
        """
        Creates the time-related features.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.Series: DataFrame updated with the time related features.
        """
        start_dates = df[self.date_col]
        df[TF.WEEKDAY] = start_dates.dt.day_name()
        df[TF.MONTH] = start_dates.dt.month_name()
        df[TF.QUARTER] = start_dates.dt.quarter
        df[TF.SEASON] = df[TF.MONTH].replace(self.mappings)
        df[TF.YEAR] = start_dates.dt.year
        return df


class TimePeriods(FeatureEngineeringTask):
    """Adds start_date and end_date (removing the time component) - to be used for further feature generations."""
    def apply(self, df):

        df[TF.START_DATE] = df[SF.STARTED].dt.normalize()
        df[TF.END_DATE] = df[SF.STOPPED].dt.normalize()
        return df


class Holidays(FeatureEngineeringTask):
    """Finds whether a holiday occurred within a certain number of days of the session date."""
    @staticmethod
    def get_holidays(df):

        country_code = df[SF.COUNTRY].iloc[0]
        start_year = df[SF.STARTED].dt.year.min()
        end_year = df[SF.STARTED].dt.year.max()

        holiday_list = []
        for year in range(start_year, end_year + 1):
            # Dynamically get the country-specific holiday calendar
            country_holidays = getattr(holidays, country_code, None)
            if country_holidays is None:
                raise ValueError(f"Country code {country_code} is not supported.")
            year_holidays = country_holidays(years=[year])
            holiday_list += list(year_holidays.keys())
        full_holiday_list = [holiday for holiday in holiday_list]
        # Binary column with True indicating that the date lies within a certain number of days from a holiday
        df[TF.HOLIDAY] = df[SF.STARTED].dt.date.apply(
            lambda x: any(abs((x - holiday).days) <= Parameters.DAYS_FROM_HOLIDAY for holiday in full_holiday_list))
        return df

    def apply(self, df):
        """
        Executes the get_holidays step.

        Returns:
            DataFrame: The updated dataset.
        """
        df = self.get_holidays(df)
        return df


class Population(FeatureEngineeringTask):
    """Adds population data to a dataset using the GeoNames API, grouped by city and country."""
    def __init__(self, username="arsenis", sleep_time=1.0):
        """
        Args:
            username (str): GeoNames API username.
            sleep_time (float): Time to wait between API calls (seconds).
        """
        self.username = username
        self.sleep_time = sleep_time

    def get_population(self, group: pd.DataFrame) -> pd.DataFrame:
        city = group[SF.CITY].iloc[0]
        country = group[SF.COUNTRY].iloc[0]
        logging.info(f"Fetching population for: {city}, {country}")

        url = f"http://api.geonames.org/searchJSON?q={city}&country={country}&maxRows=1&username={self.username}"

        time.sleep(self.sleep_time)

        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            if data['geonames']:
                population = data['geonames'][0].get('population')
                logging.info(f"Found population for {city}: {population}")
            else:
                population = None
                logging.warning(f"No population data found for {city}, {country}")

        except Exception as e:
            logging.error(f"Error fetching data for {city}, {country}: {e}")
            population = None

        group[TF.POPULATION] = population
        return group

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the population enrichment to the given DataFrame.

        Args:
            df (pd.DataFrame): Input dataset with 'city' and 'country' columns.

        Returns:
            pd.DataFrame: Dataset with an additional population column.
        """
        return (
            df.groupby([SF.CITY, SF.COUNTRY], group_keys=False)[df.columns]
            .apply(self.get_population)
        )


class CategorizePopulation(FeatureEngineeringTask):
    """Categorizes the type of living area based on total population."""
    def __init__(self, bins_pop, labels_pop):
        """
        Initializes the Categorize_population task with population bins and labels.

        Args:
            bins_pop (list): Population bins for categorization.
            labels_pop (list): Labels for the population categories.
        """
        self.bins_pop = bins_pop
        self.labels_pop = labels_pop

    def apply(self, df) -> pd.DataFrame:
        """
        Applies the population categorizing to the Dataframe.

        Args:
            df (DataFrame): The DataFrame to process.

        Returns:
            DataFrame: The updated DataFrame with city type categories.
        """

        # When population is small, sometimes it is not found. In such cases set population to 1000
        df.loc[(df[TF.POPULATION].isna()) | (df[TF.POPULATION] == 0), TF.POPULATION] = 1000

        # Categorize cities based on population
        df[TF.CITY_TYPE] = pd.cut(df[TF.POPULATION], bins=self.bins_pop, labels=self.labels_pop, right=False)
        return df


class SessionProcessor(FeatureEngineeringTask):
    """
    Processes EV charging sessions that span multiple days by splitting them into daily entries.
    This effectively means that each row in the dataset corresponds to a single day.
    """
    @staticmethod
    def split_session_across_days(row) -> list:
        """
        Splits a charging session that spans multiple days into daily parts.

        Args:
            row (pd.Series): A single row/session from the DataFrame.

        Returns:
            list: A list of pd.Series objects, each representing a portion of the session on a single day.
        """
        if row[TF.START_DATE] != row[TF.END_DATE]:  # Check if session spans multiple days
            session_span = (row[TF.END_DATE] - row[TF.START_DATE]).days
            energy = row[SF.ENERGY_WH]
            last_energy_change = row[SF.LAST_ENERGY_CHANGE]
            charging_duration = row[TF.CHARGING_DURATION]

            rows = []
            # Iterate over the days the session spans
            for n in range(session_span + 1):
                updated_row = row.copy()

                # The new date this row will represent
                current_date = row[TF.START_DATE] + datetime.timedelta(days=n)
                day_start = pd.Timestamp(f"{row[TF.START_DATE]} 00:00:00") + datetime.timedelta(days=n)
                day_end = pd.Timestamp(f"{row[TF.START_DATE]} 23:59:59") + datetime.timedelta(days=n)

                # Adjust times for the first and last day of the session
                if n == 0:
                    day_start = row[SF.STARTED]
                elif n == session_span:
                    day_end = row[SF.STOPPED]

                # Determine the energy used during this day's portion
                if (last_energy_change - current_date).days > 0:
                    # this part of the session takes place before charging ends
                    duration_time = (day_end - day_start).total_seconds() + 1
                    daily_energy = energy * duration_time / charging_duration
                    daily_last_energy_change = day_end

                    # This is the final charging day
                elif (last_energy_change - current_date).days == 0:
                    duration_time = (last_energy_change - day_start).total_seconds()
                    daily_energy = energy * duration_time / charging_duration
                    daily_last_energy_change = last_energy_change

                    # session continues into the next day but the vehicle is fully charged
                else:
                    duration_time = 0
                    daily_energy = 0
                    daily_last_energy_change = None

                day_duration = (day_end - day_start).total_seconds()
                # for session parts other than the final, take account of the extra second at midnight
                if n != session_span:
                    day_duration += 1

                # Update fields
                updated_row[TF.START_DATE] = current_date
                updated_row[TF.END_DATE] = current_date
                updated_row[SF.STARTED] = day_start
                updated_row[SF.STOPPED] = day_end
                updated_row[TF.WEEKDAY] = pd.to_datetime(current_date).day_name()
                updated_row[SF.LAST_ENERGY_CHANGE] = daily_last_energy_change
                updated_row[TF.SESSION_DURATION] = int(day_duration)
                updated_row[SF.ENERGY_WH] = daily_energy
                updated_row[TF.ENERGY_KWH] = daily_energy / 1000
                updated_row[TF.CHARGING_DURATION] = duration_time

                rows.append(updated_row)
            return rows
        else:  # Session is completed within a single day, no splitting needed
            return [row]

    def apply(self, df) -> pd.DataFrame:
        """
        Applies the population enrichment to the given DataFrame.

        Args:
            df (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: A new DataFrame where sessions spanning multiple days
                          have been split into separate daily records.
        """
        expanded_rows = []
        for _, row in df.iterrows():
            expanded_rows.extend(self.split_session_across_days(row))
        df = pd.DataFrame(expanded_rows)
        return df


class PeriodAllocation(FeatureEngineeringTask):
    """
    Splits each day into predefined time periods and computes how long each session overlaps with each period.
    """
    def __init__(self, time_periods):
        """
        Args:
            time_periods (dict): Dictionary with period names as keys and tuples of
                                 (start_time, end_time) as values.
        """
        self.time_periods = time_periods
        self.time_periods_durations = self.calculate_period_durations()

    def calculate_period_durations(self) -> dict:
        """
        Calculates the duration of each period in seconds.

        Returns:
            time_periods_durations (dict): Mapping of period names to duration.
        """
        time_periods_durations = {}
        for key, values in self.time_periods.items():
            time_periods_durations[key] = int((pd.Timestamp.combine(pd.Timestamp.today(), values[1]) -
                                               pd.Timestamp.combine(pd.Timestamp.today(), values[0])).total_seconds())
        # Account for a missing second in the 'night' period
        time_periods_durations['night'] += 1
        return time_periods_durations

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies period allocation to the DataFrame of sessions using vectorized operations.
        Each original session row is expanded into multiple rows, one for each period.

        Args:
            df (pd.DataFrame): DataFrame containing session data, with 'started' and 'stopped' datetime columns.

        Returns:
            pd.DataFrame: A new DataFrame with sessions split by period, including overlap duration.
        """

        # Prepare a DataFrame of time periods
        periods_data = [
            {'period_name': p_name, 'start_time_obj': p_times[0], 'end_time_obj': p_times[1],
             TF.PERIOD_DURATION: self.time_periods_durations[p_name]} for p_name, p_times in self.time_periods.items()
        ]
        periods_df = pd.DataFrame(periods_data)

        # Create a session/day_period Cartesian product, creating a row for every session-period combination.
        df_exploded = df.merge(periods_df, how='cross')

        # Get the start of the day for each session
        session_start_date = df_exploded[SF.STARTED].dt.floor('D')

        # Time in the day when each session starts and ends
        df_exploded['period_start_dt'] = (session_start_date +
                                          pd.to_timedelta(df_exploded['start_time_obj'].astype(str)))

        df_exploded['period_end_dt'] = (session_start_date +
                                        pd.to_timedelta(df_exploded['end_time_obj'].astype(str)))

        # Compute overlap between session and period
        overlap_start = np.maximum(df_exploded[SF.STARTED], df_exploded['period_start_dt'])
        overlap_end = np.minimum(df_exploded[SF.STOPPED], df_exploded['period_end_dt'])

        overlap_duration = (overlap_end - overlap_start).dt.total_seconds()

        # Set overlap to 0 where there is no actual overlap
        overlap_duration = overlap_duration.mask(overlap_start >= overlap_end, 0)

        # Make a 1-second adjustment for 'night' period ending at 23:59:59
        is_night_period_mask = (df_exploded['period_name'] == 'night')
        session_ends_at_235959_mask = (df_exploded[SF.STOPPED].dt.time == datetime.time(23, 59, 59))
        adjustment_mask = is_night_period_mask & session_ends_at_235959_mask

        overlap_duration.loc[adjustment_mask] += 1

        df_exploded[TF.ACTIVE_TIME] = overlap_duration
        df_exploded[TF.DAY_PERIOD] = df_exploded['period_name']

        cols_to_keep = df.columns.tolist() + [TF.DAY_PERIOD, TF.PERIOD_DURATION, TF.ACTIVE_TIME]

        station_busyness = ChargePointBusyness(df_exploded[cols_to_keep], self.time_periods_durations)
        return station_busyness.apply()


class ChargePointBusyness(FeatureEngineeringTask):
    """Computes busyness metrics for each charging point over a full time span and per day period."""
    def __init__(self, df, time_periods_durations):
        """
        Args:
            df (pd.DataFrame): Data after period allocation (one row per session-period).
            time_periods_durations (dict): Duration (in seconds) of each defined day period.
        """
        self.df = df
        self.time_periods_durations = time_periods_durations

    def day_period_busyness(self, subgroup, active_time) -> pd.DataFrame:
        """
        Computes the average busyness for given subgroup (given period and charging point).

        Args:
            subgroup (pd.DataFrame): Subset of rows from a single charging point and period.
            active_time (float): Total operational time of the charging point.

        Returns:
            pd.DataFrame: Updated subgroup with TF.AVERAGE_PERIOD_BUSYNESS.
        """
        key = subgroup[TF.DAY_PERIOD].unique()[0]
        value = self.time_periods_durations[key]
        period_active_time = active_time * (value / (24 * 3600))  # Normalize to the total time in a day
        period_usage_time = subgroup[TF.ACTIVE_TIME].sum()
        subgroup[TF.AVERAGE_PERIOD_BUSYNESS] = period_usage_time / period_active_time
        return subgroup

    def total_busyness(self, group) -> pd.DataFrame:
        """
        Computes both overall and period-level average busyness for a given charging point.

        Args:
            group (pd.DataFrame): All session-periods for a single charging point.

        Returns:
            pd.DataFrame: Group with added busyness columns.
        """
        charging_point_id = group[SF.ID].unique()[0]
        positions = self.df[SF.ID] == charging_point_id
        active_time = (self.df.loc[positions, SF.STOPPED].max() -
                       self.df.loc[positions, SF.STARTED].min()).total_seconds()
        usage_time = group[TF.ACTIVE_TIME].sum()
        group[TF.AVERAGE_BUSYNESS] = usage_time / active_time

        # Calculate day period busyness
        group_day_periods = group.groupby(TF.DAY_PERIOD, group_keys=False)[group.columns].apply(
            lambda x: self.day_period_busyness(x, active_time))
        return group_day_periods

    def charging_station_busyness(self) -> pd.DataFrame:
        """
        Applies the total charge point busyness computation across all charging points.

        Returns:
            pd.DataFrame: Updated DataFrame with busyness features.
        """
        self.df = self.df.groupby(SF.ID, group_keys=False)[self.df.columns].apply(self.total_busyness)
        return self.df

    def apply(self, df=None) -> pd.DataFrame:
        """
        Applies the full busyness computation pipeline.

        Returns:
            pd.DataFrame: Final DataFrame with busyness stats.
        """
        self.df = self.charging_station_busyness()
        return self.df


class GapDays(FeatureEngineeringTask):
    """
    Calculates the number of days since a charging point was last active.
    """
    def apply(self, df) -> pd.DataFrame:
        """
        Adds a 'gap_days' column indicating days since last usage for each charging point.

        Args:
            df (pd.DataFrame): Input dataset with activity and date information.

        Returns:
            pd.DataFrame: Final DataFrame including the gap_days column.
        """
        df = df.sort_values([SF.ID, TF.START_DATE]).copy()

        # Only consider dates where there was activity
        df['last_used_date'] = df[TF.START_DATE].where(df[TF.ACTIVE_TIME] > 0)

        n_periods = df[TF.DAY_PERIOD].nunique()
        # Shift by one day and forward-fill for each charging point
        df['last_used_date'] = df.groupby(SF.ID)['last_used_date'].transform(lambda x: x.shift(n_periods).ffill())

        # The first dates if inactive are Na. Fill with last usage date prior to date range of interest
        df['last_used_date'] = df['last_used_date'].fillna(df[TF.LAST_SESSION_PRE_DATA])

        df[TF.GAP_DAYS] = (df[TF.START_DATE] - df['last_used_date']).dt.days
        df[TF.GAP_DAYS] = df[TF.GAP_DAYS].fillna(0).astype(int)

        df.drop(columns=['last_used_date'], inplace=True)
        return df


class FeatureEngineer:
    """Applies a sequence of feature engineering tasks to a dataset."""
    def __init__(self, engineering_tasks):
        """
        Args:
            engineering_tasks (list): List of engineering tasks/classes to call.
        """
        self.engineering_tasks = engineering_tasks

    def create(self, df) -> pd.DataFrame:
        """
        Applies all tasks in the pipeline to the input DataFrame.

        Args:
            df (pd.DataFrame): Input Dataset.

        Returns:
            pd.DataFrame: Final Dataset enriched with all features.
        """
        for task in self.engineering_tasks:
            if isinstance(task, tuple):
                task_class, kwargs = task
                logger.info(f"Applying feature engineering task: {task_class.__class__.__name__} with args {kwargs}")
                df = task_class.apply(df, **kwargs)
            else:
                logger.info(f"Applying feature engineering task: {task.__class__.__name__}")
                df = task.apply(df)
        return df
