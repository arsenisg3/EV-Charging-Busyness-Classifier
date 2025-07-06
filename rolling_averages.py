from data_fields import SessionFields as SF, TrainingFields as TF
from parameters import Parameters
import features
import pandas as pd
from features import FeatureEngineeringTask


class Aggregate:
    """
    Performs and aggregation by charging point in order to have a single date/day period/charging point combination.
    """
    def __init__(self, df):
        """
        Initializes the charging point aggregation task.

        Args:
            df (pd.DataFrame): Dataframe possessing start date, period of the day and charging point ID columns.
        """
        required_columns = {TF.START_DATE, TF.DAY_PERIOD, SF.ID}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        self.df = df

    def aggregate_by_point(self) -> pd.DataFrame:
        """
        Performs the aggregation.

        Returns:
            pd.DataFrame: Aggregated dataframe.
        """
        agg_df = self.df.groupby([TF.START_DATE, TF.DAY_PERIOD, SF.ID]).agg(
            {
                SF.TYPE.value: 'first',
                TF.FIRST_DAY.value: 'first',
                TF.FINAL_DAY.value: 'first',
                TF.LAST_SESSION_PRE_DATA: 'first',
                TF.ACTIVE_TIME.value: 'sum',
                TF.PERIOD_DURATION: 'first',
                TF.MONTH.value: 'first',  # Keep the first location
                SF.COUNTRY.value: 'first',
                TF.WEEKDAY: 'first',
                TF.SEASON: 'first',
                TF.YEAR: 'first',
                SF.CITY: 'first',
                SF.LOCATION: 'first',
                TF.HOLIDAY: 'first',
                TF.CITY_TYPE: 'first',
                TF.STATION_POINTS: 'first',
                TF.STATION_AVG_POWER: 'first',
                TF.POINT_AVG_POWER: 'first',
                TF.GAP_DAYS: 'first',
                TF.AVERAGE_BUSYNESS: 'first',
                TF.AVERAGE_PERIOD_BUSYNESS: 'first'
            }
        ).reset_index()

        return agg_df


class OperationTime(FeatureEngineeringTask):
    """
    Calculates the total amount of days and weeks the point has been in operation, as well as days and weeks since
    the desired start range began.
    """
    def apply(self, agg_df) -> pd.DataFrame:
        """
        Args:
            agg_df (pd.DataFrame): Aggregated dataframe.

        Returns:
            pd.DataFrame: Dataset including the new time features.
        """
        agg_df[TF.START_DATE] = pd.to_datetime(agg_df[TF.START_DATE], errors='coerce')

        # Calculate days since the beginning of the considered data for each charging_point
        agg_df[TF.DAYS_SINCE_START] = agg_df.groupby(SF.ID)[TF.START_DATE].transform(
            lambda x: (x - agg_df[TF.START_DATE].min()).dt.days)

        # Calculate corresponding weeks
        agg_df[TF.WEEKS_SINCE_START] = agg_df[TF.DAYS_SINCE_START] // 7

        # Calculate days and weeks since a given charging point first began operating
        agg_df[TF.DAYS_SINCE_POINT_ACTIVE] = (agg_df[TF.START_DATE] - agg_df[TF.FIRST_DAY]).dt.days
        agg_df[TF.WEEKS_SINCE_POINT_ACTIVE] = agg_df[TF.DAYS_SINCE_POINT_ACTIVE] // 7

        return agg_df


class DailyPeriodBusyness(FeatureEngineeringTask):
    """Calculates the daily busyness level and the corresponding category."""
    def __init__(self, time_periods, time_periods_durations, bins, labels):
        """
        Args:
            time_periods (dict): Dictionary with period names as keys and tuples of (start_time, end_time) as values.
            time_periods_durations (dict): Duration (in seconds) of each defined day period.
            bins (list): List defining the busyness levels.
            labels (list): List of strings of the busyness categories.
        """
        self.time_periods = time_periods
        self.time_periods_durations = time_periods_durations
        self.bins = bins
        self.labels = labels

    def apply(self, agg_df):
        """
        Returns:
            pd.DataFrame: Final DataFrame with daily busyness level and category columns.
        """
        agg_df[TF.START_DATE] = agg_df[TF.START_DATE].dt.date

        for time_period in self.time_periods.keys():
            period_indices = agg_df[agg_df[TF.DAY_PERIOD] == time_period].index
            agg_df.loc[period_indices, TF.DAILY_PERIOD_BUSYNESS] = (
                        agg_df.loc[period_indices, TF.ACTIVE_TIME] /
                        self.time_periods_durations[time_period])

            # according to the fraction, define how busy the charging point is
            agg_df.loc[period_indices, TF.DAILY_PERIOD_BUSYNESS_CAT] = (
                pd.cut(agg_df.loc[period_indices, TF.DAILY_PERIOD_BUSYNESS], bins=self.bins, labels=self.labels,
                       right=False))

        return agg_df


class FillMissingDates:
    """
    Ensures every station has no missing row per day and day period combination within the desired date range.
    For stations that have not yet started operating and/or have stopped operation, the start / end range is adjusted.
    """
    def __init__(self, agg_df, mappings, date_col):
        """
        Args:
            agg_df (pd.DataFrame): Aggregated dataframe.
            mappings (dict): Dictionary of months mapped to seasons.
            date_col (str): Column name for start date.
        """
        self.agg_df = agg_df
        self.mappings = mappings
        self.date_col = date_col
        self.df_resampled = self.fill_missing_dates_and_periods()

    def fill_missing_dates_and_periods(self) -> pd.DataFrame:
        """
        Fills missing dates and day periods within each station's date range.

        Returns:
             pd.DataFrame: Dataset with missing periods and dates filled in.
        """
        self.agg_df[TF.START_DATE] = pd.to_datetime(self.agg_df[TF.START_DATE])
        all_day_periods = self.agg_df[TF.DAY_PERIOD].unique()

        self.agg_df = self.agg_df.set_index([TF.START_DATE, TF.DAY_PERIOD, SF.ID])

        new_df = []
        # iterate over all charging points
        for point in self.agg_df.index.get_level_values(SF.ID).unique():
            point_df = self.agg_df.xs(point, level=SF.ID).reset_index()

            point_df = point_df.drop_duplicates(subset=[TF.START_DATE, TF.DAY_PERIOD])

            # first and final session dates within desired time range
            min_date = point_df[TF.START_DATE].min()
            max_date = point_df[TF.START_DATE].max()

            # Check if point was active before min_date (meaning that point was unused during first few days of year)
            if point_df[TF.FIRST_DAY].min() < min_date:
                # first active day occurs before minimum date in range of interest
                if point_df[TF.FIRST_DAY].min() < pd.to_datetime(Parameters.FILTER_REMOVE_BEFORE):
                    min_date = pd.to_datetime(Parameters.FILTER_REMOVE_BEFORE).date()
                else:
                    min_date = point_df[TF.FIRST_DAY].min()

            # Same with max_date
            if point_df[TF.FINAL_DAY].max() > max_date:  # point sees usage beyond the max date of interest

                # final active day occurs after maximum date in range of interest
                if point_df[TF.FIRST_DAY].max() > pd.to_datetime(Parameters.FILTER_REMOVE_AFTER):
                    max_date = pd.to_datetime(Parameters.FILTER_REMOVE_AFTER).date()
                else:
                    max_date = point_df[TF.FINAL_DAY].max()

            all_dates = pd.date_range(min_date, max_date, freq='D')

            # Create a full multi-index
            full_index = pd.MultiIndex.from_product(
                [all_dates, all_day_periods],
                names=[TF.START_DATE, TF.DAY_PERIOD]
            )

            # Reindex to fill missing values
            new_point_df = point_df.set_index([TF.START_DATE, TF.DAY_PERIOD]).reindex(full_index)

            new_point_df[SF.ID] = point
            new_df.append(new_point_df.reset_index())

        df_resampled = pd.concat(new_df, ignore_index=True)

        # Forward-fill station-level info grouped by ID
        station_cols = [
            SF.TYPE, TF.FIRST_DAY, TF.FINAL_DAY, TF.LAST_SESSION_PRE_DATA, SF.COUNTRY, SF.CITY, SF.LOCATION,
            TF.CITY_TYPE, TF.STATION_POINTS, TF.STATION_AVG_POWER, TF.POINT_AVG_POWER, TF.AVERAGE_BUSYNESS,
            TF.PERIOD_DURATION
        ]

        df_resampled = df_resampled.sort_values([SF.ID, TF.START_DATE, TF.DAY_PERIOD])
        df_resampled[station_cols] = df_resampled.groupby(SF.ID)[station_cols].ffill().bfill()
        return df_resampled

    def enrich_missing_rows(self) -> pd.DataFrame:
        """
        Fills nan values of the updated dataset.

        Returns:
             pd.DataFrame: Updated dateset with corrected column values.
        """
        self.df_resampled[TF.DAILY_PERIOD_BUSYNESS] = self.df_resampled[TF.DAILY_PERIOD_BUSYNESS].fillna(0)
        self.df_resampled[TF.DAILY_PERIOD_BUSYNESS_CAT] = (
            self.df_resampled[TF.DAILY_PERIOD_BUSYNESS_CAT].fillna('Not Busy'))
        self.df_resampled[TF.ACTIVE_TIME] = self.df_resampled[TF.ACTIVE_TIME].fillna(0)

        time_features = features.TimeFeatures(self.mappings, self.date_col)
        self.df_resampled = time_features.apply(self.df_resampled)

        self.df_resampled[SF.STARTED] = self.df_resampled[TF.START_DATE]

        # Recall the Holiday and GapDay tasks in order to fill in the values of the new rows
        self.df_resampled = features.Holidays().apply(self.df_resampled)
        self.df_resampled[TF.HOLIDAY] = self.df_resampled[TF.HOLIDAY].astype('uint8')
        self.df_resampled = features.GapDays().apply(self.df_resampled)

        # fill in the nan values of the new rows with the average period busyness values of each ID point
        self.df_resampled[TF.AVERAGE_PERIOD_BUSYNESS] = (
            self.df_resampled.groupby([SF.ID, TF.DAY_PERIOD])[TF.AVERAGE_PERIOD_BUSYNESS]
            .transform('first')
        )

        return self.df_resampled

    def apply(self) -> pd.DataFrame:
        self.df_resampled = self.enrich_missing_rows()
        return self.df_resampled


class RollingAverages(FeatureEngineeringTask):
    """Computes rolling averages for the charging point's overall daily busyness."""
    def apply(self, df_resampled) -> pd.DataFrame:

        grouped_ds = df_resampled.groupby([TF.START_DATE, SF.ID])[TF.ACTIVE_TIME].sum().reset_index()
        # Overall busyness of charging point within the day
        grouped_ds[TF.DAILY_BUSYNESS] = grouped_ds[TF.ACTIVE_TIME] / (24 * 3600)

        rolling_avg_features = [TF.ROLLING_AVG_1D, TF.ROLLING_AVG_7D, TF.ROLLING_AVG_30D]
        for window, feature in zip([1, 7, 30], rolling_avg_features):
            grouped_ds[feature] = (
                grouped_ds.groupby(SF.ID)[TF.DAILY_BUSYNESS]
                .apply(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
                .reset_index(level=0, drop=True).fillna(0)
            )

        df_resampled = df_resampled.merge(
            grouped_ds[
                [TF.START_DATE, SF.ID, TF.DAILY_BUSYNESS, TF.ROLLING_AVG_1D, TF.ROLLING_AVG_7D, TF.ROLLING_AVG_30D]],
            on=[TF.START_DATE, SF.ID],
            how='left'
        )
        return df_resampled


class RollingPeriodAverages(FeatureEngineeringTask):
    """Computes rolling averages for the charging point's period daily busyness."""
    def apply(self, df_resampled) -> pd.DataFrame:
        period_rolling_avg_features = [TF.ROLLING_PERIOD_AVG_1D, TF.ROLLING_PERIOD_AVG_7D, TF.ROLLING_PERIOD_AVG_30D]

        for window, feature in zip([1, 7, 30], period_rolling_avg_features):
            df_resampled[feature] = (
                df_resampled
                .sort_values([SF.ID, TF.DAY_PERIOD, TF.START_DATE])
                .groupby([SF.ID, TF.DAY_PERIOD])[TF.DAILY_PERIOD_BUSYNESS]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))  # Shift by 1 day
            ).fillna(0)

        return df_resampled


class RollingPeriodIQR(FeatureEngineeringTask):
    """Computes rolling IQR for the charging point's period daily busyness in order to measure the spread of consistency
    of the busyness for each period."""
    def apply(self, df_resampled: pd.DataFrame) -> pd.DataFrame:
        period_rolling_iqr_features = [TF.ROLLING_PERIOD_IQR_7D, TF.ROLLING_PERIOD_IQR_30D]

        for window, feature in zip([7, 30], period_rolling_iqr_features):

            q3 = df_resampled.sort_values([SF.ID, TF.DAY_PERIOD, TF.START_DATE]).groupby([SF.ID, TF.DAY_PERIOD])[TF.DAILY_PERIOD_BUSYNESS] \
                .transform(lambda x: x.rolling(window=window, min_periods=1).quantile(0.75))

            q1 = df_resampled.sort_values([SF.ID, TF.DAY_PERIOD, TF.START_DATE]).groupby([SF.ID, TF.DAY_PERIOD])[TF.DAILY_PERIOD_BUSYNESS] \
                .transform(lambda x: x.rolling(window=window, min_periods=1).quantile(0.25))

            df_resampled[feature] = (q3 - q1).shift(1).fillna(0)

        return df_resampled


class PastBusyness(FeatureEngineeringTask):
    """Computes the charging point overall busyness on the same weekday a week earlier."""
    def __init__(self, day_period_no):
        """
        Args:
            day_period_no (int): Number of time periods within the day.
        """
        self.day_period_no = day_period_no

    def apply(self, df_resampled) -> pd.DataFrame:
        df_resampled[TF.PAST_WEEK_SAME_PERIOD] = (
            df_resampled.groupby(SF.ID)[TF.DAILY_PERIOD_BUSYNESS]
            .apply(lambda x: x.shift(7 * self.day_period_no))
            .reset_index(level=0, drop=True)
        ).fillna(0)

        return df_resampled


class CumulativeAverages(FeatureEngineeringTask):
    """Computes daily and weekly cumulative averages for busyness."""
    def apply(self, df_resampled) -> pd.DataFrame:

        # Weekly cumulative mean
        weekly_cumulative_avg = (
            df_resampled.groupby([SF.ID, TF.WEEKS_SINCE_START])[TF.DAILY_BUSYNESS]
            .mean()
            .groupby(level=0, group_keys=False)  # Prevents adding another index level
            .apply(lambda x: x.expanding().mean().shift(1))
            .fillna(0)
        )

        # Reset index to turn 'ID' and 'weeks_since_start' into columns
        weekly_cumulative_avg_reset = weekly_cumulative_avg.reset_index().rename(
            columns={TF.DAILY_BUSYNESS: TF.WEEKLY_CUMULATIVE_AVG}
        )

        # Merge on the relevant columns
        df_resampled = df_resampled.merge(
            weekly_cumulative_avg_reset,
            how='left',
            on=[SF.ID, TF.WEEKS_SINCE_START]  # Merge on these columns
        )

        # Daily cumulative mean
        daily_cumulative_avg = (
            df_resampled.groupby([SF.ID, TF.DAYS_SINCE_START])[TF.DAILY_BUSYNESS]
            .mean()
            .groupby(level=0, group_keys=False)  # Prevents adding another index level
            .apply(lambda x: x.expanding().mean().shift(1))
            .fillna(0)
        )

        # Reset index to turn 'evse_uid' and 'weeks_since_start' into columns
        daily_cumulative_avg_reset = daily_cumulative_avg.reset_index().rename(
            columns={TF.DAILY_BUSYNESS: TF.DAILY_CUMULATIVE_AVG}
        )

        # Merge on the relevant columns
        df_resampled = df_resampled.merge(
            daily_cumulative_avg_reset,
            how='left',
            on=[SF.ID, TF.DAYS_SINCE_START]  # Merge on these columns
        )

        return df_resampled


class PeriodCumulativeAverages(FeatureEngineeringTask):
    """Computes daily and weekly busyness cumulative averages for each day period."""
    def apply(self, df_resampled) -> pd.DataFrame:
        # Calculate period-level mean for each week
        weekly_mean = df_resampled.groupby([SF.ID, TF.DAY_PERIOD, TF.WEEKS_SINCE_START])[
            TF.DAILY_PERIOD_BUSYNESS].mean()

        # Reset to flat structure for better control
        weekly_mean = weekly_mean.reset_index()

        # Sort values to ensure proper ordering before expanding
        weekly_mean = weekly_mean.sort_values(by=[SF.ID, TF.DAY_PERIOD, TF.WEEKS_SINCE_START])

        # Group and compute expanding average
        weekly_mean[TF.PERIOD_WEEKLY_CUMULATIVE_AVG] = (
            weekly_mean
            .groupby([SF.ID, TF.DAY_PERIOD])[TF.DAILY_PERIOD_BUSYNESS]
            .transform(lambda x: x.expanding().mean().shift(1).fillna(0))
        )

        # Now merge back into original df_resampled
        df_resampled = df_resampled.merge(
            weekly_mean[[SF.ID, TF.DAY_PERIOD, TF.WEEKS_SINCE_START, TF.PERIOD_WEEKLY_CUMULATIVE_AVG]],
            how='left',
            on=[SF.ID, TF.DAY_PERIOD, TF.WEEKS_SINCE_START]
        )

        # Daily mean per period
        daily_mean = df_resampled.groupby([SF.ID, TF.DAY_PERIOD, TF.DAYS_SINCE_START])[
            TF.DAILY_PERIOD_BUSYNESS].mean().reset_index()
        daily_mean = daily_mean.sort_values(by=[SF.ID, TF.DAY_PERIOD, TF.DAYS_SINCE_START])

        daily_mean[TF.PERIOD_DAILY_CUMULATIVE_AVG] = (
            daily_mean
            .groupby([SF.ID, TF.DAY_PERIOD])[TF.DAILY_PERIOD_BUSYNESS]
            .transform(lambda x: x.expanding().mean().shift(1).fillna(0))
        )

        df_resampled = df_resampled.merge(
            daily_mean[[SF.ID, TF.DAY_PERIOD, TF.DAYS_SINCE_START, TF.PERIOD_DAILY_CUMULATIVE_AVG]],
            how='left',
            on=[SF.ID, TF.DAY_PERIOD, TF.DAYS_SINCE_START]
        )

        return df_resampled
