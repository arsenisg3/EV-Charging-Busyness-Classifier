from parameters import Parameters
from data_fields import SessionFields as SF, TrainingFields as TF
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataCleanerTask(ABC):
    """Abstract base class for data cleaning tasks."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the cleaning task to the DataFrame."""
        pass


class DuplicateRemover(DataCleanerTask):
    """Remove duplicate rows"""
    def apply(self, df) -> pd.DataFrame:
        return df.drop_duplicates()


class EndBeforeStart(DataCleanerTask):
    """Deals with cases where the last energy change date is lower than the start date."""
    def apply(self, df) -> pd.DataFrame:
        unnatural_pos = df[SF.LAST_ENERGY_CHANGE] < df[SF.STARTED]
        # Set last energy change date to session end date
        df.loc[unnatural_pos, SF.LAST_ENERGY_CHANGE] = df[SF.STOPPED][unnatural_pos]
        return df


class StoppedDateMissing(DataCleanerTask):
    """Infers missing session stop values from the session duration."""
    def apply(self, df) -> pd.DataFrame:
        df.loc[df[SF.STOPPED].isna(), SF.STOPPED] = (
                df[SF.STARTED] + pd.to_timedelta(df[SF.FULL_SESSION_DURATION], unit='s'))[df[SF.STOPPED].isna()]
        return df


class NanEnergyEndDate(DataCleanerTask):
    """Deals with cases where the energy end date is Nan."""
    def apply(self, df) -> pd.DataFrame:
        # If end energy date is Nan, replace it with session end date
        nan_pos = df[SF.LAST_ENERGY_CHANGE].isna()
        df.loc[nan_pos, SF.LAST_ENERGY_CHANGE] = df[SF.STOPPED][nan_pos]
        return df


class CorrectPositions(DataCleanerTask):
    """Corrects the instances where the duration doesn't match the start - end period."""
    def apply(self, df) -> pd.DataFrame:
        faulty_pos = ((df[SF.STOPPED] - df[SF.STARTED]).dt.total_seconds() != df[SF.FULL_SESSION_DURATION])
        df.loc[faulty_pos, SF.FULL_SESSION_DURATION] = ((df[SF.STOPPED] - df[SF.STARTED]).dt.total_seconds()[faulty_pos])
        return df


class OldSessionRemover(DataCleanerTask):
    """Removes rows where the session's start date is older than a specified year (have certain dates pre 2000s)."""
    def apply(self, df) -> pd.DataFrame:
        df.drop(df[df[SF.STARTED].dt.year < Parameters.CLEAN_YEAR_REMOVE_OLDER_THAN].index, inplace=True)
        return df


class SessionDesiredRange(DataCleanerTask):
    """Maintains rows for which the session lies within the desired range."""
    def apply(self, df) -> pd.DataFrame:
        # Save the first day a given charging point began operating, as well as the final one
        df[TF.FIRST_DAY] = pd.to_datetime(df.groupby(SF.ID)[SF.STARTED].transform('min').dt.date)
        df[TF.FINAL_DAY] = pd.to_datetime(df.groupby(SF.ID)[SF.STARTED].transform('max').dt.date)

        # Last active day of the charging point prior to date range of interest (to be used for feature generation)
        latest_pre_cutoff_dates_per_id = (
            df.loc[df[SF.STARTED] < pd.to_datetime(Parameters.FILTER_REMOVE_BEFORE)]
            .groupby(SF.ID)[SF.STARTED]
            .max()
        )
        df[TF.LAST_SESSION_PRE_DATA] = df[SF.ID].map(latest_pre_cutoff_dates_per_id).dt.date
        df[TF.LAST_SESSION_PRE_DATA] = df[TF.LAST_SESSION_PRE_DATA].fillna(df[TF.FIRST_DAY])

        # Remove sessions lying outside the desired range
        df.drop(df[(df[SF.STARTED].dt.date < pd.to_datetime(Parameters.FILTER_REMOVE_BEFORE).date()) |
                   (df[SF.STARTED].dt.date >= pd.to_datetime(Parameters.FILTER_REMOVE_AFTER).date())].index,
                inplace=True)
        return df


class ShortSessionRemover(DataCleanerTask):
    """Removes rows where the session duration lasts less than a specified amount."""
    def apply(self, df) -> pd.DataFrame:
        limit = Parameters.CLEAN_MIN_SESSION_DURATION_SEC
        df.drop(df[df[SF.FULL_SESSION_DURATION] < limit].index, inplace=True)
        return df


class NoEnergySessionRemover(DataCleanerTask):
    """Removes rows where no energy is expended during the session."""
    def apply(self, df) -> pd.DataFrame:
        df.drop(df[df[SF.ENERGY_WH] <= 0].index, inplace=True)
        return df


class LongSessionRemover(DataCleanerTask):
    """Removes rows where the session duration lasts more than a specified amount."""
    def apply(self, df) -> pd.DataFrame:
        limit = Parameters.CLEAN_MAX_SESSION_DURATION_SEC
        df.drop(df[df[SF.FULL_SESSION_DURATION] > limit].index, inplace=True)
        return df


class ShortLifespanRemover(DataCleanerTask):
    """Removes charging points that have been active for a short percentage of their lifetime."""
    def apply(self, df) -> pd.DataFrame:
        session_durations = df.groupby(SF.ID)[SF.FULL_SESSION_DURATION].sum()
        # Days at which each station began and stopped operating
        start_days = df.groupby(SF.ID)[SF.STARTED].min()
        end_days = df.groupby(SF.ID)[SF.STOPPED].max()
        # Lifetime duration in seconds
        duration_seconds = (end_days - start_days).dt.total_seconds()

        # Charging points to remove
        charging_points_to_remove = session_durations[(session_durations / duration_seconds) <
                                                      Parameters.ACTIVE_LIFE].index

        df = df[~df[SF.ID].isin(charging_points_to_remove)]
        return df


class ChargerType(DataCleanerTask):
    """Selects the type of charger to work on (public or private)."""
    def apply(self, df) -> pd.DataFrame:
        df = df[df[SF.TYPE] == Parameters.FILTER_PUBLIC_OR_PRIVATE]
        return df


class MinSessionRemover(DataCleanerTask):
    """Removes charging points if they have a low number of sessions."""
    def apply(self, df) -> pd.DataFrame:
        number_of_sessions = df.groupby(SF.ID)[SF.STARTED].count()
        charging_points_to_remove = number_of_sessions[number_of_sessions < Parameters.FILTER_MIN_SESSIONS].index
        df = df[~df[SF.ID].isin(charging_points_to_remove)]
        return df


class OverlappingSessionFixer(DataCleanerTask):
    """Fixes overlapping sessions within a group of sessions."""
    @staticmethod
    def fix(group) -> pd.DataFrame:
        """
        Args:
            group (DataFrame): Group of sessions sorted by 'started_at'.

        Returns:
            DataFrame: Group with adjusted 'stopped_at' values to fix overlaps.
        """
        group = group.sort_values(SF.STARTED).reset_index(drop=True)

        # Iterate over the rows and check for overlaps
        for i in range(len(group) - 1):
            current_session = group.iloc[i]
            next_session = group.iloc[i + 1]

            # Check if 'stopped_at' overlaps with the next 'started_at'
            if current_session[SF.STOPPED] > next_session[SF.STARTED]:
                group.at[i, SF.STOPPED] = current_session[SF.LAST_ENERGY_CHANGE]

        return group

    def apply(self, df) -> pd.DataFrame:
        """
        Apply the overlapping session fix across groups in the DataFrame.

        Args:
            df (DataFrame): The entire DataFrame to process.

        Returns:
            DataFrame: DataFrame with fixed overlapping sessions.
        """
        return df.groupby([SF.ID], group_keys=False)[df.columns].apply(self.fix)


class DataCleaner:
    """Applies a sequence of cleaning tasks to a dataset."""
    def __init__(self, cleaning_tasks):
        """
        Args:
            cleaning_tasks (list): List of cleaninig tasks/classes to call.
        """
        self.cleaning_tasks = cleaning_tasks

    def clean(self, df):
        """
        Applies all tasks in the pipeline to the input DataFrame.

        Args:
            df (pd.DataFrame): Input Dataset.

        Returns:
            pd.DataFrame: Cleaned Dataset.
        """
        for task in self.cleaning_tasks:
            if isinstance(task, tuple):  # If task is a tuple, pass arguments
                task_class, kwargs = task
                logger.info(f"Applying cleaning task: {task.__class__.__name__} with args {kwargs}")
                df = task_class.apply(df, **kwargs)
            else:  # If task is a class, call its apply method without arguments
                logger.info(f"Applying cleaning task: {task.__class__.__name__}")
                df = task.apply(df)
        return df
