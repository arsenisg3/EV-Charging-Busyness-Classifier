from dataclasses import dataclass
import pandas as pd
from data_fields import SessionFields as SF, TrainingFields as TF
import os


@dataclass
class Parameters:
    """Dataclass to handle individual settings."""

    # Cleaning parameters
    CLEAN_YEAR_REMOVE_OLDER_THAN: int = 2021
    CLEAN_MIN_SESSION_DURATION_SEC: int = 30
    CLEAN_MAX_SESSION_DURATION_SEC: int = 60 * 60 * 48

    # Filtering parameters
    ACTIVE_LIFE: float = 0.05
    FILTER_MIN_SESSIONS: int = 30
    DAYS_FROM_HOLIDAY: int = 2
    FILTER_PUBLIC_OR_PRIVATE: str = "public"
    FILTER_REMOVE_BEFORE: str = "2022-01-01"
    FILTER_REMOVE_AFTER: str = "2023-01-01"

    # Aggregation group variable (effectively predict whether busyness of an individual point or a station as a whole)
    AGGREGATE = "stations"

    # Encoding parameters
    ALPHA = 10

    # Training parameters
    SEED: int = 42
    N_ITERATIONS: int = 150
    VERBOSE: int = 2
    CV: int = 3
    SPLIT_RATIO: float = 0.8
    N_JOBS: int = 2
    CUMULATIVE_FEATURE_IMPORTANCE_THRESHOLD = 0.90
    FEATURE_NUMBER_THRESHOLD: int = 10

    # Output path
    PATH = os.getcwd()
    PLOT_PATH = os.path.join(PATH, "./plots")

    # Desired mappings / classes
    MONTH_MAPPINGS = {
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'}

    # Bins and labels for categorizing based on population
    BINS_POPULATION = [0, 1e4, 1e5, 1e6, float('inf')]
    LABELS_POPULATION = ['Village', 'Town', 'City', 'Metropolis']

    # Daily time periods of interest
    TIME_PERIODS = {
        'early_morning': (pd.Timestamp("00:00:00").time(), pd.Timestamp("06:00:00").time()),
        'morning': (pd.Timestamp("06:00:00").time(), pd.Timestamp("12:00:00").time()),
        'afternoon': (pd.Timestamp("12:00:00").time(), pd.Timestamp("17:00:00").time()),
        'evening': (pd.Timestamp("17:00:00").time(), pd.Timestamp("21:00:00").time()),
        'night': (pd.Timestamp("21:00:00").time(), pd.Timestamp("23:59:59").time())
    }

    BUSYNESS_BINS = [0, 0.10, 1.01]  # Define range boundaries (proportion of the day charging point is occupied)
    BUSYNESS_LABELS = ['Not Busy', 'Busy']  # Labels for each range
    POSITIVE_CLASS_VALUE = 'Busy'

    # Define orders for encoding
    BUSYNESS_ORDER = ['Not Busy', 'Busy']
    SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']
    WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December']

    # Shifted feature dictionaries
    SHIFTED_FEATURES = {TF.HOLIDAY_1D_SHIFT: TF.HOLIDAY, TF.WEEKDAY_ENC_1D_SHIFT: TF.WEEKDAY_ENCODED,
                        TF.DAILY_BUSYNESS_2D_SHIFT: TF.ROLLING_AVG_1D,
                        TF.DAILY_PERIOD_BUSYNESS_2D_SHIFT: TF.ROLLING_PERIOD_AVG_1D}

    # Interacting features dictionaries
    FEATURE_GROUPS_MULTIPLY = {TF.WEEKDAY_HOLIDAY: (TF.WEEKDAY_ENCODED, TF.HOLIDAY),
                               TF.WEEKDAY_PERIOD: (TF.WEEKDAY_ENCODED, TF.DAY_PERIOD_ENCODED),
                               TF.WEEKDAY_HOLIDAY_1D_SHIFT: (TF.WEEKDAY_ENC_1D_SHIFT, TF.HOLIDAY_1D_SHIFT)}

    FEATURE_GROUPS_SUBTRACT = {TF.DAILY_BUSYNESS_CHANGE: (TF.ROLLING_AVG_1D, TF.DAILY_BUSYNESS_2D_SHIFT),
                               TF.DAILY_PERIOD_BUSYNESS_CHANGE: (TF.ROLLING_PERIOD_AVG_1D,
                                                                 TF.DAILY_PERIOD_BUSYNESS_2D_SHIFT)}

    # Features for training when grouped by charging point ID
    TRAINING_FEATURES_POINT = [
        TF.DAY_PERIOD_ENCODED, TF.MONTH_ENCODED, TF.SEASON_ENCODED, TF.WEEKDAY_ENCODED, TF.HOLIDAY,
        TF.STATION_AVG_POWER, TF.STATION_POINTS, TF.CITY_TYPE_ENCODED, TF.POINT_AVG_POWER, TF.ROLLING_AVG_1D,
        TF.ROLLING_AVG_7D, TF.ROLLING_AVG_30D, TF.WEEKLY_CUMULATIVE_AVG, TF.DAILY_CUMULATIVE_AVG, TF.DAYS_SINCE_START,
        TF.WEEKS_SINCE_START, TF.GAP_DAYS, TF.ROLLING_PERIOD_AVG_1D, TF.ROLLING_PERIOD_AVG_7D,
        TF.ROLLING_PERIOD_AVG_30D, TF.PAST_WEEK_SAME_PERIOD, TF.ROLLING_PERIOD_IQR_7D, TF.ROLLING_PERIOD_IQR_30D,
        TF.PERIOD_DAILY_CUMULATIVE_AVG, TF.PERIOD_WEEKLY_CUMULATIVE_AVG, TF.AVERAGE_BUSYNESS,
        TF.AVERAGE_PERIOD_BUSYNESS, TF.WEEKDAY_HOLIDAY, TF.WEEKDAY_PERIOD, TF.WEEKDAY_HOLIDAY_1D_SHIFT,
        TF.DAILY_BUSYNESS_CHANGE, TF.DAILY_PERIOD_BUSYNESS_CHANGE, TF.PERIOD_DURATION #, SF.ID
    ]

    TRAINING_FEATURES_STATION = [
            TF.DAY_PERIOD_ENCODED, TF.MONTH_ENCODED, TF.SEASON_ENCODED, TF.WEEKDAY_ENCODED, TF.HOLIDAY,
            TF.STATION_AVG_POWER, TF.STATION_POINTS, TF.CITY_TYPE_ENCODED, TF.ROLLING_AVG_1D, TF.ROLLING_AVG_7D,
            TF.ROLLING_AVG_30D, TF.WEEKLY_CUMULATIVE_AVG, TF.DAILY_CUMULATIVE_AVG, TF.DAYS_SINCE_START,
            TF.WEEKS_SINCE_START, TF.GAP_DAYS, TF.ROLLING_PERIOD_AVG_1D, TF.ROLLING_PERIOD_AVG_7D,
            TF.ROLLING_PERIOD_AVG_30D, TF.PAST_WEEK_SAME_PERIOD, TF.ROLLING_PERIOD_IQR_7D, TF.ROLLING_PERIOD_IQR_30D,
            TF.PERIOD_DAILY_CUMULATIVE_AVG, TF.PERIOD_WEEKLY_CUMULATIVE_AVG, TF.AVERAGE_BUSYNESS,
            TF.AVERAGE_PERIOD_BUSYNESS, TF.WEEKDAY_HOLIDAY, TF.WEEKDAY_PERIOD, TF.WEEKDAY_HOLIDAY_1D_SHIFT,
            TF.DAILY_BUSYNESS_CHANGE, TF.DAILY_PERIOD_BUSYNESS_CHANGE, TF.PERIOD_DURATION #, SF.LOCATION
        ]
