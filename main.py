import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_fields import SessionFields as SF, TrainingFields as TF
import pipeline

from parameters import Parameters
import cleaning, features
from scipy.stats import randint, uniform
import pickle
import os
import sys
import logging

LOG_FILE_PATH = os.path.join(Parameters.PATH, 'pipeline_run.log')
os.makedirs(Parameters.PATH, exist_ok=True)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
logging.getLogger(__name__).info(f"File logging set up to: {LOG_FILE_PATH}")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
logging.getLogger(__name__).info("Console logging set up.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # dataset paths and columns
    location = 'D:/datacamp/projects/EV charging/company'
    filepath = "connectedkerb/5.connectedkerb-sessions_on_cps_created_in_2022.csv"
    filepath2 = "connectedkerb/1.connectedkerb-charge_points_created_in_2021.csv"
    pickle_location = 'D:/datacamp/projects/EV charging/OOP code/EV_charging'
    time_cols = [SF.STARTED, SF.STOPPED, SF.LAST_ENERGY_CHANGE]

    month_mappings = Parameters.MONTH_MAPPINGS

    # Bins and labels for categorizing based on population
    bins_population = Parameters.BINS_POPULATION
    labels_population = Parameters.LABELS_POPULATION

    time_periods = Parameters.TIME_PERIODS

    day_period_no = len(time_periods.keys())

    # Create an instance of PeriodAllocation
    allocate_periods = features.PeriodAllocation(time_periods)

    # Extract the durations for each of the daily periods considered
    time_periods_durations = allocate_periods.time_periods_durations

    busyness_bins = Parameters.BUSYNESS_BINS
    busyness_labels = Parameters.BUSYNESS_LABELS

    cleaning_tasks = [
        cleaning.DuplicateRemover(),
        cleaning.EndBeforeStart(),
        cleaning.StoppedDateMissing(),
        cleaning.NanEnergyEndDate(),
        cleaning.CorrectPositions(),
        cleaning.OldSessionRemover(),
        cleaning.SessionDesiredRange(),
        cleaning.ShortSessionRemover(),
        cleaning.NoEnergySessionRemover(),
        cleaning.LongSessionRemover(),
        cleaning.ShortLifespanRemover(),
        cleaning.ChargerType(),
        cleaning.MinSessionRemover(),
        cleaning.OverlappingSessionFixer()
    ]

    engineering_tasks = [
        features.EnergyKwh(),
        features.ChargingTime(),
        features.StationAvgPower(),
        features.PointPower(),
        features.ChargingPointNumber(),
        features.TimeFeatures(month_mappings, SF.STARTED),
        features.TimePeriods(),
        features.Holidays(),
        features.Population(),
        features.CategorizePopulation(bins_population, labels_population),
        features.SessionProcessor(),
        allocate_periods,
        features.GapDays(),
    ]

    season_order = Parameters.SEASON_ORDER
    weekday_order = Parameters.WEEKDAY_ORDER
    month_order = Parameters.MONTH_ORDER

    city_type_order = labels_population
    day_period_order = list(time_periods.keys())

    target_feature = TF.DAILY_PERIOD_BUSYNESS_CAT

    target_enc_cols = [SF.CITY, SF.ID]
    ordinal_enc_cols = [TF.CITY_TYPE, TF.DAY_PERIOD]

    ordinal_enc_dict = {TF.SEASON: season_order, TF.WEEKDAY: weekday_order, TF.MONTH: month_order,
                        TF.CITY_TYPE: city_type_order, TF.DAY_PERIOD: day_period_order}

    training_features_point = Parameters.TRAINING_FEATURES_POINT

    # Model and parameter grid
    models = {
        "RandomForest": RandomForestClassifier(random_state=Parameters.SEED),
        "XGBoost": XGBClassifier(random_state=Parameters.SEED, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=Parameters.SEED)
    }

    param_grids = {
        "RandomForest": {
            'n_estimators': randint(100, 2501),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_depth': list(range(10, 21)),
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced']
        },
        "XGBoost": {
            'scale_pos_weight': [],  # to be appended based on class balance
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': randint(100, 2501),
            'max_depth': randint(3, 20),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'lambda': uniform(1, 2),
            'alpha': uniform(0, 1),
        },
        "LightGBM": {
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': randint(100, 2501),
            'max_depth': randint(3, 20),
            'num_leaves': randint(20, 60),
            'min_child_samples': randint(20, 100),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.6, 0.4),
            'class_weight': ['balanced'],
        }
    }

    # Run pipeline
    pipeline = pipeline.PipelineManager()

    final_df = None

    try:
        with open(f'{pickle_location}/aggregated_stations.pkl', 'rb') as f:
            final_df = pickle.load(f)
        pipeline.df_resampled = final_df

        logger.info(f"Successfully loaded data from {pickle_location}")
        logger.info(f"Shape of loaded data: {final_df.shape}")
    except FileNotFoundError:
        logger.error(f"Error: Pickle file not found at {pickle_location}. Please check the path.")

    if final_df is None:
        logger.info('Loading Data')
        data = pipeline.load_and_merge_data(location=location, filepath=filepath, filepath2=filepath2, time_cols=time_cols)

        logger.info('Cleaning Data')
        cleaned_data = pipeline.clean_data(cleaning_tasks=cleaning_tasks)

        logger.info('Generating Features')

        data2 = pipeline.generate_features(engineering_tasks=engineering_tasks)

        logger.info('Generating Rolling Features')
        final_df = pipeline.generate_rolling_features(time_periods=time_periods,
                                                      time_periods_durations=time_periods_durations,
                                                      bins=busyness_bins, labels=busyness_labels,
                                                      mappings=month_mappings, date_col=TF.START_DATE,
                                                      day_period_no=day_period_no)

        with open('aggregated_stations.pkl', 'wb') as f:
            pickle.dump(final_df, f)

    if Parameters.AGGREGATE == 'points':

        train_data, test_data = pipeline.preprocess(group_col=SF.ID, split_ratio=Parameters.SPLIT_RATIO,
                                                    target_feature_num=TF.DAILY_PERIOD_BUSYNESS,
                                                    target_enc_cols=target_enc_cols, ordinal_enc_dict=ordinal_enc_dict)

        logger.info('Generating Interacting Features')
        train_data_final, test_data_final = pipeline.interacting_features(Parameters.SHIFTED_FEATURES,
                                                                          Parameters.FEATURE_GROUPS_MULTIPLY,
                                                                          Parameters.FEATURE_GROUPS_SUBTRACT)
        logger.info('Apply models')
        all_model_results = pipeline.train_and_evaluate(models, param_grids, training_features_point, target_feature)

    elif Parameters.AGGREGATE == 'stations':

        agg_df_station = final_df.groupby([TF.START_DATE, TF.DAY_PERIOD, SF.LOCATION]).agg(
            {
                TF.FIRST_DAY.value: 'min',
                TF.ACTIVE_TIME.value: 'sum',
                TF.MONTH.value: 'first',
                SF.COUNTRY.value: 'first',
                TF.WEEKDAY: 'first',
                TF.SEASON: 'first',
                TF.YEAR: 'first',
                SF.CITY: 'first',
                TF.HOLIDAY: 'first',
                TF.CITY_TYPE: 'first',
                TF.STATION_POINTS: 'first',
                TF.STATION_AVG_POWER: 'first',
                TF.GAP_DAYS: 'min',
                TF.AVERAGE_BUSYNESS: 'mean',
                TF.PERIOD_DURATION: 'first',
                TF.AVERAGE_PERIOD_BUSYNESS: 'mean',
                TF.DAYS_SINCE_START: 'first',
                TF.WEEKS_SINCE_START: 'first',
                TF.DAILY_PERIOD_BUSYNESS: 'mean',
                TF.DAILY_BUSYNESS: 'mean',
                TF.ROLLING_AVG_1D: 'mean',
                TF.ROLLING_AVG_7D: 'mean',
                TF.ROLLING_AVG_30D: 'mean',
                TF.ROLLING_PERIOD_AVG_1D: 'mean',
                TF.ROLLING_PERIOD_AVG_7D: 'mean',
                TF.ROLLING_PERIOD_AVG_30D: 'mean',
                TF.ROLLING_PERIOD_IQR_7D: 'mean',
                TF.ROLLING_PERIOD_IQR_30D: 'mean',
                TF.PAST_WEEK_SAME_PERIOD: 'mean',
                TF.DAILY_CUMULATIVE_AVG: 'mean',
                TF.WEEKLY_CUMULATIVE_AVG: 'mean',
                TF.PERIOD_DAILY_CUMULATIVE_AVG: 'mean',
                TF.PERIOD_WEEKLY_CUMULATIVE_AVG: 'mean',
            }
        ).reset_index()

        pipeline.df_resampled = agg_df_station

        target_enc_cols = [SF.CITY, SF.LOCATION]
        ordinal_enc_dic = {TF.SEASON: season_order, TF.WEEKDAY: weekday_order, TF.MONTH: month_order,
                           TF.CITY_TYPE: city_type_order, TF.DAY_PERIOD: day_period_order}

        training_features_station = Parameters.TRAINING_FEATURES_STATION

        for time_period in time_periods.keys():
            period_indices = agg_df_station[agg_df_station[TF.DAY_PERIOD] == time_period].index

            # According to the fraction, define how busy the charging point is
            agg_df_station.loc[period_indices, TF.DAILY_PERIOD_BUSYNESS_CAT] = (
                pd.cut(agg_df_station.loc[period_indices, TF.DAILY_PERIOD_BUSYNESS], bins=busyness_bins,
                       labels=busyness_labels, right=False))

        train_data, test_data = pipeline.preprocess(group_col=SF.LOCATION, split_ratio=Parameters.SPLIT_RATIO,
                                                    target_feature_num=TF.DAILY_PERIOD_BUSYNESS,
                                                    target_enc_cols=target_enc_cols, ordinal_enc_dict=ordinal_enc_dic)

        logger.info('Generating Interacting Features')
        train_data_final, test_data_final = pipeline.interacting_features(Parameters.SHIFTED_FEATURES,
                                                                          Parameters.FEATURE_GROUPS_MULTIPLY,
                                                                          Parameters.FEATURE_GROUPS_SUBTRACT)

        logger.info('Apply models')
        all_model_results = pipeline.train_and_evaluate(models, param_grids, training_features_station, target_feature)

    logger.info('Finished')