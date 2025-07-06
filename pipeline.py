import loading, cleaning, features, rolling_averages, preprocessing, data_preprocessing_utils, model_training, interacting_features
from parameters import Parameters
from vizualisation import Visualizer

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PipelineManager:
    """
    Manages the end-to-end pipeline for data loading, cleaning,
    feature engineering, preprocessing, and model training.
    """
    def __init__(self):
        self.data = None
        self.agg_df = None
        self.df_resampled = None
        self.train_data = None
        self.test_data = None
        self.model_results = None

    def load_and_merge_data(self, location, filepath, filepath2, time_cols) -> pd.DataFrame:
        """
        Load the datasets and merge them.

        Args:
            location (str): The directory containing the files.
            filepath (str): Filepath to the first dataset file.
            filepath2 (str): Filepath to  the second dataset file.
            time_cols (list): List of columns to parse as datetime.

        Returns:
            data (pd.DataFrame): Dataframe containing the merged data.
        """
        loader = loading.DataLoader(location, filepath, filepath2)
        self.data = loader.load_data(time_cols)
        self.data = loader.merge_data()
        return self.data

    def clean_data(self, cleaning_tasks) -> pd.DataFrame:
        """
        Perform the desired cleaning tasks.

        Args:
            cleaning_tasks (list): List of cleaning tasks.

        Returns:
            data (pd.DataFrame): Post cleaning DataFrame.
        """
        cleaner = cleaning.DataCleaner(cleaning_tasks)
        self.data = cleaner.clean(self.data)
        return self.data

    def generate_features(self, engineering_tasks) -> pd.DataFrame:
        """
        Engineer the desired features.

        Args:
            engineering_tasks (list): List of engineering_tasks tasks.

        Returns:
            data (pd.DataFrame): DataFrame with added features.
        """
        engineer = features.FeatureEngineer(engineering_tasks)
        self.data = engineer.create(self.data)
        return self.data

    def generate_rolling_features(self, time_periods, time_periods_durations, bins, labels,
                                  mappings, date_col, day_period_no) -> pd.DataFrame:
        """
        Engineer additional features such as rolling averages.

        Args:
            time_periods (dict): Dictionary with period names as keys and tuples of (start_time, end_time) as values.
            time_periods_durations (dict): Duration (in seconds) of each defined day period.
            bins (list): List defining the busyness levels.
            labels (list): List of strings of the busyness categories.
            mappings (dict): Dictionary of months mapped to seasons.
            date_col (str): Column name for start date.
            day_period_no (int): Number of time periods within the day.

        Returns:
            df_resampled (pd.DataFrame): DataFrame with additional features.
        """
        logger.info('Aggregating the data')
        aggregator = rolling_averages.Aggregate(self.data)
        self.agg_df = aggregator.aggregate_by_point()
        logger.info('Aggregation complete!')
        operation_time = rolling_averages.OperationTime()
        self.agg_df = operation_time.apply(self.agg_df)
        daily_busyness = rolling_averages.DailyPeriodBusyness(time_periods, time_periods_durations, bins, labels)
        self.agg_df = daily_busyness.apply(self.agg_df)
        fill = rolling_averages.FillMissingDates(self.agg_df, mappings, date_col)
        self.df_resampled = fill.apply()
        fill_updated = rolling_averages.OperationTime()
        self.df_resampled = fill_updated.apply(self.df_resampled)
        rolling_avg = rolling_averages.RollingAverages()
        self.df_resampled = rolling_avg.apply(self.df_resampled)
        rolling_period_avg = rolling_averages.RollingPeriodAverages()
        self.df_resampled = rolling_period_avg.apply(self.df_resampled)
        rolling_period_iqr = rolling_averages.RollingPeriodIQR()
        self.df_resampled = rolling_period_iqr.apply(self.df_resampled)
        busyness = rolling_averages.PastBusyness(day_period_no)
        self.df_resampled = busyness.apply(self.df_resampled)
        cumulative_avg = rolling_averages.CumulativeAverages()
        self.df_resampled = cumulative_avg.apply(self.df_resampled)
        period_cumulative_avg = rolling_averages.PeriodCumulativeAverages()
        self.df_resampled = period_cumulative_avg.apply(self.df_resampled)
        return self.df_resampled

    def preprocess(self, group_col, split_ratio, target_feature_num,
                   target_enc_cols, ordinal_enc_dict) -> tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data (split and encode the data).

        Args:
            group_col (string): Column name of group column.
            split_ratio (float): Train/test split ratio.
            target_feature_num (str): Numerical version of the target feature.
            target_enc_cols (list): List of column names to target encode.
            ordinal_enc_dict (dict): Dictionary of features to ordinal encode and

        Returns:
            train_data (pd.DataFrame): Processed train data.
            test_data (pd.DataFrame): Processed test data.
        """

        logging.info('Processing the data')
        splitter = preprocessing.DataSplitting()
        self.train_data, self.test_data = splitter.sequential_split(self.df_resampled, group_col, split_ratio)
        target_encoder = preprocessing.TargetEncoding()
        self.train_data, self.test_data = target_encoder.smoothed_target_encode(self.train_data, self.test_data,
                                                                                target_enc_cols, target_feature_num)
        logger.info(f'Train data shape: {self.train_data.shape}')
        logger.info(f'Test data shape: {self.test_data.shape}')
        ordinal_encoder = preprocessing.OrdinalEncoding()
        balanced_train_data, self.test_data = ordinal_encoder.ordinal_encode(self.train_data, self.test_data,
                                                                             ordinal_enc_dict)
        return self.train_data, self.test_data

    def interacting_features(self, shifted_features, feature_groups_multiply, feature_groups_subtract):
        """
        Create additional features for the dataset, resulting from the interaction of current features.

        Args:
            shifted_features (dict): Dictionary of new feature names and corresponding features to shift.
            feature_groups_multiply (dict): Dictionary of new feature names and required features to multiply.
            feature_groups_subtract (dict): Dictionary of new feature names and required features to subtract.

        Returns:
            train_data (pd.DataFrame): Train data with additional features.
            test_data (pd.DataFrame): Test data with additional features.
        """
        logging.info('Creating interaction features')
        interacting_feature_creator = interacting_features.InteractingFeatures()

        # Shifted features
        for shifted_feature, feature_to_shift in shifted_features.items():
            self.train_data = interacting_feature_creator.create_shift(self.train_data, shifted_feature,
                                                                       feature_to_shift)
            self.test_data = interacting_feature_creator.create_shift(self.test_data, shifted_feature, feature_to_shift)

        # Multiplicative interactive features
        for new_feature, feature_group in feature_groups_multiply.items():
            self.train_data = interacting_feature_creator.create_multiplication(
                self.train_data, new_feature, feature_group[0], feature_group[1])
            self.test_data = interacting_feature_creator.create_multiplication(
                self.test_data, new_feature, feature_group[0], feature_group[1])

        # Subtraction interactive features
        for new_feature, feature_group in feature_groups_subtract.items():
            self.train_data = interacting_feature_creator.create_subtraction(
                self.train_data, new_feature, feature_group[0], feature_group[1])
            self.test_data = interacting_feature_creator.create_subtraction(
                self.test_data, new_feature, feature_group[0], feature_group[1])

        drop_features = interacting_features.FeaturesToDrop()
        self.train_data = drop_features.features_to_drop(self.train_data, shifted_features.keys())
        self.test_data = drop_features.features_to_drop(self.test_data, shifted_features.keys())

        return self.train_data, self.test_data

    def train_and_evaluate(self, models, param_grids, training_features, target_feature):

        """
        Orchestrates the entire model training, evaluation, and tuning pipeline. After the initial general training
        an optimized one is realised.

        Args:
            models (dict): Dictionary of model instances.
            param_grids (dict): Dictionary of hyperparameter grids for each model.
            training_features (list): List of column names to use as features.
            target_feature (str): Column name of the target variable.

        Returns:
            tuple: Several score metrics for the models considered, for both the initial and optimized training.
        """
        logging.info('Training and evaluating the model')
        train_test_splitter = data_preprocessing_utils.TrainTestSplit()
        X_train, y_train, X_test, y_test = train_test_splitter.train_test_split(self.train_data, self.test_data,
                                                                                training_features, target_feature)

        X_train.columns = [str(col) for col in X_train.columns]
        X_test.columns = [str(col) for col in X_test.columns]

        target_feature_encoder = data_preprocessing_utils.TargetFeatureEncoder()
        y_train_encoded, y_test_encoded, positive_class_encoded_value, \
            positive_class_string_label, encoder_classes = target_feature_encoder.encode_labels(y_train, y_test)

        logger.debug(f"'Busy' is encoded as: {positive_class_encoded_value}")
        logger.debug(f"LabelEncoder classes: {encoder_classes}")
        logger.debug(f"y_train value counts (raw): {y_train.value_counts()}")
        logger.debug(f"y_train_encoded value counts: {pd.Series(y_train_encoded).value_counts()}")

        # Dictionaries to store initial and threshold optimized results
        initial_run_results = {
            'best_hyperparameters': {},
            'optimal_thresholds': {},
            'metrics_default_threshold': {},
            'metrics_optimal_f1_threshold': {},
            'best_estimators': {}
        }

        # Dictionaries to store results for final optimized run (includes feature importance)
        optimized_run_results = {
            'optimized_model_performance': {},
            'optimized_model_details': {}
        }

        for name, model in models.items():
            logger.info(f"\n{'=' * 50}\nProcessing Model: {name}\n{'=' * 50}")

            # For XGBoost quantify the class imbalance
            if name == 'XGBoost':
                param_grids[name]['scale_pos_weight'].append(
                    np.round(y_train.value_counts()['Not Busy'] / y_train.value_counts()['Busy'], 2))

            model_trainer = model_training.ModelTrainer(model, name, param_grids[name], positive_class_encoded_value,
                                                        positive_class_string_label, encoder_classes)

            best_estimator = model_trainer.train(X_train, y_train_encoded)

            y_pred_proba, y_train_pred = model_trainer.evaluate(X_train, y_train_encoded, X_test, y_test_encoded)

            # Store best estimator
            initial_run_results['best_estimators'][name] = best_estimator
            initial_run_results['best_hyperparameters'][name] = best_estimator.get_params()

            # Store default threshold metrics
            y_pred_default = best_estimator.predict(X_test)
            y_train_pred_default = best_estimator.predict(X_train)
            initial_run_results['metrics_default_threshold'][name] = {
                'accuracy': accuracy_score(y_test_encoded, y_pred_default),
                'accuracy_train': accuracy_score(y_train_encoded, y_train_pred_default),
                'precision': precision_score(y_test_encoded, y_pred_default,
                                             pos_label=positive_class_encoded_value, zero_division=0),
                'recall': recall_score(y_test_encoded, y_pred_default,
                                       pos_label=positive_class_encoded_value, zero_division=0),
                'f1': f1_score(y_test_encoded, y_pred_default, pos_label=positive_class_encoded_value,
                               zero_division=0),
                'roc_auc': roc_auc_score(y_test_encoded, y_pred_proba)
            }

            # Perform Threshold Tuning for initial model
            optimal_threshold_initial, opt_precision_initial, opt_recall_initial, opt_f1_initial = (
                model_trainer.tune_threshold(
                    y_test_encoded, y_pred_proba, y_train_encoded, y_train_pred
                ))

            initial_run_results['optimal_thresholds'][name] = optimal_threshold_initial
            initial_run_results['metrics_optimal_f1_threshold'][name] = {
                'accuracy': accuracy_score(y_test_encoded, (y_pred_proba >= optimal_threshold_initial).astype(int)),
                'precision': opt_precision_initial,
                'recall': opt_recall_initial,
                'f1': opt_f1_initial,
                'roc_auc': roc_auc_score(y_test_encoded, y_pred_proba)
            }

            # Optimized Model Training and Evaluation
            logger.info("\n" + "-" * 50)
            logger.info(f"--- Optimizing {name} with Selected Features ---")
            logger.info("-" * 50 + "\n")

            # Feature Selection based on Importance
            selected_features = X_train.columns.tolist()
            cumulative_feature_selection_threshold = Parameters.CUMULATIVE_FEATURE_IMPORTANCE_THRESHOLD

            if hasattr(best_estimator, 'feature_importances_'):
                importances = best_estimator.feature_importances_
                feature_importances_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values(by='importance', ascending=False)

                # Normalize importances
                feature_importances_df['normalized_importance'] = (feature_importances_df['importance'] /
                                                                   feature_importances_df['importance'].sum())

                # Calculate cumulative importance
                feature_importances_df['cumulative_importance'] = feature_importances_df[
                    'normalized_importance'].cumsum()

                max_desired_feature_no = Parameters.FEATURE_NUMBER_THRESHOLD
                # If the number of features to be kept is bellow a threshold, keep them
                if sum(feature_importances_df['cumulative_importance'] <= cumulative_feature_selection_threshold) <= max_desired_feature_no:
                    high_importance_features_df = feature_importances_df[
                        feature_importances_df['cumulative_importance'] <= cumulative_feature_selection_threshold
                        ]

                    low_importance_features_df = feature_importances_df[
                        feature_importances_df['cumulative_importance'] > cumulative_feature_selection_threshold
                        ]
                else:  # else maintain only the most impactful ones
                    high_importance_features_df = feature_importances_df.iloc[:max_desired_feature_no]

                    low_importance_features_df = feature_importances_df.iloc[max_desired_feature_no:]

                selected_features = [str(feature) for feature in high_importance_features_df['feature'].tolist()]
                non_selected_features = set(X_train.columns) - set(selected_features)
                if len(selected_features) < len(X_train.columns):
                    logger.info(
                        f"[{name}] Selected {len(selected_features)} features out of {len(X_train.columns)}"
                    )
                    logger.info(f"Removed {len(feature_importances_df['cumulative_importance']) - len(selected_features)}"
                                f" features for {name}: {non_selected_features}")

                    # extract plot of feature importances
                    visualizer = Visualizer(model, name, positive_class_encoded_value, encoder_classes)
                    visualizer.plot_feature_importances(high_importance_features_df, low_importance_features_df)
                else:
                    logger.info(
                        f"[{name}] No features removed based on cumulative importance threshold - all features contribute sufficiently.")

            else:
                logger.warning(
                    f"[{name}] Model does not have feature importances. Using all features for optimization.")

            # Prepare data with selected features
            X_train_optimized = X_train[selected_features]
            X_test_optimized = X_test[selected_features]

            # Retrain the model with selected features and best hyperparameters
            logger.info(f"\nRetraining {name} with selected features and best hyperparameters...")

            # Create a new instance of the model type and set its best parameters
            re_trained_model = type(model)()
            re_trained_model.set_params(**initial_run_results['best_hyperparameters'][name])

            re_trained_model.fit(X_train_optimized, y_train_encoded)

            # Calculate y_train_pred_optimized after retraining ---
            y_train_pred_optimized = re_trained_model.predict(X_train_optimized)

            y_pred_proba_optimized = re_trained_model.predict_proba(X_test_optimized)[:, positive_class_encoded_value]

            # Calculate predicted labels for the TEST set
            y_test_pred_optimized = re_trained_model.predict(X_test_optimized)

            # Re-run threshold tuning on the new model's probabilities
            optimal_threshold_retrained, opt_precision_retrained, opt_recall_retrained, opt_f1_retrained = \
                model_trainer.tune_threshold(
                    y_test_encoded, y_pred_proba_optimized, y_train_encoded, y_train_pred_optimized
                )

            # Store results of the optimized model
            optimized_run_results['optimized_model_performance'][name] = {
                'optimal_threshold': optimal_threshold_retrained,
                'accuracy': accuracy_score(y_test_encoded,
                                           (y_pred_proba_optimized >= optimal_threshold_retrained).astype(int)),
                'precision': opt_precision_retrained,
                'recall': opt_recall_retrained,
                'f1': opt_f1_retrained,
                'roc_auc': roc_auc_score(y_test_encoded, y_pred_proba_optimized),
                'selected_features_count': len(selected_features)
            }
            optimized_run_results['optimized_model_details'][name] = {
                'model_instance': re_trained_model,
                'selected_features': selected_features
            }

        # Store all results as an instance attribute
        self.model_results = {
            'initial_run': initial_run_results,
            'optimized_run': optimized_run_results
        }

        return self.model_results
