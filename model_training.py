from parameters import Parameters
from vizualisation import Visualizer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc, precision_recall_curve
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelTrainer:
    """Trains the selected model."""
    def __init__(self, model, model_name, param_grid, positive_class_value, positive_class_label, encoder_classes):
        """
        Args:
            model: Model used (e.g. XGBoost).
            model_name (str): Model name.
            param_grid (dict): Dictionary of parameters to tune.
            positive_class_value (int): Value of positive class.
            positive_class_label (str): Label of positive class.
            encoder_classes (numpy.ndarray): Array of unique class labels in the order they were encoded.
        """
        self.model = model
        self.model_name = model_name
        self.param_grid = param_grid
        self.best_estimator = None
        self.positive_class_value = positive_class_value
        self.positive_class_label = positive_class_label
        self.encoder_classes = encoder_classes
        self.visualizer = Visualizer(model, model_name, positive_class_value, encoder_classes)

    def train(self, X_train, y_train_encoded) -> BaseEstimator:
        """
        Train the selected model.

        Args:
            X_train (pd.DataFrame): Features from training set.
            y_train_encoded (pd.Series):  Encoded target variable from training set.

        Returns:
            best_estimator (BaseEstimator): Best parameters for the model.
        """
        search = RandomizedSearchCV(
            self.model, self.param_grid, n_iter=Parameters.N_ITERATIONS,
            cv=Parameters.CV, scoring='f1', n_jobs=Parameters.N_JOBS,
            random_state=Parameters.SEED, verbose=Parameters.VERBOSE
        )
        search.fit(X_train, y_train_encoded)
        self.best_estimator = search.best_estimator_

        logger.info(f"Best Parameters for {self.model_name}: {search.best_params_}")
        logger.info(f"Best Cross-validation F1-Score: {search.best_score_:.2f}")
        return self.best_estimator

    def evaluate(self, X_train, y_train_encoded, X_test, y_test_encoded) -> tuple[pd.Series, pd.Series]:
        """Evaluate the model on the test set.

        Args:
            X_train (pd.DataFrame): Features from training set.
            y_train_encoded (pd.Series):  Encoded target variable from training set.
            X_test (pd.DataFrame): Features from test set.
            y_test_encoded (pd.Series):  Encoded target variable from test set.

        Returns:
            y_pred_proba(pd.Series): Predicted probabilities for the positive class on the test set.
            y_train_pred(pd.Series): Predicted labels for the training set.
        """
        y_train_pred = self.best_estimator.predict(X_train)
        y_pred = self.best_estimator.predict(X_test)
        y_pred_proba = self.best_estimator.predict_proba(X_test)[:, self.positive_class_value]

        self._print_metrics(y_test_encoded, y_pred, y_train_encoded, y_train_pred, y_pred_proba, "Default Threshold")
        return y_pred_proba, y_train_pred

    def tune_threshold(self, y_test_encoded, y_pred_proba, y_train_encoded, y_train_pred) -> tuple[float, float, float, float]:
        """
        Tunes the classification threshold for the model to optimize F1-score.
        This method analyzes the precision-recall curve to find the optimal threshold
        that maximizes the F1-score on the test set and visualises the results.

        Args:
            y_test_encoded (pd.Series): Encoded true labels for the test set.
            y_pred_proba (pd.Series): Predicted probabilities for the positive class on the test set.
            y_train_encoded (pd.Series): Encoded true labels for the training set.
            y_train_pred (pd.Series): Predicted labels for the training set.

        Returns:
            tuple: A tuple containing:
                - optimal_threshold (float): The threshold that maximizes the F1-score.
                - optimal_precision (float): Precision at the optimal threshold.
                - optimal_recall (float): Recall at the optimal threshold.
                - optimal_f1_score (float): The maximized F1-score.
        """

        logger.info(f"\n--- Threshold Tuning for Optimal F1-Score for {self.model_name} ---")
        precision_points, recall_points, thresholds = precision_recall_curve(
            y_test_encoded, y_pred_proba, pos_label=self.positive_class_value)
        f1_scores = 2 * (precision_points * recall_points) / (precision_points + recall_points + 1e-10)

        default_threshold_idx = np.argmin(np.abs(thresholds - 0.5))

        optimal_f1_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_f1_threshold_idx]
        optimal_precision = precision_points[optimal_f1_threshold_idx]
        optimal_recall = recall_points[optimal_f1_threshold_idx]
        optimal_f1_score = f1_scores[optimal_f1_threshold_idx]

        logger.info(f"Optimal Threshold: {optimal_threshold:.2f}, F1: {f1_scores[optimal_f1_threshold_idx]:.2f}")

        logger.info(f"\nDefault (0.5) Threshold Performance:")
        logger.info(
            f"Precision: {precision_points[default_threshold_idx]:.2f}, Recall: {recall_points[default_threshold_idx]:.2f}, F1-score: {f1_scores[default_threshold_idx]:.2f}")
        logger.info(f"Optimal F1 Threshold Performance:")
        logger.info(f"Threshold: {optimal_threshold:.2f}")
        logger.info(
            f"Precision: {optimal_precision:.2f}, Recall: {optimal_recall:.2f}, F1-score: {optimal_f1_score:.2f}")

        # Apply the optimal threshold
        y_pred_tuned = (y_pred_proba >= optimal_threshold).astype(int)

        self._print_metrics(y_test_encoded, y_pred_tuned, y_train_encoded, y_train_pred, y_pred_proba,
                            "Optimal F1 Threshold")

        self.visualizer.plot_precision_recall(recall_points, precision_points, thresholds, optimal_recall, optimal_precision, optimal_threshold, optimal_f1_score)
        self.visualizer.plot_f1_score_threshold(thresholds, f1_scores, optimal_threshold)

        return optimal_threshold, optimal_precision, optimal_recall, optimal_f1_score

    def _print_metrics(self, y_true, y_pred, y_train_true, y_train_pred, y_pred_proba, threshold_type) -> None:
        """
        Computes accuracy, precision, recall, F1-score, ROC AUC,
        and the confusion matrix, then logs them along with a classification report.

        Args:
            y_true (pd.Series): True labels for the test set.
            y_pred (np.ndarray): Predicted labels for the test set.
            y_train_true (pd.Series): True labels for the training set.
            y_train_pred (np.ndarray): Predicted labels for the training set.
            y_pred_proba (np.ndarray): Predicted probabilities for the positive class on the test set.
            threshold_type (str): A string indicating the type of threshold used (e.g. "Optimal F1 Threshold").

        Returns:
            None: This function does not return any value; it prints metrics directly.
        """
        logger.info(f"\n--- Model Performance Metrics ({threshold_type}) ---")

        accuracy = accuracy_score(y_true, y_pred)
        train_accuracy = accuracy_score(y_train_true, y_train_pred)
        precision = precision_score(y_true, y_pred, pos_label=self.positive_class_value, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=self.positive_class_value, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=self.positive_class_value, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred)

        logger.info(f"Accuracy for {self.model_name}: {accuracy:.2f} | Training Accuracy {self.model_name}: {train_accuracy:.2f}")
        logger.info(f"Precision for '{self.positive_class_label}': {precision:.2f} | Recall for '{self.positive_class_label}': {recall:.2f} \n"
                    f" | F1 for '{self.positive_class_label}': {f1:.2f} | ROC AUC: {roc_auc:.2f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        self.visualizer.plot_confusion_matrix(cm, threshold_type)
        self.visualizer.plot_roc_curve(y_true, y_pred_proba)
        logger.info(classification_report(y_true, y_pred, target_names=self.encoder_classes, zero_division=0))
