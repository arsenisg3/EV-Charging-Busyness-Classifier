from parameters import Parameters

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (roc_curve, auc)
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Visualizer:
    """Generates plots visualizing the results."""
    def __init__(self, model, model_name, positive_class_value, encoder_classes):
        """
        Args:
            model: Model used (e.g. XGBoost).
            model_name (str): Model name.
            positive_class_value (int): Value of positive class.
            encoder_classes (numpy.ndarray): Array of unique class labels in the order they were encoded.
        """
        self.model = model
        self.model_name = model_name
        self.positive_class_value = positive_class_value
        self.encoder_classes = encoder_classes
        os.makedirs(Parameters.PLOT_PATH, exist_ok=True)

    def plot_confusion_matrix(self, cm, threshold_type) -> None:
        """Plots the confusion matrix.

        Args:
            cm: Confusion matrix.
            threshold_type (str): A string indicating the type of threshold used (e.g. "Optimal F1 Threshold").
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.encoder_classes,
                    yticklabels=self.encoder_classes)
        title = f'Confusion Matrix ({self.model_name} - {threshold_type})'
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(Parameters.PLOT_PATH, f'Confusion_Matrix_({self.model_name}_{threshold_type}).png'))
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba) -> None:
        """Plots the roc curve.

        Args:
            y_true (pandas.Series): True binary labels (0 or 1).
            y_pred_proba (numpy.ndarray): Predicted probabilities of the positive class.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=self.positive_class_value)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='navy')
        title = f'ROC Curve for ({self.model_name})'
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Parameters.PLOT_PATH, f'ROC_Curve_for_{self.model_name}.png'))
        plt.close()

    def plot_precision_recall(self, recall_points, precision_points, thresholds, optimal_recall, optimal_precision,
                              optimal_threshold, optimal_f1_score):
        """ Plots the Precision-Recall curve, highlighting the default and optimal F1 thresholds.

        Args:
            recall_points (numpy.ndarray): Array of recall values corresponding to different thresholds.
            precision_points (numpy.ndarray): Array of precision values corresponding to different thresholds.
            thresholds (numpy.ndarray): Array of probability thresholds.
            optimal_recall (numpy.ndarray): Recall value at optimal F1 threshold.
            optimal_precision (numpy.ndarray): Precision value at optimal F1 threshold.
            optimal_threshold (numpy.ndarray): Probability threshold that maximizes the F1-score.
            optimal_f1_score (numpy.ndarray): Maximum F1-score achieved.

        """
        plt.figure(figsize=(10, 7))
        plt.plot(recall_points, precision_points, color='blue', lw=2, label='Precision-Recall Curve')

        default_threshold_idx = np.argmin(np.abs(thresholds - 0.5))
        plt.plot(recall_points[default_threshold_idx], precision_points[default_threshold_idx],
                 'o', markersize=10, color='red',
                 label=f'Default 0.5 Threshold (P={precision_points[default_threshold_idx]:.2f}, R={recall_points[default_threshold_idx]:.2f})')

        plt.plot(optimal_recall, optimal_precision, 'X', markersize=12, color='green',
                 label=f'Optimal F1 Threshold ({optimal_threshold:.2f}) (P={optimal_precision:.2f}, R={optimal_recall:.2f}, F1={optimal_f1_score:.2f})')

        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision (Positive Predictive Value)')
        title = f'Precision-Recall Curve for {self.model_name} (Threshold Tuning)'
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.savefig(os.path.join(Parameters.PLOT_PATH, f'Precision_Recall_Curve_for_{self.model_name}_(Threshold Tuning).png'))
        plt.close()

    def plot_f1_score_threshold(self, thresholds, f1_scores, optimal_threshold) -> None:
        """Plots F1-score vs Threshold.

        Args:
            thresholds (numpy.ndarray): Array of thresholds.
            f1_scores (numpy.ndarray): Array of F1-scores.
            optimal_threshold (float): Threshold that maximizes F1-score.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(thresholds, f1_scores[:-1], color='purple', lw=2)
        plt.axvline(x=optimal_threshold, color='green', linestyle='--',
                    label=f'Optimal F1 Threshold: {optimal_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')
        title = f'F1-score vs. Threshold for {self.model_name}'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Parameters.PLOT_PATH, f'F1-score_vs_Threshold_for_{self.model_name}.png'))
        plt.close()

    def plot_feature_importances(self, high_importance_features_df, low_importance_features_df) -> None:
        """Plots the importances of the selected high importance features.

        Args:
            high_importance_features_df (DataFrame): DataFrame containing high importance training features to be kept.
            low_importance_features_df (DataFrame): DataFrame containing low importance training features to be removed.
        """

        logger.info(f"\n--- Considered features and their importances for {self.model_name} ---")
        logger.info(high_importance_features_df[['feature', 'normalized_importance', 'cumulative_importance']])

        plt.figure(figsize=(12, min(8, int(high_importance_features_df.shape[0] * 0.5))))
        sns.barplot(x='importance', y='feature', data=high_importance_features_df, palette='viridis')
        title = f'Selected Feature Importances for {self.model_name}'
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(Parameters.PLOT_PATH, f'Selected_Feature_Importances_for_{self.model_name}.png'))
        plt.close()

        if not low_importance_features_df.empty:
            logger.info(f"\n--- Not Considered Features for {self.model_name} and their Importance ---")
            logger.info(low_importance_features_df[['feature', 'normalized_importance', 'cumulative_importance']])
