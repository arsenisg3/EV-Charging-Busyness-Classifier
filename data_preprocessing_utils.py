from parameters import Parameters

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TrainTestSplit:
    """Splits data into training and test sets, each with training features and target variable."""
    @staticmethod
    def train_test_split(train_data, test_data, training_features, target_feature) \
            -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            training_features (list): Features to be considered for training.
            target_feature (str): Target feature.

        Returns:
            X_train (pd.DataFrame): Features from training set.
            y_train (pd.Series): Target variable from training set.
            X_test (pd.DataFrame): Features from test set.
            y_test (pd.Series): Target variable from test set.
        """
        X_train = train_data[training_features]
        y_train = train_data[target_feature]
        X_test = test_data[training_features]
        y_test = test_data[target_feature]
        return X_train, y_train, X_test, y_test


class TargetFeatureEncoder:
    """Encode the target variable."""
    @staticmethod
    def encode_labels(y_train, y_test):
        """
        Args:
            y_train (pd.Series): Target variable from training set.
            y_test (pd.Series): Target variable from test set.

        Returns:
            y_train_encoded (pd.Series): Encoded target variable from training set.
            y_test_encoded (pd.Series): Encoded target variable from test set.
            positive_class_value (int): Value of positive class.
            positive_class_label (str): Label of positive class.
            encoder_classes (numpy.ndarray): Array of unique class labels in the order they were encoded.
        """
        encoder = LabelEncoder()
        encoder.classes_ = np.array(Parameters.BUSYNESS_ORDER)
        y_train_encoded = encoder.transform(y_train)
        y_test_encoded = encoder.transform(y_test)

        positive_class_value = encoder.transform([Parameters.POSITIVE_CLASS_VALUE])[0]
        positive_class_label = Parameters.POSITIVE_CLASS_VALUE
        encoder_classes = encoder.classes_

        return y_train_encoded, y_test_encoded, positive_class_value, positive_class_label, encoder_classes
