import pandas as pd
from data_fields import SessionFields as SF, TrainingFields as TF
from sklearn.preprocessing import OrdinalEncoder
from parameters import Parameters


class DataSplitting:
    """Splits data into training and testing sets."""
    @staticmethod
    def sequential_split(df, group_col, split_ratio=0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            df (pd.DataFrame): Full dataset.
            group_col (string): Column name of group column.
            split_ratio (float): Train/test split ratio.

        Returns:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
        """
        train_data = []
        test_data = []

        # Group by the specified column
        for _, group in df.groupby(group_col):
            # Sort each group by date (or other relevant column)
            group = group.sort_values(by=TF.START_DATE)

            # Determine the split point
            split_idx = int(len(group) * split_ratio)

            # Split into training and test sets
            train_data.append(group.iloc[:split_idx])
            test_data.append(group.iloc[split_idx:])

        # Concatenate results
        train_data = pd.concat(train_data)
        test_data = pd.concat(test_data)

        return train_data, test_data


class TargetEncoding:
    """Target encodes the desired features."""
    @staticmethod
    def smoothed_target_encode(train_data, test_data, target_enc_cols, target_feature_num, alpha=10) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            target_enc_cols (list): List of column names to target encode.
            target_feature_num (str): Numerical version of the target feature.
            alpha (float): Alpha parameter for target encoding.

        Returns:
            train_data (pd.DataFrame): Training data with updated target encoding.
            test_data (pd.DataFrame): Test data with updated target encoding.
        """

        for target_enc_col in target_enc_cols:
            target_means = train_data.groupby(target_enc_col)[target_feature_num].mean()
            global_mean = train_data[target_feature_num].mean()

            # Smoothed means
            smoothed_means = (target_means * train_data.groupby(target_enc_col)[target_feature_num].count() + global_mean * alpha) / \
                             (train_data.groupby(target_enc_col)[target_feature_num].count() + Parameters.ALPHA)

            # Map to train and test
            train_data[f"{target_enc_col}_encoded"] = train_data[target_enc_col].map(smoothed_means)
            test_data[f"{target_enc_col}_encoded"] = test_data[target_enc_col].map(smoothed_means).fillna(global_mean)
        return train_data, test_data


class OrdinalEncoding:
    """Ordinal encodes the desired features."""
    @staticmethod
    def ordinal_encode(train_data, test_data, ordinal_enc_dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            ordinal_enc_dict (dict): Dictionary of features to ordinal encode and
            the corresponding ordered categorical data.

        Returns:
            train_data (pd.DataFrame): Training data with updated target encoding.
            test_data (pd.DataFrame): Test data with updated target encoding.
        """
        for ordinal_enc_col, order in ordinal_enc_dict.items():
            encoder = OrdinalEncoder(categories=[order])
            train_data[f'{ordinal_enc_col}_encoded'] = encoder.fit_transform(train_data[[ordinal_enc_col]]).astype(int)
            test_data[f'{ordinal_enc_col}_encoded'] = encoder.transform(test_data[[ordinal_enc_col]]).astype(int)
        return train_data, test_data
