#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd


class Injector:
    def __init__(self, error_seq=list()):
        """Initialize with the sequence of errors to be injected."""
        self.error_seq = error_seq

    def inject(self, data_X, data_y, data_X_orig=None, data_y_orig=None, seed=None):
        """Inject the sequence of errors into the data. Return the dirty data with injected errors."""
        np.random.seed(seed)
        copy_X_data = data_X.copy()
        copy_y_data = data_y.copy()
        if data_X_orig is None:
            copy_X_orig = data_X.copy()
        else:
            copy_X_orig = data_X_orig.copy()
        if data_y_orig is None:
            copy_y_orig = data_y.copy()
        else:
            copy_y_orig = data_y_orig.copy()
        for err in self.error_seq:
            copy_X_data, copy_y_data, copy_X_orig, copy_y_orig = err.inject(copy_X_data, copy_y_data, copy_X_orig, copy_y_orig, seed)
        return copy_X_data, copy_y_data, copy_X_orig, copy_y_orig
    
    def load_seq(self, filename):
        """Load the sequence of errors from a yaml/json file."""
        raise NotImplementedError

class DataError:
    def inject(self, data_X, data_y, data_X_orig, data_y_orig):
        """Inject this error into the specified data."""
        raise NotImplementedError


class MissingValueError(DataError):
    def __init__(self, column, pattern=None, ratio=0.3, seed=42):
        """Initialize with the column in which to inject missing values, the pattern that describes the subset of data suffering from missing value errors, and the ratio of values to remove."""
        self.pattern = pattern 
        self.column = column
        self.ratio = ratio
        self.seed = seed

    def inject(self, data_X, data_y, data_X_orig, data_y_orig, seed=None):
        """Randomly replace some values in the specified column with NaN.
        If self.pattern is not None, inject missing values only to the tuples that are described by the pattern."""
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        if self.pattern is None:
            nan_indices = np.random.choice(data_X_orig.shape[0], int(data_X.shape[0] * self.ratio), replace=False)
            data_X[nan_indices, self.column] = np.nan
        else:
            binary_indicator = self.pattern(data_X_orig, data_y_orig)
            pattern_indices = np.where(binary_indicator == 1)[0]
            num_to_replace = int(len(pattern_indices) * self.ratio)
            replace_indices = np.random.choice(pattern_indices, size=num_to_replace, replace=False)
            data_X.iloc[replace_indices, self.column] = np.nan
        return data_X, data_y, data_X_orig, data_y_orig


class SamplingError(DataError):
    def __init__(self, pattern=None, ratio=0.3, seed=42):
        """Initialize with the pattern that describes the subset of data suffering from sampling errors and the ratio of values to remove."""
        self.pattern = pattern
        self.ratio = ratio
        self.seed = seed

    def inject(self, data_X, data_y, data_X_orig, data_y_orig, seed=None):
        """Randomly drop tuples based on some attribute. 
        If self.pattern is not None, inject missing values only to the tuples that are described by the pattern."""
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        if self.pattern is None:
            rows_to_drop = np.random.choice(data_X_orig.shape[0], int(data_X.shape[0] * self.ratio), replace=False)
            data_X = data_X.drop(index=rows_to_drop)
            data_y = data_y.drop(index=rows_to_drop)
            data_X_orig = data_X_orig.drop(index=rows_to_drop)
            data_y_orig = data_y_orig.drop(index=rows_to_drop)
        else:
            binary_indicator = self.pattern(data_X_orig, data_y_orig)
            pattern_indices = np.where(binary_indicator == 1)[0]
            num_to_drop = int(len(pattern_indices) * self.ratio)
            rows_to_drop = np.random.choice(pattern_indices, size=num_to_drop, replace=False)
            data_X = data_X.drop(index=rows_to_drop)
            data_y = data_y.drop(index=rows_to_drop)
            data_X_orig = data_X_orig.drop(index=rows_to_drop)
            data_y_orig = data_y_orig.drop(index=rows_to_drop)
        data_X = data_X.reset_index(drop=True)
        data_y = data_y.reset_index(drop=True)
        data_X_orig = data_X.reset_index(drop=True)
        data_y_orig = data_y.reset_index(drop=True)
        return data_X, data_y, data_X_orig, data_y_orig


class DuplicateError(DataError):
    def __init__(self, pattern=None, ratio=0.3, seed=42):
        """Initialize with the pattern that describes the subset of data suffering from deplicate errors, the ratio of rows to duplicate."""
        self.pattern = pattern
        self.ratio = ratio
        self.seed = seed

    def inject(self, data_X, data_y, data_X_orig, data_y_orig, seed=None):
        """Randomly duplicate some rows. Randomly duplicate tuples. If self.pattern is not None, randomly duplicate tuples that are described by the pattern."""
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        if self.pattern is None:
            num_to_duplicate = int(len(data_X_orig) * self.ratio)
            duplicated_rows_X_orig = data_X_orig.sample(n=num_to_duplicate, replace=True, random_state=42)
            duplicated_rows_y_orig = data_y_orig[duplicated_rows_X_orig.index]
            duplicated_rows_y = data_y[duplicated_rows_X_orig.index]
            duplicated_rows_X = data_X[duplicated_rows_X_orig.index]
            data_X = pd.concat([data_X, duplicated_rows_X], ignore_index=True)
            data_y = pd.concat([data_y, duplicated_rows_y], ignore_index=True)
            data_X_orig = pd.concat([data_X_orig, duplicated_rows_X_orig], ignore_index=True)
            data_y_orig = pd.concat([data_y_orig, duplicated_rows_y_orig], ignore_index=True)
        else:
            binary_indicator = self.pattern(data_X_orig, data_y_orig)
            pattern_indices = np.where(binary_indicator == 1)[0]
            num_to_duplicate = int(len(pattern_indices) * self.ratio)
            replace_indices = np.random.choice(pattern_indices, size=num_to_duplicate, replace=False)
            duplicated_rows_X = data_X.iloc[replace_indices]
            duplicated_rows_y = data_y.iloc[replace_indices]
            duplicated_rows_X_orig = data_X_orig.iloc[replace_indices]
            duplicated_rows_y_orig = data_y_orig.iloc[replace_indices]
            data_X = pd.concat([data_X, duplicated_rows_X], ignore_index=True)
            data_y = pd.concat([data_y, duplicated_rows_y], ignore_index=True)
            data_X_orig = pd.concat([data_X_orig, duplicated_rows_X_orig], ignore_index=True)
            data_y_orig = pd.concat([data_y_orig, duplicated_rows_y_orig], ignore_index=True)
        data_X = data_X.reset_index(drop=True)
        data_y = data_y.reset_index(drop=True)
        data_X_orig = data_X_orig.reset_index(drop=True)
        data_y_orig = data_y_orig.reset_index(drop=True)
        return data_X, data_y, data_X_orig, data_y_orig


class LabelError(DataError):
    def __init__(self, pattern=None, ratio=0.3, seed=42):
        """Initialize with the pattern that describes the subset of data suffering from label flip errors, and the ratio of labels to relabel."""
        self.pattern = pattern
        self.ratio = ratio
        self.seed = seed

    def inject(self, data_X, data_y, data_X_orig, data_y_orig, seed=None):
        """Randomly flip some labels. If self.pattern is not None, flip labels only for tuples that are described by the pattern."""
        num_classes = len(set(data_y_orig))
        target_set = set(range(num_classes))
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        if self.pattern is None:
            num_rows_to_modify = int(len(data_y_orig) * self.ratio)
            indices_to_modify = np.random.choice(data_y_orig.index, num_rows_to_modify, replace=False)
            data_y.iloc[indices_to_modify] = data_y.iloc[indices_to_modify].apply(
                lambda x: np.random.choice(list(target_set - {x})))
        else:
            binary_indicator = self.pattern(data_X_orig, data_y_orig)
            pattern_indices = np.where(binary_indicator == 1)[0]
            num_to_modify = int(len(pattern_indices) * self.ratio)
            indices_to_modify = np.random.choice(pattern_indices, num_to_modify, replace=False)
            data_y.iloc[indices_to_modify] = data_y.iloc[indices_to_modify].apply(
                lambda x: np.random.choice(list(target_set - {x})))
        return data_X, data_y, data_X_orig, data_y_orig

class OutlierError(DataError):
    def __init__(self, column, pattern=None, ratio=0.3, multiplier=1.5, seed=42):
        """Initialize with the column index in which to introduce outliers and the multiplier to apply to the values."""
        self.column = column
        self.pattern = pattern
        self.ratio = ratio
        self.multiplier = multiplier
        self.seed = seed

    def inject(self, data_X, data_y, data_X_orig, data_y_orig, seed=None):
        """Inject outliers into the specified column by multiplying all values (excluding NaNs) by the user-defined multiplier."""
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        # Check if the selected column is numerical
        column_name = data_X_orig.columns[self.column]
        if not np.issubdtype(data_X_orig[column_name].dtype, np.number):
            raise ValueError(f"Column '{column_name}' is not numerical.")

        if self.pattern is None:
            # Apply the multiplier to non-NaN values in the specified column
            data_X[column_name] = data_X[column_name].astype(float)
            data_X[column_name] = data_X[column_name].apply(lambda x: x * self.multiplier if not pd.isna(x) else x)
        else:
            binary_indicator = self.pattern(data_X_orig, data_y_orig)
            pattern_indices = np.where(binary_indicator == 1)[0]
            num_to_multiply = int(len(pattern_indices) * self.ratio)
            indices_to_multiply = np.random.choice(pattern_indices, num_to_multiply, replace=False)
            data_X[column_name] = data_X[column_name].astype(float)
            data_X.loc[indices_to_multiply, column_name] = data_X.loc[indices_to_multiply, column_name].apply(lambda x: x * self.multiplier if not pd.isna(x) else x)
        return data_X, data_y, data_X_orig, data_y_orig

