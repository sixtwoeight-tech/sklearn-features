""" Short summary

Longer summary with more details.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


DAYS_OF_WEEK = ["Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                ]

MONTHS_OF_YEAR = ["January",
                  "February",
                  "March",
                  "April",
                  "May",
                  "June",
                  "July",
                  "August",
                  "September",
                  "October",
                  "November",
                  "December",
                  ]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Transforms a DataFrame into a Series by selecting a single column by key.
    """

    DEFAULT_PIPELINE_NAME = 'data_frame_selector'

    def __init__(self, key):
        self._key = key

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df[self._key]


class SeriesReshaper(BaseEstimator, TransformerMixin):
    """
    Transforms a Series of size N into an (N, 1) shaped numpy array.
    """

    DEFAULT_PIPELINE_NAME = 'series_reshaper'

    def fit(self, ds, y=None):
        self.feature_name_ = ds.name
        return self

    def transform(self, ds):
        return ds.values.reshape(-1, 1)

    def get_feature_names(self):
        return [self.feature_name_]


class DataFrameReshaper(BaseEstimator, TransformerMixin):
    """
    Transforms a DataFrame of size NxM into an (N, M) shaped numpy array.
    """

    DEFAULT_PIPELINE_NAME = 'dataframe_reshaper'

    def fit(self, df, y=None):
        self.feature_names_ = list(df.keys())

    def transform(self, df):
        return df.values

    def get_feature_names(self):
        return self.feature_names_


class _SeriesTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a Series into a new Series.
    """

    def __init__(self, feature_name=None):
        self.feature_name_ = feature_name

    def fit(self, ds, y=None):
        if self.feature_name_ is None:
            self.feature_name_ = ds.name
        self._fit(ds, y)
        return self

    def _fit(self, ds, y):
        raise NotImplementedError()

    def transform(self, ds):
        return self._transform(ds)

    def _transform(self, ds):
        raise NotImplementedError()

    def get_feature_names(self):
        return [self.feature_name_]


class _SeriesDataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names_ = feature_names

    def fit(self, ds, y=None):
        self._fit(ds, y)
        return self

    def _fit(self, ds, y):
        raise NotImplementedError()

    def transform(self, ds):
        return self._transform(ds)

    def _transform(self, ds):
        raise NotImplementedError()

    def get_feature_names(self):
        return self.feature_names_


class NullTransformer(_SeriesTransformer):
    """ Pass through the Series completely unchanged.
    """

    DEFAULT_PIPELINE_NAME = 'null'

    def _fit(self, ds, y):
        pass

    def _transform(self, ds):
        return ds


class ScalingTransformer(_SeriesTransformer):
    """ Apply a constant scaling factor to a Series.
    """

    DEFAULT_PIPELINE_NAME = 'scaling'

    def __init__(self, scaling_factor):
        super().__init__()
        self._scaling_factor = scaling_factor

    def _fit(self, ds, y):
        pass

    def _transform(self, ds):
        return self._scaling_factor * ds


class DateAttributeTransformer(_SeriesTransformer):
    """ Select a particular attribute from the .dt property of a Series.

    https://pandas.pydata.org/pandas-docs/stable/api.html#datetimelike-properties
    """
    DEFAULT_PIPELINE_NAME = 'datetime_attribute'

    def __init__(self, attr):
        super().__init__()
        self._attr = attr

    def _fit(self, ds, y):
        pass

    def _transform(self, ds):
        return getattr(ds.dt, self._attr)


class MultiDateTransformer(_SeriesTransformer):

    DEFAULT_PIPELINE_NAME = 'multidate'

    def __init__(self, dates):
        super().__init__()
        self._dates = dates

    def _fit(self, ds, y):
        pass

    def _transform(self, ds):
        return ds.dt.date.isin(self._dates)


class LinearDateTransformer(_SeriesTransformer):
    """ Convert a datetime Series into a float Series.

    Perform a linear transformation based on `d0` and `delta`.

    Defaults:
     `d0`: training_ds.min()
     `delta`: 1 day
    """

    DEFAULT_PIPELINE_NAME = 'linear_date'

    def __init__(self, d0=None, delta=pd.Timedelta(1, 'D')):
        super().__init__()
        self.d0_ = d0
        self.delta_ = delta

    def _fit(self, ds, y):
        if self.d0_ is None:
            self.d0_ = ds.min()

    def _transform(self, ds):
        return (ds - self.d0_) / self.delta_


class LabelEncoderWithUnknown(_SeriesTransformer):
    """ Convert a categorical feature into values [0, n], where
    [0, n) represent the known categories from the training data and
    n represents unknown data.
    """

    DEFAULT_PIPELINE_NAME = 'labels_with_unknown'

    def _fit(self, ds, y):
        self.classes_ = ds.unique()
        self.classes_.sort()
        return self

    def _transform(self, ds):
        ret = pd.Series(self.classes_.searchsorted(ds), index=ds.index, name=ds.name)
        return ret.where(ds.isin(self.classes_), len(self.classes_))


class OneHotWithUnknown(_SeriesDataFrameTransformer):

    DEFAULT_PIPELINE_NAME = 'onehot_with_unknown'

    def _fit(self, ds, y):
        self.n_cols_ = len(ds.unique()) + 1  # +1 to allow for missing categories
        if self.feature_names_ is None:
            self.feature_names_ = [str(x) for x in range(self.n_cols_ - 1)] + ['missing']

    def _transform(self, ds):
        df = pd.DataFrame(False, dtype=bool, columns=self.feature_names_, index=ds.index)
        for x in range(self.n_cols_):
            df[self.feature_names_[x]] = ds == x
        return df


class OneHotWithFixedFeatures(_SeriesDataFrameTransformer):

    DEFAULT_PIPELINE_NAME = 'onehot_with_fixed_features'

    def _fit(self, ds, y):
        self.n_cols_ = len(self.feature_names_)

    def _transform(self, ds):
        df = pd.DataFrame(False, dtype=bool, columns=self.feature_names_, index=ds.index)
        for x in range(self.n_cols_):
            df[self.feature_names_[x]] = ds == x
        return df


class _Pipeline(Pipeline):

    def __init__(self, steps, memory=None):
        cleaned_steps = []
        for step in steps:
            if not isinstance(step, tuple):
                step = (step.DEFAULT_PIPELINE_NAME, step)
            cleaned_steps.append(step)
        super().__init__(cleaned_steps, memory)

    def get_feature_names(self):
        return self.steps[-1][-1].get_feature_names()


def series_pipeline(key, steps):
    return _Pipeline(steps=[DataFrameSelector(key)] + steps + [SeriesReshaper()])


def dataframe_pipeline(key, steps):
    return _Pipeline(steps=[DataFrameSelector(key)] + steps + [DataFrameReshaper()])
