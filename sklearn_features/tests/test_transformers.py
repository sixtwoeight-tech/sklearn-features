from datetime import datetime, date
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion


from sklearn_features.transformers import ScalingTransformer, NullTransformer, DateAttributeTransformer, OneHotWithFixedFeatures, MultiDateTransformer, LinearDateTransformer, LabelEncoderWithUnknown, OneHotWithUnknown
from sklearn_features.transformers import DataFrameSelector, SeriesReshaper, series_pipeline, DAYS_OF_WEEK, dataframe_pipeline


def _create_test_data():
    df = pd.DataFrame({"col_A": [1, 2, 3, 2, 1],
                       "col_B": [1.0, 2.0, 3.0, 2.0, 1.0],
                       "dates": [datetime(2014, 2, 3),
                                 datetime(2014, 10, 4),
                                 datetime(2015, 3, 10),
                                 datetime(2016, 5, 30),
                                 datetime(2018, 10, 12),
                                 ]
                       })
    return df


def test_null_transformer():
    df = _create_test_data()
    for column in ["col_A", "col_B"]:
        transformer = NullTransformer()
        transformer.fit(df[column])
        ds = transformer.transform(df[column])
        assert (ds == df[column]).all()


def test_scaling_transformer():
    df = _create_test_data()
    for column in ["col_A", "col_B"]:
        scaling_factor = 2.0
        transformer = ScalingTransformer(scaling_factor)
        transformer.fit(df[column])
        ds = transformer.transform(df[column])
        assert (ds == scaling_factor * df[column]).all()


def test_date_attribute_transformer():
    df = _create_test_data()
    transformer = DateAttributeTransformer('dayofweek')
    transformer.fit(df['dates'])
    ret = transformer.transform(df['dates'])
    assert (ret == df['dates'].dt.dayofweek).all()


def test_multi_date_transformer():
    df = _create_test_data()
    transformer = MultiDateTransformer([date(2014, 10, 4),
                                        date(2016, 5, 30),
                                        ])
    transformer.fit(df['dates'])
    ret = transformer.transform(df['dates'])
    assert (ret == [False, True, False, True, False]).all()


def test_linear_date_transformer():
    df = _create_test_data()
    transformer = LinearDateTransformer()
    transformer.fit(df['dates'])
    ret = transformer.transform(df['dates'])
    assert (ret == [0, 243, 400, 847, 1712]).all()


def test_label_encoder_with_unknown():
    df = _create_test_data()
    transformer = LabelEncoderWithUnknown()
    transformer.fit(df['col_A'])
    ret = transformer.transform(df['col_A'])
    assert (ret == [0, 1, 2, 1, 0]).all()

    ret = transformer.transform(pd.Series([4, 3, 2, 1, 0]))
    assert (ret == [3, 2, 1, 0, 3]).all()


def test_onehot_with_unknown():
    df = _create_test_data()
    encoded_col = LabelEncoderWithUnknown().fit_transform(df['col_A'])

    transformer = OneHotWithUnknown()
    transformer.fit(encoded_col)
    ret = transformer.transform(encoded_col)
    target = pd.DataFrame({"0": [True, False, False, False, True],
                           "1": [False, True, False, True, False],
                           "2": [False, False, True, False, False],
                           "missing": [False, False, False, False, False],
                           })
    assert (ret == target).all().all()

    encoded_col.iloc[0] = 3
    ret = transformer.transform(encoded_col)
    target = pd.DataFrame({"0": [False, False, False, False, True],
                           "1": [False, True, False, True, False],
                           "2": [False, False, True, False, False],
                           "missing": [True, False, False, False, False],
                           })
    assert (ret == target).all().all()


def test_onehot_with_fixed_features():
    df = _create_test_data()
    transformer = OneHotWithFixedFeatures(DAYS_OF_WEEK)
    ds = df['dates'].dt.dayofweek
    transformer.fit(ds)
    ret = transformer.transform(ds)
    target = pd.DataFrame({"Monday": [True, False, False, True, False],
                           "Tuesday": [False, False, True, False, False],
                           "Wednesday": [False, False, False, False, False],
                           "Thursday": [False, False, False, False, False],
                           "Friday": [False, False, False, False, True],
                           "Saturday": [False, True, False, False, False],
                           "Sunday": [False, False, False, False, False],
                           })
    for k in target.keys():
        assert (ret[k] == target[k]).all()


def test_dataframe_selector():
    df = _create_test_data()
    for column in ["col_A", "col_B"]:
        transformer = DataFrameSelector(column)
        transformer.fit(df)
        ds = transformer.transform(df)
        assert (ds == df[column]).all()


def test_series_reshaper():
    df = _create_test_data()
    for column in ["col_A", "col_B"]:
        ds = df[column]
        transformer = SeriesReshaper()
        transformer.fit(ds)
        ret = transformer.transform(ds)
        assert ret.shape == (ds.size, 1)


def test_series_pipeline():
    df = _create_test_data()
    for column in ["col_A", "col_B"]:
        scaling_factor = 2.0
        pipeline = series_pipeline(column, [ScalingTransformer(scaling_factor)])
        pipeline.fit(df)
        ret = pipeline.transform(df)
        assert ret.shape == (df[column].size, 1)
        assert (ret.reshape((ret.shape[0], )) == df[column].values * scaling_factor).all()


def test_dataframe_pipeline():
    df = _create_test_data()
    pipeline = dataframe_pipeline('dates', [DateAttributeTransformer('dayofweek'),
                                            OneHotWithFixedFeatures(DAYS_OF_WEEK),
                                            ])
    pipeline.fit(df)
    ret = pipeline.transform(df)
    target = np.array([[True, False, False, False, False, False, False],
                       [False, False, False, False, False, True, False],
                       [False, True, False, False, False, False, False],
                       [True, False, False, False, False, False, False],
                       [False, False, False, False, True, False, False],
                       ])
    assert (ret == target).all().all()


def test_feature_union():
    df = _create_test_data()
    features = FeatureUnion(transformer_list=[('scale_A', series_pipeline('col_A', [ScalingTransformer(2.0)])),
                                              ('scale_B', series_pipeline('col_B', [ScalingTransformer(-5.0)])),
                                              ('null_A', series_pipeline('col_A', [NullTransformer()])),
                                              ('daysofweek', series_pipeline('dates', [DateAttributeTransformer('dayofweek')])),
                                              ('is_some_day', series_pipeline('dates', [MultiDateTransformer([date(2014, 10, 4),
                                                                                                              date(2016, 5, 30),
                                                                                                              ])])),
                                              ('linear_date', series_pipeline('dates', [LinearDateTransformer()])),
                                              ('label_A', series_pipeline('col_A', [LabelEncoderWithUnknown()])),
                                              ])
    features.fit(df)
    ret = features.transform(df)
    feature_names = features.get_feature_names()
    assert ret.shape[0] == df.shape[0]
    assert ret.shape[1] == len(feature_names)
