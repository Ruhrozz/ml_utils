from typing import List, Dict

import pandas as pd

from sklearn.preprocessing import LabelEncoder


def cat_ohe(data: pd.DataFrame, columns: List[str], prefix="ohe_", drop=False) -> pd.DataFrame:
    ""One hot encoding for categorical features"""
    ohe_data = pd.get_dummies(data[columns], prefix=prefix, prefix_sep="")

    data = pd.concat([data, ohe_data], axis=1)

    if drop:
        data.drop(columns, axis=1, inplace=True)

    return data


def cat_le(data: pd.DataFrame, columns: List[str], prefix="le_", drop=False) -> pd.DataFrame:
    """Label encoding for categorical features"""
    le = LabelEncoder()

    for column in columns:
        data[prefix + column] = le.fit_transform(data[column])

    if drop:
        data.drop(columns, axis=1, inplace=True)

    return data


def cat_freq(data: pd.DataFrame, columns: List[str], prefix="freq_", drop=False) -> pd.DataFrame:
    """Frequency encoding for categorical features"""
    for column in columns:
        fe = data.groupby(column).size() / len(data)
        data[prefix + column] = data[column].map(fe)

    if drop:
        data.drop(columns, axis=1, inplace=True)

    return data


def cat_ord(data: pd.DataFrame, columns: List[str], ordinals: List[Dict[str, int]],  prefix="ord_", drop=False) -> pd.DataFrame:
    """Label encoding for categorical ordinal features"""
    assert len(columns) == len(ordinals)

    for column, ordinal in zip(columns, ordinals):
        data[prefix + column] = data[column].map(ordinal)

    if drop:
        data.drop(columns, axis=1, inplace=True)

    return data
