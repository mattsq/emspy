import os
import pandas as pd
import pytest

from mock_connection import MockConnection
from mock_query import MockFltQuery


test_path = os.path.dirname(os.path.realpath(__file__))


def _make_query():
    connection = MockConnection(user='', pwd='')
    query = MockFltQuery(
        connection,
        'ems24-app',
        data_file=os.path.join(test_path, 'mock_metadata.db'),
    )
    query._FltQuery__queryset = {'select': [], 'format': 'none'}
    return query


def _convert(query, columns, header_names, rows):
    query._FltQuery__columns = columns
    json_output = {
        'header': [{'name': n} for n in header_names],
        'rows': rows,
    }
    return query._FltQuery__to_dataframe(json_output)


def test_to_dataframe_duplicate_column_names_number():
    query = _make_query()
    columns = [
        {'id': 'fid-a', 'type': 'number', 'name': 'Value'},
        {'id': 'fid-b', 'type': 'number', 'name': 'Value'},
    ]
    df = _convert(query, columns, ['Value', 'Value'], [['1', '10'], ['2', '20']])

    assert list(df.columns) == ['Value', 'Value']
    assert df.iloc[:, 0].tolist() == [1, 2]
    assert df.iloc[:, 1].tolist() == [10, 20]
    assert pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    assert pd.api.types.is_numeric_dtype(df.iloc[:, 1])


def test_to_dataframe_duplicate_column_names_mixed_types():
    query = _make_query()
    columns = [
        {'id': 'fid-a', 'type': 'number', 'name': 'Flag'},
        {'id': 'fid-b', 'type': 'boolean', 'name': 'Flag'},
        {'id': 'fid-c', 'type': 'dateTime', 'name': 'Flag'},
    ]
    rows = [
        ['1', 'true', '2024-01-02T03:04:05Z'],
        ['2', 'false', '2024-06-07T08:09:10Z'],
    ]
    df = _convert(query, columns, ['Flag', 'Flag', 'Flag'], rows)

    assert pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    assert pd.api.types.is_bool_dtype(df.iloc[:, 1])
    assert pd.api.types.is_datetime64_any_dtype(df.iloc[:, 2])
    assert df.iloc[:, 0].tolist() == [1, 2]
    assert df.iloc[:, 1].tolist() == [True, True]


def test_to_dataframe_unique_column_names_still_works():
    query = _make_query()
    columns = [
        {'id': 'fid-a', 'type': 'number', 'name': 'A'},
        {'id': 'fid-b', 'type': 'boolean', 'name': 'B'},
    ]
    df = _convert(query, columns, ['A', 'B'], [['7', 'true'], ['8', 'false']])

    assert pd.api.types.is_numeric_dtype(df['A'])
    assert pd.api.types.is_bool_dtype(df['B'])
    assert df['A'].tolist() == [7, 8]
