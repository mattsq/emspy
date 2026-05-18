import os
import pytest
import pandas as pd

import emspy
from emspy.query import Fieldset
from emspy.query.fieldset import _parse_fieldset_fields
from mock_connection import MockConnection
from mock_ems import MockEMS
from mock_query import MockFltQuery


# Fixtures -------------------------------------------------------------------

@pytest.fixture
def mocks(monkeypatch):
    monkeypatch.setattr(emspy.query.query, 'EMS', MockEMS)
    monkeypatch.setattr(emspy, 'Connection', MockConnection)


@pytest.fixture
def fieldset(mocks):
    ems_id = 1
    connection = MockConnection(user='', pwd='')
    return Fieldset(connection, ems_id)


@pytest.fixture
def fltquery(mocks, tmp_path):
    # MockFltQuery wires Fieldset into FltQuery via _init_assets and uses the
    # bundled mock_metadata.db for the unused fieldtree.
    test_path = os.path.dirname(os.path.realpath(__file__))
    db_path = os.path.join(test_path, 'mock_metadata.db')
    return MockFltQuery(MockConnection(user='', pwd=''), 'ems-mock',
                         data_file=db_path)


# get_groups -----------------------------------------------------------------

def test_get_groups_returns_dataframe(fieldset):
    df = fieldset.get_groups()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['id', 'name', 'description']
    assert 'mock-group-id' in df['id'].tolist()
    assert 'Mock Group' in df['name'].tolist()


def test_get_groups_uses_cache_on_second_call(fieldset, monkeypatch):
    fieldset.get_groups()  # priming call
    call_count = {'n': 0}
    real_request = fieldset._conn.request

    def counting_request(*args, **kwargs):
        call_count['n'] += 1
        return real_request(*args, **kwargs)

    monkeypatch.setattr(fieldset._conn, 'request', counting_request)
    fieldset.get_groups()
    assert call_count['n'] == 0


def test_get_groups_refresh_bypasses_cache(fieldset, monkeypatch):
    fieldset.get_groups()  # priming call
    call_count = {'n': 0}
    real_request = fieldset._conn.request

    def counting_request(*args, **kwargs):
        call_count['n'] += 1
        return real_request(*args, **kwargs)

    monkeypatch.setattr(fieldset._conn, 'request', counting_request)
    fieldset.get_groups(refresh=True)
    assert call_count['n'] == 1


# get_group ------------------------------------------------------------------

def test_get_group_splits_subgroups_and_fieldsets(fieldset):
    df = fieldset.get_group('mock-group-id')
    assert list(df.columns) == ['type', 'id', 'name', 'description']
    groups = df[df['type'] == 'group']
    fieldsets = df[df['type'] == 'fieldset']
    assert len(groups) == 1
    assert groups.iloc[0]['name'] == 'Sub Group'
    assert len(fieldsets) == 2
    assert set(fieldsets['name'].tolist()) == {'Mock Fieldset', 'Alt Shape Fieldset'}


def test_get_group_empty(fieldset):
    df = fieldset.get_group('empty-group-id')
    assert len(df) == 0
    assert list(df.columns) == ['type', 'id', 'name', 'description']


def test_get_group_validates_group_id(fieldset):
    with pytest.raises(ValueError):
        fieldset.get_group('')
    with pytest.raises(ValueError):
        fieldset.get_group(None)


# get_fieldset ---------------------------------------------------------------

def test_get_fieldset_schemaitems_shape(fieldset):
    fs = fieldset.get_fieldset('mock-group-id', 'Mock Fieldset')
    assert fs['name'] == 'Mock Fieldset'
    assert fs['group_id'] == 'mock-group-id'
    assert fs['ems_id'] == 1
    df = fs['fields']
    assert list(df.columns) == ['id', 'name', 'type']
    assert df.iloc[0]['id'] == 'field-id-1'
    assert df.iloc[0]['name'] == 'Takeoff Airport Code'
    assert df.iloc[0]['type'] == 'string'
    assert len(df) == 3


def test_get_fieldset_alt_fields_shape(fieldset):
    # Older response shape using 'fields' + id/name/dataType
    fs = fieldset.get_fieldset('mock-group-id', 'Alt Shape Fieldset')
    df = fs['fields']
    assert df.iloc[0]['id'] == 'alt-field-1'
    assert df.iloc[0]['name'] == 'Alt One'
    assert df.iloc[0]['type'] == 'number'
    assert len(df) == 2


def test_get_fieldset_cache(fieldset, monkeypatch):
    fieldset.get_fieldset('mock-group-id', 'Mock Fieldset')
    call_count = {'n': 0}
    real_request = fieldset._conn.request

    def counting_request(*args, **kwargs):
        call_count['n'] += 1
        return real_request(*args, **kwargs)

    monkeypatch.setattr(fieldset._conn, 'request', counting_request)
    fieldset.get_fieldset('mock-group-id', 'Mock Fieldset')
    assert call_count['n'] == 0


# find -----------------------------------------------------------------------

def test_find_unambiguous_match(fieldset):
    hits = fieldset.find('Mock Fieldset')
    assert len(hits) == 1
    assert hits[0]['group_id'] == 'mock-group-id'
    assert hits[0]['path'] == ['Mock Group']


def test_find_walks_into_subgroups(fieldset):
    hits = fieldset.find('Deep Fieldset')
    assert len(hits) == 1
    assert hits[0]['group_id'] == 'sub-group-id'
    assert hits[0]['path'] == ['Mock Group', 'Sub Group']


def test_find_returns_multiple_for_ambiguous_name(fieldset):
    hits = fieldset.find('Dup Fieldset')
    assert len(hits) == 2
    group_ids = sorted(h['group_id'] for h in hits)
    assert group_ids == ['dup-group-a', 'dup-group-b']


def test_find_skips_permission_gated_subgroups(fieldset, capsys):
    # 'Forbidden Group' raises HTTPError on get_group; the walk should
    # swallow it, print a friendly note, and continue.
    hits = fieldset.find('Mock Fieldset')
    assert len(hits) == 1
    captured = capsys.readouterr()
    assert 'Skipping fieldset group' in captured.out
    assert 'Forbidden Group' in captured.out
    assert 'HTTP 403' in captured.out
    assert 'Continuing search' in captured.out


def test_find_returns_empty_when_not_found(fieldset):
    assert fieldset.find('Nonexistent Fieldset') == []


# clear_cache ----------------------------------------------------------------

def test_clear_cache_empties_all_three(fieldset):
    fieldset.get_groups()
    fieldset.get_group('mock-group-id')
    fieldset.get_fieldset('mock-group-id', 'Mock Fieldset')
    assert fieldset._root_cache is not None
    assert 'mock-group-id' in fieldset._group_cache
    assert ('mock-group-id', 'Mock Fieldset') in fieldset._fieldset_cache

    fieldset.clear_cache()
    assert fieldset._root_cache is None
    assert fieldset._group_cache == {}
    assert fieldset._fieldset_cache == {}


# Parser-level robustness ----------------------------------------------------

def test_parse_fields_handles_bare_strings():
    df = _parse_fieldset_fields(['just-an-id', 'another-id'])
    assert list(df.columns) == ['id', 'name', 'type']
    assert df.iloc[0]['id'] == 'just-an-id'
    assert df.iloc[0]['name'] is None


def test_parse_fields_handles_snake_case():
    df = _parse_fieldset_fields([{'moniker': 'x', 'description': 'X',
                                   'data_type': 'number'}])
    assert df.iloc[0]['type'] == 'number'


def test_parse_fields_empty():
    df = _parse_fieldset_fields([])
    assert list(df.columns) == ['id', 'name', 'type']
    assert len(df) == 0


# FltQuery.select_fieldset integration --------------------------------------

def test_select_fieldset_with_explicit_group(fltquery):
    fltquery.select_fieldset('Mock Fieldset', group='mock-group-id')
    queryset = fltquery._FltQuery__queryset
    columns = fltquery._FltQuery__columns

    assert len(queryset['select']) == 3
    assert queryset['select'][0] == {
        'fieldId': 'field-id-1',
        'aggregate': 'none',
    }
    assert len(columns) == 3
    assert columns[0]['id'] == 'field-id-1'
    assert columns[0]['name'] == 'Takeoff Airport Code'
    assert columns[0]['type'] == 'string'


def test_select_fieldset_by_name_walks_tree(fltquery):
    fltquery.select_fieldset('Mock Fieldset')
    queryset = fltquery._FltQuery__queryset
    assert len(queryset['select']) == 3
    assert {e['fieldId'] for e in queryset['select']} == {
        'field-id-1', 'field-id-2', 'field-id-3'
    }


def test_select_fieldset_ambiguous_name_raises(fltquery):
    with pytest.raises(ValueError) as exc:
        fltquery.select_fieldset('Dup Fieldset')
    msg = str(exc.value)
    assert 'ambiguous' in msg
    assert 'Dup Group A' in msg
    assert 'Dup Group B' in msg


def test_select_fieldset_not_found_raises(fltquery):
    with pytest.raises(ValueError) as exc:
        fltquery.select_fieldset('Nonexistent Fieldset')
    assert 'not found' in str(exc.value)


def test_select_fieldset_with_prefetched_dict(fltquery):
    # Pre-fetch via Fieldset, then pass the dict to select_fieldset.
    fs = Fieldset(fltquery._conn, fltquery._ems_id).get_fieldset(
        'mock-group-id', 'Alt Shape Fieldset'
    )
    fltquery.select_fieldset(fs)
    queryset = fltquery._FltQuery__queryset
    assert len(queryset['select']) == 2
    assert queryset['select'][0]['fieldId'] == 'alt-field-1'


def test_select_fieldset_with_aggregate(fltquery):
    fltquery.select_fieldset('Mock Fieldset', group='mock-group-id',
                              aggregate='avg')
    for entry in fltquery._FltQuery__queryset['select']:
        assert entry['aggregate'] == 'avg'


def test_select_fieldset_stacks_with_regular_select(fltquery):
    # Verifies fieldset entries coexist with select() entries in one queryset.
    # Stub a fake select() entry directly rather than going through fieldtree
    # discovery, which is unrelated to what we're testing.
    fltquery._FltQuery__queryset['select'].append({
        'fieldId': 'pre-existing-field',
        'aggregate': 'none',
    })
    fltquery._FltQuery__columns.append({
        'id': 'pre-existing-field', 'name': 'Pre-existing', 'type': 'string'
    })
    pre_len = len(fltquery._FltQuery__queryset['select'])
    fltquery.select_fieldset('Mock Fieldset', group='mock-group-id')
    post_len = len(fltquery._FltQuery__queryset['select'])
    assert post_len == pre_len + 3
    # Both kinds of entries are flat {fieldId, aggregate} dicts.
    assert fltquery._FltQuery__queryset['select'][0]['fieldId'] == 'pre-existing-field'


def test_select_fieldset_rejects_bad_type(fltquery):
    with pytest.raises(TypeError):
        fltquery.select_fieldset(12345)


def test_select_fieldset_rejects_unknown_aggregate(fltquery):
    with pytest.raises(SystemExit):
        fltquery.select_fieldset('Mock Fieldset', group='mock-group-id',
                                  aggregate='median')


def test_select_fieldset_empty_fieldset_is_noop(fltquery):
    fltquery.select_fieldset('Empty Fieldset', group='mock-group-id')
    assert len(fltquery._FltQuery__queryset['select']) == 0
