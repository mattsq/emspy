import os
import pytest
from mock_connection import MockConnection
from mock_query import MockFltQuery

test_path = os.path.dirname(os.path.realpath(__file__))

FLIGHT_RECORD_GUID = (
    '[-hub-][field]'
    '[[[ems-core][entity-type][foqa-flights]]'
    '[[ems-core][base-field][flight.uid]]]'
)

ENGINE_SERIES_GUID = (
    '[-hub-][field]'
    '[[[ems-core][entity-type][foqa-flights]]'
    '[[airframe-engine-field-set][base-field][engine-series-2]]]'
)


def make_query():
    """Create a MockFltQuery with fieldtree populated."""
    connection = MockConnection(user='', pwd='')
    query = MockFltQuery(connection, 'ems24-app',
                         data_file=os.path.join(test_path, 'mock_metadata.db'))
    query.set_database('FDW Flights')
    query.update_fieldtree(
        'Flight Information',
        exclude_tree=['Processing', 'Date Times', 'FlightPulse']
    )
    query.update_fieldtree(
        'Aircraft Information',
        exclude_tree=['Airframe Information', 'Engine Information',
                      'Fleet Information']
    )
    return query


def test_select_guid_single():
    """select_guid adds a single GUID to the queryset."""
    query = make_query()
    query.select_guid(FLIGHT_RECORD_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 1
    assert qs['select'][0]['fieldId'] == FLIGHT_RECORD_GUID
    assert qs['select'][0]['aggregate'] == 'none'


def test_select_guid_multiple():
    """select_guid accepts multiple GUIDs at once."""
    query = make_query()
    query.select_guid(FLIGHT_RECORD_GUID, ENGINE_SERIES_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 2
    assert qs['select'][0]['fieldId'] == FLIGHT_RECORD_GUID
    assert qs['select'][1]['fieldId'] == ENGINE_SERIES_GUID


def test_select_guid_aggregate():
    """select_guid supports the aggregate keyword."""
    query = make_query()
    query.select_guid(FLIGHT_RECORD_GUID, aggregate='avg')
    qs = query.in_dict()
    assert qs['select'][0]['aggregate'] == 'avg'


def test_select_guid_invalid_aggregate():
    """select_guid exits on invalid aggregate."""
    query = make_query()
    with pytest.raises(SystemExit):
        query.select_guid(FLIGHT_RECORD_GUID, aggregate='invalid')


def test_select_guid_cached():
    """GUIDs already in the fieldtree are resolved from cache, not API."""
    query = make_query()
    # FLIGHT_RECORD_GUID is in fieldtree after update_fieldtree
    query.select_guid(FLIGHT_RECORD_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 1
    assert qs['select'][0]['fieldId'] == FLIGHT_RECORD_GUID


def test_select_guid_api_lookup():
    """GUIDs not in fieldtree are resolved via the API and cached."""
    connection = MockConnection(user='', pwd='')
    query = MockFltQuery(connection, 'ems24-app',
                         data_file=os.path.join(test_path, 'mock_metadata.db'))
    query.set_database('FDW Flights')
    # Don't call update_fieldtree so fieldtree is empty;
    # resolve_guid should fall back to the API
    query.select_guid(FLIGHT_RECORD_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 1
    assert qs['select'][0]['fieldId'] == FLIGHT_RECORD_GUID


def test_select_guid_combines_with_select():
    """select_guid and select can be used together."""
    query = make_query()
    query.select('Flight Record')
    query.select_guid(ENGINE_SERIES_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 2
    assert qs['select'][0]['fieldId'] == FLIGHT_RECORD_GUID
    assert qs['select'][1]['fieldId'] == ENGINE_SERIES_GUID


def test_deselect_after_select_guid_cached():
    """deselect removes a field that was added via select_guid (cached path)."""
    query = make_query()
    query.select_guid(FLIGHT_RECORD_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 1

    query.deselect('Flight Record')
    qs = query.in_dict()
    assert len(qs['select']) == 0


def test_deselect_after_select_guid_api():
    """deselect removes a field added via select_guid's API fallback path."""
    connection = MockConnection(user='', pwd='')
    query = MockFltQuery(connection, 'ems24-app',
                         data_file=os.path.join(test_path, 'mock_metadata.db'))
    query.set_database('FDW Flights')
    # Empty fieldtree forces resolve_guid to use the API fallback,
    # which produces a dict with parent_id=None
    query.select_guid(FLIGHT_RECORD_GUID)
    qs = query.in_dict()
    assert len(qs['select']) == 1

    # Now populate the fieldtree so deselect can find the field by name
    query.update_fieldtree(
        'Flight Information',
        exclude_tree=['Processing', 'Date Times', 'FlightPulse']
    )
    query.deselect('Flight Record')
    qs = query.in_dict()
    assert len(qs['select']) == 0
