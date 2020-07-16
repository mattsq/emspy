from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library

try:
    from urllib.error import HTTPError
except:
    from urllib2 import HTTPError

standard_library.install_aliases()
import pprint as pp
from builtins import map
from builtins import object
import json, urllib.request, urllib.parse, urllib.error, urllib.request, urllib.error, urllib.parse, ssl, sys, io, gzip
from numbers import Number


class MockConnection(object):
    """
    Object for connection to EMS API
    """

    def __init__(self, user=None, pwd=None, proxies=None, verbose=False, ignore_ssl_errors=False, server="prod",
                 server_url=None):

        self.__user = user
        self.__pwd = pwd
        self.__proxies = proxies
        self.__ntrials = 0
        self.__uri_root = None
        self.__ignore_ssl_errors = ignore_ssl_errors
        self.token = None
        self.token_type = None

        # We assign the uri root to a member variable up front, and use that everywhere to
        # simplify. In order to use an alternate uri root, it must be specified in the constructor.
        if server_url is not None:
            self.__uri_root = server_url
        else:
            self.__uri_root = 'https://ems.efoqa.com/api'

    def request(self,
                rtype="GET", uri=None, uri_keys=None, uri_args=None,
                headers=None, body=None, data=None, jsondata=None, proxies=None,
                verbose=False
                ):
        resp_h, content = [], []
        if uri_keys == ('profile', 'search'):
            if body['search'] == 'A PROFILE THAT SHOULD NEVER EXIST':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', '10a4992b-5d70-4b0a-b15a-d89c71dec26a'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Wed, 14 Aug 2019 18:37:31 GMT'), ('Connection', 'close'), ('Content-Length', '122')]
                content = []
            else:
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', 'a2f0a219-f89a-467c-a087-d0acb8a0ef41'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Tue, 13 Aug 2019 16:34:18 GMT'), ('Connection', 'close'), ('Content-Length', '590')]
                content = [{'localId': 88, 'id': '5b8dc8cb-c8cb-c8cb-c8cb-c8cb39e6c8cb', 'name': 'Duplicate Profile', 'treeLocation': [{'id': 'ffffffff-0000-0000-0000-000000000000', 'name': 'APM Profiles'}, {'id': '1918e33f-a7f5-469a-8c2d-310b506ab797', 'name': 'Standard Library Profiles'}, {'id': 'e3ae6cb3-8200-487d-9ebd-8fd6cd68b6b2', 'name': 'Efficiency'}, {'id': 'a1d76ee9-ec62-4cd0-bb53-9cc92228f13b', 'name': 'Block-Cost Model'}], 'library': True, 'currentVersion': 9}, {'localId': 108, 'id': '12e212e2-12e2-12e2-12e2-ad12e2a14acc', 'name': 'Single Real Profile 2', 'treeLocation': [{'id': 'ffffffff-0000-0000-0000-000000000000', 'name': 'APM Profiles'}, {'id': '1918e33f-a7f5-469a-8c2d-310b506ab797', 'name': 'Standard Library Profiles'}, {'id': 'e3ae6cb3-8200-487d-9ebd-8fd6cd68b6b2', 'name': 'Efficiency'}, {'id': 'a1d76ee9-ec62-4cd0-bb53-9cc92228f13b', 'name': 'Block-Cost Model'}], 'library': True, 'currentVersion': 9}, {'localId': 56, 'id': 'f163eeee-63ee-63ee-63ee-1b363eed63ee', 'name': 'Single Profile 3', 'treeLocation': [{'id': 'ffffffff-0000-0000-0000-000000000000', 'name': 'APM Profiles'}, {'id': '1918e33f-a7f5-469a-8c2d-310b506ab797', 'name': 'Standard Library Profiles'}, {'id': 'e3ae6cb3-8200-487d-9ebd-8fd6cd68b6b2', 'name': 'Efficiency'}, {'id': 'a1d76ee9-ec62-4cd0-bb53-9cc92228f13b', 'name': 'Block-Cost Model'}], 'library': True, 'currentVersion': 8}, {'localId': 99, 'id': 'c152c3dd-496a-496a-496a-bd5fc3ddbd5f', 'name': 'Single Real Profile', 'treeLocation': [{'id': 'ffffffff-0000-0000-0000-000000000000', 'name': 'APM Profiles'}, {'id': '1918e33f-a7f5-469a-8c2d-310b506ab797', 'name': 'Standard Library Profiles'}, {'id': 'e3ae6cb3-8200-487d-9ebd-8fd6cd68b6b2', 'name': 'Efficiency'}, {'id': 'a1d76ee9-ec62-4cd0-bb53-9cc92228f13b', 'name': 'Block-Cost Model'}], 'library': True, 'currentVersion': 8}, {'localId': 500000, 'id': '5b8d6db7-c8cb-474f-b951-4c8e39e6eba1', 'name': 'Duplicate Profile', 'treeLocation': [{'id': 'ffffffff-0000-0000-0000-000000000000', 'name': 'APM Profiles'}, {'id': '1918e33f-a7f5-469a-8c2d-310b506ab797', 'name': 'Standard Library Profiles'}, {'id': 'e3ae6cb3-8200-487d-9ebd-8fd6cd68b6b2', 'name': 'Efficiency'}, {'id': 'a1d76ee9-ec62-4cd0-bb53-9cc92228f13b', 'name': 'Block-Cost Model'}], 'library': True, 'currentVersion': 10}]
            return resp_h, content
        elif uri_keys == ('analytic', 'search') and jsondata is not None:
            print(jsondata['id'])
            if jsondata['id'] == 'fake-bar-alt-id-that-exists=':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', 'c3bf17f2-a1d4-454b-bc1f-ea669cbe5842'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Wed, 25 Sep 2019 19:28:19 GMT'), ('Connection', 'close'), ('Content-Length', '1022')]
                content = {'id': 'fake-bar-alt-id-that-exists=',
                           'name': 'Baro-Corrected Altitude (ft)',
                           'description': 'The altitude above mean sea level (MSL) obtained by applying corrections to the pressure altitude using the altimeter setting.  This parameter may have discrete jumps during these corrections.',
                           'units': 'ft'}
                return resp_h, content
            elif jsondata['id'] == 'fake-pressure-alt-id-that-exists=':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', 'c3bf17f2-a1d4-454b-bc1f-ea669cbe5842'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Wed, 25 Sep 2019 19:28:19 GMT'), ('Connection', 'close'), ('Content-Length', '1022')]
                content = {'id': 'fake-pressure-alt-id-that-exists=',
                           'name': 'Pressure Altitude (ft)',
                           'description': 'Altitude derived from static air pressure.  A value of zero should always correspond to static air pressure = 29.92 inches of mercury.',
                           'units': 'ft'}
                return resp_h, content
            elif jsondata['id'] == 'fake-pressure-alt-id-that-DOES-NOT-exist=':
                raise HTTPError('url', 'code', 'msg', 'hdrs', None)
        elif uri_keys == ('database', 'field_group'):
            if body is None:
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', '8473bd96-3140-4866-86ba-9a1f22042de2'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Thu, 09 Jul 2020 15:53:40 GMT'), ('Connection', 'close'), ('Content-Length', '524')]
                content = {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[--][internal-field-group][root]]]', 'name': '<root>', 'groups': [{'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[[ems-core][type-link][downloads * flights]]][[ems-core][internal-field-group][download-info]]]', 'name': 'Download Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[[ems-core][type-link][downloads * flights]]][[ems-aux][internal-field-group][download-review]]]', 'name': 'Download Review'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][flight-info]]]', 'name': 'Flight Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][aircraft-info]]]', 'name': 'Aircraft Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-aux][internal-field-group][flight-review]]]', 'name': 'Flight Review'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][data-info]]]', 'name': 'Data Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][nav-info]]]', 'name': 'Navigation Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-dal-fdwflightconstantsfromodw][internal-field-group][f0000000-f000-f000-f000-f00000000000]]]', 'name': 'Operational Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[weather][internal-field-group][wx-info]]]', 'name': 'Weather Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-apm][flight-group][profile-container:folder-ffffffff000000000000000000000000]]]', 'name': 'Profiles'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[flight-identification][internal-field-group][external-flight-info]]]', 'name': 'External Flight Information'}], 'fields': []}
            elif body['groupId'] == '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][aircraft-info]]]':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', 'ac03b088-cd29-4dac-b26e-8371e7dfaf3c'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Thu, 09 Jul 2020 16:41:05 GMT'), ('Connection', 'close'), ('Content-Length', '751')]
                content = {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][aircraft-info]]]', 'name': 'Aircraft Information', 'groups': [{'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[airframe-information][internal-field-group][airframe-information]]]', 'name': 'Airframe Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[esn-information][internal-field-group][engine-information]]]', 'name': 'Engine Information'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[[ems-aux][type-link][flights * fleets]]][[ems-aux][internal-field-group][fleet-info]]]', 'name': 'Fleet Information'}], 'fields': [{'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.fleet]]]', 'type': 'discrete', 'name': 'Fleet'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.aircraft]]]', 'type': 'discrete', 'name': 'Tail Number'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.aircraft-version]]]', 'type': 'discrete', 'name': 'Nose Number'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.operator]]]', 'type': 'discrete', 'name': 'Operator'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.client-data-0]]]', 'type': 'discrete', 'name': 'Flight Classification'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.client-data-1]]]', 'type': 'discrete', 'name': 'Fleet Group'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.client-data-2]]]', 'type': 'discrete', 'name': 'Customer'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.client-data-3]]]', 'type': 'discrete', 'name': 'File Type'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[[ems-aux][type-link][fdw-flights * aircraft]]][[ems-aux][base-field][fdw-aircraft.active-state]]]', 'type': 'discrete', 'name': 'Aircraft Active State'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][aircraft-icao-code]]]', 'type': 'discrete', 'name': 'Aircraft ICAO Code'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe]]]', 'type': 'discrete', 'name': 'Airframe'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe-engine-series]]]', 'type': 'discrete', 'name': 'Airframe Engine Series'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe-engine]]]', 'type': 'discrete', 'name': 'Airframe Engine Type'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe-engine-winglets]]]', 'type': 'discrete', 'name': 'Airframe Engine Winglets'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][fleet-group]]]', 'type': 'discrete', 'name': 'Airframe Group'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe-manufacturer]]]', 'type': 'discrete', 'name': 'Airframe Manufacturer'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][airframe-winglets]]]', 'type': 'discrete', 'name': 'Airframe Winglets'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][cabin-class]]]', 'type': 'discrete', 'name': 'Cabin Class'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][engine-manufacturer]]]', 'type': 'discrete', 'name': 'Engine Manufacturer'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][engine-series-2]]]', 'type': 'discrete', 'name': 'Engine Series'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][engine-series]]]', 'type': 'discrete', 'name': 'Engine Type'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[airframe-engine-field-set][base-field][winglets]]]', 'type': 'discrete', 'name': 'Winglets'}]}
            elif body['groupId'] == '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][flight-info]]]':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', 'e850cba4-e99a-4049-b95b-26f81ad3d279'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Thu, 09 Jul 2020 16:45:11 GMT'), ('Connection', 'close'), ('Content-Length', '544')]
                content = {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][flight-info]]]', 'name': 'Flight Information', 'groups': [{'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[ems-core][internal-field-group][foqa-flights-processing]]]', 'name': 'Processing'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[fdw-date-times][internal-field-group][date-times]]]', 'name': 'Date Times'}, {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[source-flight-info][internal-field-group][source-system-information]]]', 'name': 'Source System Information'}], 'fields': [{'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.uid]]]', 'type': 'number', 'name': 'Flight Record'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.exact-date]]]', 'type': 'dateTime', 'name': 'Flight Date (Exact)'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.masked-date]]]', 'type': 'dateTime', 'name': 'Flight Date (Month/Year)'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.date-confidence]]]', 'type': 'discrete', 'name': 'Flight Date Confidence'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.flight-num-int]]]', 'type': 'number', 'name': 'Flight Number'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.flight-num-str]]]', 'type': 'string', 'name': 'Flight Number String'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.exist-takeoff]]]', 'type': 'boolean', 'name': 'Takeoff Valid'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.exist-landing]]]', 'type': 'boolean', 'name': 'Landing Valid'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.exist-takeoff-or-landing]]]', 'type': 'boolean', 'name': 'Takeoff or Landing Valid'}]}
            elif body['groupId'] == '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[source-flight-info][internal-field-group][source-system-information]]]':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', '64272b3a-f862-4ade-8252-229749bf7c8b'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Thu, 09 Jul 2020 16:51:20 GMT'), ('Connection', 'close'), ('Content-Length', '450')]
                content = {'id': '[-hub-][field-group][[[ems-core][entity-type][foqa-flights]][[source-flight-info][internal-field-group][source-system-information]]]', 'name': 'Source System Information', 'groups': [], 'fields': [{'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][from-source-ems]]]', 'type': 'boolean', 'name': 'From Source EMS'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][source-ems-installation]]]', 'type': 'discrete', 'name': 'Source EMS Installation'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][source-flight-record]]]', 'type': 'number', 'name': 'Source Flight Record'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][source-tail-number]]]', 'type': 'string', 'name': 'Source Tail Number'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][source-fleet-id]]]', 'type': 'number', 'name': 'Source Fleet ID'}, {'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[source-flight-info][base-field][source-download-record-creation-date]]]', 'type': 'dateTime', 'name': 'Source Download Record Creation Date'}]}
            return resp_h, content
        elif uri_keys == ('database', 'field'):
            if uri_args[2] == '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.date-confidence]]]':
                resp_h = [('Cache-Control', 'no-store, must-revalidate, no-cache'), ('Pragma', 'no-cache'), ('Content-Type', 'application/json; charset=utf-8'), ('Content-Encoding', 'gzip'), ('Vary', 'Accept-Encoding'), ('Server', 'Microsoft-IIS/8.5'), ('X-Adi-Unique-Id', '8679594c-1e56-4ae8-8793-610284f07323'), ('X-Powered-By', 'ARR/3.0'), ('Date', 'Thu, 16 Jul 2020 15:22:40 GMT'), ('Connection', 'close'), ('Content-Length', '279')]
                content = {'type': 'discrete', 'discreteValues': {'1': 'Low', '0': 'Unknown', '2': 'High'}, 'name': 'Flight Date Confidence', 'id': '[-hub-][field][[[ems-core][entity-type][foqa-flights]][[ems-core][base-field][flight.date-confidence]]]'}
            return resp_h, content


def print_resp(resp):
    for r in resp:
        pp.pprint(r)
