# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:26:53 2022

@author: iant
"""


import requests
import time
import json
from datetime import datetime, timezone
username = "margherita.coccia@gmail.com"
password = "17AEdith"
secret = "wZaRN7rpjn3FoNyF5IFuxg9uMzYJcvOoQ8QWiIqS3hfk6gLhVlG57j5YNoZL2Rtc"


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

class Tado:
  json_content        = { 'Content-Type': 'application/json'}
  api                 = 'https://my.tado.com/api/v2'
  api_acme            = 'https://acme.tado.com/v1'
  api_minder          = 'https://minder.tado.com/v1'
  timeout        = 15

  def __init__(self, username, password, secret):
    self.username = username
    self.password = password
    self.secret = secret
    self._login()
    self.id = self.get_me()['homes'][0]['id']

  def _login(self):
    """Login and setup the HTTP session."""
    url='https://auth.tado.com/oauth/token'
    data = { 'client_id'     : 'tado-web-app',
             'client_secret' : self.secret,
             'grant_type'    : 'password',
             'password'      : self.password,
             'scope'         : 'home.user',
             'username'      : self.username }
    request = requests.post(url, data=data, timeout=self.timeout)
    request.raise_for_status()
    response = request.json()
    self.access_token = response['access_token']
    self.token_expiry = time.time() + float(response['expires_in'])
    self.refresh_token = response['refresh_token']
    self.access_headers = {'Authorization': 'Bearer ' + response['access_token']}

  def _api_call(self, cmd, data=False, method='GET'):
    """Perform an API call."""
    def call_delete(url):
      r = requests.delete(url, headers=self.access_headers, timeout=self.timeout)
      r.raise_for_status()
      return r
    def call_put(url, data):
      r = requests.put(url, headers={**self.access_headers, **self.json_content}, data=json.dumps(data), timeout=self.timeout)
      r.raise_for_status()
      return r
    def call_get(url):
      r = requests.get(url, headers=self.access_headers, timeout=self.timeout)
      r.raise_for_status()
      return r

    self.refresh_auth()
    url = '%s/%s' % (self.api, cmd)
    if method == 'DELETE':
      return call_delete(url)
    elif method == 'PUT' and data:
      return call_put(url, data).json()
    elif method == 'GET':
      return call_get(url).json()

  def _api_acme_call(self, cmd, data=False, method='GET'):
    """Perform an API call."""
    def call_delete(url):
      r = requests.delete(url, headers=self.access_headers, timeout=self.timeout)
      r.raise_for_status()
      return r
    def call_put(url, data):
      r = requests.put(url, headers={**self.access_headers, **self.json_content}, data=json.dumps(data), timeout=self.timeout)
      r.raise_for_status()
      return r
    def call_get(url):
      r = requests.get(url, headers=self.access_headers, timeout=self.timeout)
      r.raise_for_status()
      return r

    self.refresh_auth()
    url = '%s/%s' % (self.api_acme, cmd)
    if method == 'DELETE':
      return call_delete(url)
    elif method == 'PUT' and data:
      return call_put(url, data).json()
    elif method == 'GET':
      return call_get(url).json()



  def refresh_auth(self):
    """Refresh the access token."""
    if time.time() < self.token_expiry - 30:
      return
    url='https://auth.tado.com/oauth/token'
    data = { 'client_id'     : 'tado-web-app',
             'client_secret' : self.secret,
             'grant_type'    : 'refresh_token',
             'refresh_token' : self.refresh_token,
             'scope'         : 'home.user'
           }
    try:
      request = requests.post(url, data=data, timeout=self.timeout)
      request.raise_for_status()
    except:
      self._login()
      return
    response = request.json()
    self.access_token = response['access_token']
    self.token_expiry = time.time() + float(response['expires_in'])
    self.refresh_token = response['refresh_token']
    self.access_headers['Authorization'] = 'Bearer ' + self.access_token




  def get_capabilities(self, zone):
    data = self._api_call('homes/%i/zones/%i/capabilities' % (self.id, zone))
    return data

  def get_devices(self):
    data = self._api_call('homes/%i/devices' % self.id)
    return data

  def get_device_usage(self):
    data = self._api_call('homes/%i/deviceList' % self.id)
    return data


  def get_home(self):
    data = self._api_call('homes/%i' % self.id)
    return data




  def get_me(self):
    data = self._api_call('me')
    return data

  def get_mobile_devices(self):
    data = self._api_call('homes/%i/mobileDevices' % self.id)
    return data

  def get_schedule(self, zone):
    data = self._api_call('homes/%i/zones/%i/schedule/activeTimetable' % (self.id, zone))
    return data

  def get_schedule_blocks(self, zone, schedule):
    return self._api_call('homes/%i/zones/%i/schedule/timetables/%i/blocks' % (self.id, zone, schedule))



  def get_state(self, zone):
    data = self._api_call('homes/%i/zones/%i/state' % (self.id, zone))
    return data

  def get_measuring_device(self, zone):
    """
    Gets the active measuring device of a zone
    Args:
      zone (int): The zone ID.
    Returns:
      dict: A dictionary with the current measuring informations.
    """

    data = self._api_call('homes/%i/zones/%i/measuringDevice' % (self.id, zone))
    return data

  def get_users(self):
    """Get all users of your home."""
    data = self._api_call('homes/%i/users' % self.id)
    return data

  def get_weather(self):
    data = self._api_call('homes/%i/weather' % self.id)
    return data

  def get_zones(self):
    data = self._api_call('homes/%i/zones' % self.id)
    return data




  def get_report(self, zone, date):
    data = self._api_call('homes/%i/zones/%i/dayReport?date=%s' % (self.id, zone, date))
    return data


  def get_temperature_offset(self, device_serial):
    data = self._api_call('devices/%s/temperatureOffset' % device_serial)
    return data

  def get_air_comfort(self):
    data = self._api_call('homes/%i/airComfort' % self.id)
    return data

  def get_air_comfort_geoloc(self, latitude, longitude):
    data = self._api_acme_call('homes/%i/airComfort?latitude=%f&longitude=%f' % (self.id, latitude, longitude))
    return data



  def get_zone_states(self):
    data = self._api_call('homes/%i/zoneStates' % (self.id))
    return data

  

t = Tado(username, password, secret)
# print(t.get_me())

zone = 1
latitude = 50.8096237
longitude = 4.4042363
schedule = 1 #3 day mon-fri, sat, sun
dt_str = "2022-11-19"
device_serial = 'IB3230678528'

t.get_capabilities(zone)
t.get_devices()
t.get_device_usage()
t.get_home()
t.get_me()
t.get_mobile_devices()
t.get_schedule(zone) #1 i.e. 3 day mon-fri, sat, sun
t.get_schedule_blocks(zone, schedule) #detailed schedule
t.get_state(zone) #sensorDataPoints insideTemperature humidity
t.get_measuring_device(zone)
t.get_users()
t.get_weather() #solarIntensity, outsideTemperature
t.get_zones() #1 = chloe, 2 = anna
# t.get_away_configuration(zone)
t.get_report(zone, dt_str)
# t.get_temperature_offset(device_serial) # 0.0
# t.get_air_comfort() #cold, comfy, last open window
t.get_air_comfort_geoloc(latitude, longitude) #pollen, pollutants
# t.get_zone_states() #all data for zones


def get_zone_dict(t):
    zones_d = t.get_zones()
    zones = {d["name"].upper():d["id"] for d in zones_d}
    return zones


def get_sensor_data(t, zone):
    zone_state = t.get_state(zone)
    dt_str = zone_state["sensorDataPoints"]["humidity"]["timestamp"]
    temperature = zone_state["sensorDataPoints"]["insideTemperature"]["celsius"]
    humidity = zone_state["sensorDataPoints"]["humidity"]["percentage"]
    return dt_str, temperature, humidity


def get_sensors_data(t, zones):
    
    for i, (zone_name, zone_id) in enumerate(zones.items()):
        dt_str, temperature, humidity = get_sensor_data(t, zone_id)
        
        if i == 0:
            dt = datetime.strptime(dt_str, DATETIME_FORMAT)
            out = [dt, zone_name, temperature, humidity]
        else:
            out.extend([zone_name, temperature, humidity])
    return out



def weather(t):
    weather_d = t.get_weather()
    dt_str = weather_d["solarIntensity"]["timestamp"]
    solar = weather_d["solarIntensity"]["percentage"]
    temperature = weather_d["outsitdeTemperature"]["celsius"]
    weather = weather_d["weatherState"]["value"]
    return dt_str, temperature, weather, solar



zones = get_zone_dict(t)
out = get_sensors_data(t, zones)