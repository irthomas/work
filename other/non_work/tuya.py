# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 22:34:25 2022

@author: iant
"""


from datetime import datetime, timedelta
import time
import tinytuya



class tuya(object):
    
    def __init__(self):
        self.connect()
        self.get_devices()
        
    def connect(self):
        self.tt = tinytuya.Cloud(
                apiRegion="eu", 
                apiKey="taagrjpkja5qvmcvsq8v", 
                apiSecret="4690c0d50b33478a9617550dfcef99f6", 
                apiDeviceID="68087008485519506dc4")


    def get_devices(self):
        """get dictionary of devices"""
        devices = self.tt.getdevices()
        self.device_d = {device["name"]:device["id"] for device in devices}
    

    def get_power(self, name):
        """get power in watts"""
        id_ = self.device_d[name]
        status = self.tt.getstatus(id_)
        return status["result"][4]["value"] / 10.0


    def switch_on(self, name):
        id_ = self.device_d[name]
        commands = {
            "commands": [
                {"code": "switch_1", "value": True},
                {"code": "countdown_1", "value": 0},
            ]
        }
        result = self.tt.sendcommand(id_, commands)
        return result

    def switch_off(self, name):
        id_ = self.device_d[name]
        commands = {
            "commands": [
                {"code": "switch_1", "value": False},
                {"code": "countdown_1", "value": 0},
            ]
        }
        result = self.tt.sendcommand(id_, commands)
        return result




tuya = tuya()
print(tuya.device_d)
names = ["Ian Laptop", "Washing machine", "Fridge and freezer"]


dt = 10.0

print("\t\t".join(names))
for loop in range(10):
    now = datetime.now()
    # print(str(now)[:-3])
    powers = [tuya.get_power(name) for name in names]
    print("\t\t\t\t\t\t".join(["%0.1f" %power for power in powers]))
    # print("%0.1fW\t\t%0.1fW\t\t%0.1fW" %(*powers))
    
    next_dt = now + timedelta(seconds = dt)
    
    seconds_remaining = (next_dt - datetime.now()).total_seconds()
    
    time.sleep(seconds_remaining)

# def switch_on()


# # Display Properties of Device
# result = c.getproperties(id)
# print("Properties of device:\n", result)



# # Send Command - Turn on switch
# print("Results\n:", result)