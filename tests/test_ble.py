# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 20:54:05 2025

@author: iant
"""
import nest_asyncio  # noqa
nest_asyncio.apply()  # noqa

from bleak import BleakClient
import asyncio


async def main():
    ble_address = "41425BBE5436"

    async with BleakClient(ble_address) as client:
        # we’ll do the read/write operations here
        print("Connected to BLE device")
        print(client.is_connected)

asyncio.run(main())
