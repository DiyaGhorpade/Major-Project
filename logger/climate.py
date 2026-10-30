"""Bengaluru-inspired simulation, not live weather or calibrated sensor data.

Monthly temperature anchors: IMD Bengaluru 1981–2010 climatological table:
https://city.imd.gov.in/citywx/extreme/OCT/bengaluru2.htm
Humidity and shaded light profiles are illustrative growing-site assumptions.
Light retains the linked Wokwi sketch's 0–1000 scale, not outdoor lux.
"""

import math
import random
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
TEMPERATURE_NORMALS = (
    (15.8, 27.9), (17.5, 30.7), (20.0, 33.1), (22.0, 34.0),
    (21.7, 33.3), (20.4, 29.6), (19.9, 28.3), (19.8, 27.8),
    (19.8, 28.6), (19.6, 28.2), (18.0, 27.2), (16.2, 26.5),
)


class Climate:
    def __init__(self):
        self.cloud = 0.25
        self.temperature_noise = 0.0
        self.humidity_noise = 0.0

    def sample(self, now=None, elapsed=2):
        now = (now or datetime.now(IST)).astimezone(IST)
        hour = now.hour + now.minute / 60 + now.second / 3600
        low, high = TEMPERATURE_NORMALS[now.month - 1]
        # Minimum near 03:00 and maximum near 15:00; daylight 06:00–18:00.
        warmth = (1 + math.cos(2 * math.pi * (hour - 15) / 24)) / 2
        daylight = max(0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        step = min(max(elapsed, 0), 60) / 60
        self.cloud = max(0, min(0.85, self.cloud + random.uniform(-0.015, 0.015) * step))
        self.temperature_noise = max(-0.5, min(0.5, self.temperature_noise + random.uniform(-0.05, 0.05) * step))
        self.humidity_noise = max(-2, min(2, self.humidity_noise + random.uniform(-0.2, 0.2) * step))
        temperature = low + (high - low) * warmth - self.cloud * daylight + self.temperature_noise
        humid_season = 6 <= now.month <= 11
        humidity = (87 if humid_season else 75) - (25 if humid_season else 35) * warmth + self.cloud * 5 + self.humidity_noise
        light = 900 * daylight * (1 - 0.65 * self.cloud)
        return round(temperature, 2), round(max(25, min(95, humidity)), 2), round(light, 2)
