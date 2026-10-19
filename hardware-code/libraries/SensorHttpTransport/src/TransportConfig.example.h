#pragma once

// Use the Jetson's LAN IPv4 address (not 0.0.0.0, a URL, or localhost).
#define SENSOR_WIFI_SSID "YOUR_WIFI_SSID"
#define SENSOR_WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define SENSOR_JETSON_HOST "192.168.1.100"

// Optional overrides, in milliseconds:
// #define SENSOR_HTTP_CONNECT_TIMEOUT_MS 2000
// #define SENSOR_HTTP_TIMEOUT_MS 3000
