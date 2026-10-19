#pragma once

// Copy TransportConfig.example.h to TransportConfig.local.h in this directory.
// The local file is ignored by Git. Build flags may also define these values.
#if __has_include("TransportConfig.local.h")
#include "TransportConfig.local.h"
#endif

#ifndef SENSOR_WIFI_SSID
#define SENSOR_WIFI_SSID ""
#endif
#ifndef SENSOR_WIFI_PASSWORD
#define SENSOR_WIFI_PASSWORD ""
#endif
#ifndef SENSOR_JETSON_HOST
#define SENSOR_JETSON_HOST ""
#endif
#ifndef SENSOR_HTTP_CONNECT_TIMEOUT_MS
#define SENSOR_HTTP_CONNECT_TIMEOUT_MS 2000
#endif
#ifndef SENSOR_HTTP_TIMEOUT_MS
#define SENSOR_HTTP_TIMEOUT_MS 3000
#endif

static_assert(SENSOR_HTTP_CONNECT_TIMEOUT_MS > 0 &&
              SENSOR_HTTP_CONNECT_TIMEOUT_MS <= 60000,
              "HTTP connect timeout must be 1..60000 ms");
static_assert(SENSOR_HTTP_TIMEOUT_MS > 0 && SENSOR_HTTP_TIMEOUT_MS <= 60000,
              "HTTP timeout must be 1..60000 ms");
