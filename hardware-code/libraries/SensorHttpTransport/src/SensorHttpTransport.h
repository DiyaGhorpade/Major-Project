#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "TransportConfig.h"

// No acquisition, buffering, or record retries live here. All methods must be
// called from the same firmware context as collectSample().
class SensorHttpTransport {
public:
  enum LocalError {
    NotConfigured = -1000,
    WifiDisconnected = -1001,
    HttpBeginFailed = -1002
  };

  void begin() {
    // A numeric address avoids DNS waits outside HTTPClient's connect timeout.
    IPAddress address;
    configured_ = SENSOR_WIFI_SSID[0] != '\0' &&
                  address.fromString(SENSOR_JETSON_HOST);
    if (!configured_) {
      lastResult_ = NotConfigured;
      return;
    }
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(false);
    WiFi.begin(SENSOR_WIFI_SSID, SENSOR_WIFI_PASSWORD);
    lastConnectAttempt_ = millis();
    // No wait for association: acquisition remains callable while offline.
  }

  void poll() {
    if (!configured_ || WiFi.status() == WL_CONNECTED) {
      return;
    }
    const unsigned long now = millis();
    if (now - lastConnectAttempt_ >= 10000UL) {
      lastConnectAttempt_ = now;
      WiFi.reconnect();
    }
  }

  // One headerless CSV row, one POST, no retries. Only HTTP 200 is acceptance.
  // Negative HTTPClient results may mean the receiver accepted the row but the
  // response was lost. Never infer definite non-delivery from such a result.
  int upload(const String& csv) {
    if (!configured_) {
      return recordResult(NotConfigured);
    }
    if (WiFi.status() != WL_CONNECTED) {
      return recordResult(WifiDisconnected);
    }

    WiFiClient client;
    HTTPClient http;
    http.setConnectTimeout(SENSOR_HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(SENSOR_HTTP_TIMEOUT_MS);
    http.setReuse(false);
    http.setFollowRedirects(HTTPC_DISABLE_FOLLOW_REDIRECTS);
    if (!http.begin(client, SENSOR_JETSON_HOST, 5000, "/upload", false)) {
      http.end();
      return recordResult(HttpBeginFailed);
    }
    http.addHeader("Content-Type", "text/plain");
    const int result = http.POST(csv);
    // Do not read an arbitrary response body: the existing receiver signals
    // acceptance with HTTP 200 after writing, and rejection with HTTP 400/500.
    http.end();
    return recordResult(result);
  }

  int lastResult() const { return lastResult_; }
  unsigned long failedUploads() const { return failedUploads_; }
  int lastFailure() const { return lastFailure_; }

private:
  int recordResult(int result) {
    lastResult_ = result;
    if (result != 200) {
      ++failedUploads_;
      lastFailure_ = result;
    }
    return result;
  }

  bool configured_ = false;
  unsigned long lastConnectAttempt_ = 0;
  int lastResult_ = 0;  // No upload attempted yet (unless configuration failed).
  unsigned long failedUploads_ = 0;
  int lastFailure_ = 0;
};
