#pragma once

#include <Arduino.h>
#include <WebServer.h>
#include <atomic>
#include <climits>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// The server task owns WebServer; only loop() owns sensor acquisition/upload.
// A single atomic slot covers pending AND active work. No command queue/retry.
class SensorCommandServer {
public:
  bool begin() {
    server_.on("/collect", HTTP_POST, [this]() { requestCollection(); });
    return xTaskCreate(serve, "sensor-command", 4096, this, 1, nullptr) == pdPASS;
  }

  void poll(void (*collect)(int)) {
    int session = slot_.load();
    if (session <= 0) return;
    slot_.store(-1);
    collect(session);
    slot_.store(0);
  }

private:
  std::atomic<int> slot_{0}; // 0 idle, positive pending, -1 collecting
  WebServer server_{80};

  void requestCollection() {
    if (!server_.hasArg("session_id") || server_.args() != 1) {
      server_.send(400, "text/plain", "Expected session_id form field\n");
      return;
    }
    String raw = server_.arg("session_id");
    int session = 0;
    for (unsigned int i = 0; i < raw.length(); ++i) {
      char c = raw[i];
      if (c < '0' || c > '9' || session > (INT_MAX - (c - '0')) / 10) {
        server_.send(400, "text/plain", "Invalid session_id\n");
        return;
      }
      session = session * 10 + (c - '0');
    }
    if (session == 0) {
      server_.send(400, "text/plain", "Invalid session_id\n");
      return;
    }
    int idle = 0;
    if (!slot_.compare_exchange_strong(idle, session)) {
      server_.send(409, "text/plain", "Busy\n");
      return;
    }
    server_.send(202, "text/plain", "Accepted; completion requires received records\n");
  }

  static void serve(void* context) {
    auto* self = static_cast<SensorCommandServer*>(context);
    self->server_.begin();
    for (;;) {
      self->server_.handleClient();
      vTaskDelay(pdMS_TO_TICKS(10));
    }
  }
};
