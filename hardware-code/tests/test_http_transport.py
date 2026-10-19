"""Host-only compilation/execution of the real sketches against Arduino mocks.

Run: python -m unittest discover -s hardware-code/tests -v
Requires g++; no ESP32, network access, or additional Python packages.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "libraries/SensorHttpTransport/src"

MOCK = r'''
#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <functional>
#include <map>
struct String {
  std::string s;
  unsigned int length() const { return s.size(); }
  char operator[](unsigned int i) const { return s[i]; }
  String(const char* v): s(v) {}
  String(int v): s(std::to_string(v)) {}
  String(unsigned long v): s(std::to_string(v)) {}
  String(float v) { std::ostringstream o; o<<std::fixed<<std::setprecision(2)<<v; s=o.str(); }
  String operator+(const String& v) const { return String((s+v.s).c_str()); }
};
std::vector<std::string> events, serialRows, bodies;
unsigned long clockMs=0;
unsigned long millis() { events.push_back("time"); return clockMs; }
void delay(unsigned long ms) { events.push_back("delay:"+std::to_string(ms)); clockMs+=ms; }
const int OUTPUT=1;
void pinMode(int p,int m) { assert(m==OUTPUT); events.push_back("pin:"+std::to_string(p)); }
void digitalWrite(int p,int v) { events.push_back("mux:"+std::to_string(p)+":"+std::to_string(v)); }
int analogRead(int p) { assert(p==34); events.push_back("adc"); return 2048; }
long map(long x,long a,long b,long c,long d) { return (x-a)*(d-c)/(b-a)+c; }
struct SerialMock {
  void begin(int b) { assert(b==115200); }
  void println(const String& s) { serialRows.push_back(s.s+"\r\n"); events.push_back("serial"); }
};
SerialMock Serial;
bool badReading=false;
int dhtPin=0, wireSda=0, wireScl=0;
struct TempAndHumidity { float temperature,humidity; };
struct DHTesp {
  enum {DHT22=22};
  void setup(int p,int t) { assert(t==DHT22); dhtPin=p; }
  TempAndHumidity getTempAndHumidity() { events.push_back("dht"); return {badReading?static_cast<float>(NAN):23.5f,badReading?static_cast<float>(NAN):61.25f}; }
};
struct BH1750 {
  void begin() {}
  float readLightLevel() { events.push_back("light"); return badReading?-2.f:512.f; }
};
struct WireMock { void begin(int a,int b) { wireSda=a; wireScl=b; } };
WireMock Wire;
enum {WIFI_STA=1, WL_CONNECTED=3};
struct {
  int state=0, begins=0, reconnects=0;
  void mode(int m) { assert(m==WIFI_STA); }
  void setAutoReconnect(bool b) { assert(!b); }
  void begin(const char*,const char*) { ++begins; }
  int status() { return state; }
  bool reconnect() { ++reconnects; return true; }
} WiFi;
struct IPAddress {
  bool fromString(const char* s) { unsigned a,b,c,d; char tail;
    return std::sscanf(s,"%u.%u.%u.%u%c",&a,&b,&c,&d,&tail)==4 && a<256 && b<256 && c<256 && d<256; }
};
struct WiFiClient {};
enum {HTTPC_DISABLE_FOLLOW_REDIRECTS=0};
bool httpBegin=true;
int responseCode=200, ends=0, posts=0, connects=0, reads=0;
struct HTTPClient {
  void setConnectTimeout(int t) { connects=t; }
  void setTimeout(uint16_t t) { reads=t; }
  void setReuse(bool b) { assert(!b); }
  void setFollowRedirects(int m) { assert(m==HTTPC_DISABLE_FOLLOW_REDIRECTS); }
  bool begin(WiFiClient&,String h,uint16_t p,String path,bool tls) {
    assert(h.s=="192.168.1.100" && p==5000 && path.s=="/upload" && !tls); return httpBegin; }
  void addHeader(String k,String v) { assert(k.s=="Content-Type" && v.s=="text/plain"); }
  int POST(String s) { ++posts; bodies.push_back(s.s); events.push_back("post"); return responseCode; }
  void end() { ++ends; }
};
'''

MOCK += r'''
const int HTTP_POST=1, pdPASS=1;
#define pdMS_TO_TICKS(ms) (ms)
int xTaskCreate(void (*)(void*),const char*,int,void*,int,void*) { return pdPASS; }
void vTaskDelay(int) {}
struct WebServer {
  static WebServer* instance;
  std::function<void()> handler;
  std::map<std::string,std::string> form;
  int status=0;
  explicit WebServer(int port) { assert(port==80); instance=this; }
  void on(const char* path,int method,std::function<void()> fn) {
    assert(std::string(path)=="/collect" && method==HTTP_POST); handler=fn;
  }
  bool hasArg(const char* name) { return form.count(name); }
  int args() { return form.size(); }
  String arg(const char* name) { return String(form[name].c_str()); }
  void send(int code,const char*,const char*) { status=code; }
  void begin() {}
  void handleClient() {}
};
WebServer* WebServer::instance=nullptr;
'''

MAIN = r'''
int main() {
  assert(NODE_ID==EXPECTED_NODE && CHANNEL_COUNT==EXPECTED_CHANNELS);
  setup();
  assert(dhtPin==EXPECTED_DHT && wireSda==EXPECTED_SDA && wireScl==EXPECTED_SCL);
  assert(serialRows.size()==1 && serialRows[0]==
    "timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light\r\n");
  assert(WiFi.begins==1 && posts==0 && clockMs==0);
  for(int i=0;i<100;++i) loop();
  assert(posts==0 && serialRows.size()==1); // No automatic acquisition.
  loop();
  assert(WiFi.reconnects==1); // No busy connection loop.
  WiFi.state=WL_CONNECTED;
  clockMs=0; events.clear(); serialRows.clear();
  collectSample(7);
  assert(serialRows.size()==CHANNEL_COUNT && bodies.size()==CHANNEL_COUNT);
  std::vector<std::string> expected={"dht","light"};
  for(int c=0;c<CHANNEL_COUNT;++c) {
    expected.push_back("time");
    for(int b=0;b<4;++b) expected.push_back("mux:"+std::to_string(16+b)+":"+std::to_string((c>>b)&1));
    expected.push_back("delay:10"); expected.push_back("adc");
    expected.push_back("serial"); expected.push_back("post");
    expected.push_back("delay:"+std::to_string(PLANT_READ_DELAY));
    std::string csv=std::to_string(c*(10+PLANT_READ_DELAY))+",7,"+std::to_string(NODE_ID)+","+
      std::to_string(c+1)+","+std::to_string(c+1)+",50,23.50,61.25,512.00";
    assert(bodies[c]==csv && serialRows[c]==csv+"\r\n");
  }
  assert(events==expected && sensorTransport.lastResult()==200 && sensorTransport.failedUploads()==0);
  assert(connects==2000 && reads==3000 && ends==posts);
  int previous=posts;
  for(int code : {400,500,302,-1,-11}) {
    responseCode=code;
    assert(sensorTransport.upload("one,row")==code);
    assert(posts==++previous && ends==posts); // Exactly one attempt and cleanup.
  }
  assert(sensorTransport.failedUploads()==5 && sensorTransport.lastFailure()==-11);
  responseCode=200;
  assert(sensorTransport.upload("one,row")==200 && sensorTransport.failedUploads()==5);
  assert(sensorTransport.lastFailure()==-11); // Success doesn't erase failure evidence.
  httpBegin=false;
  assert(sensorTransport.upload("one,row")==SensorHttpTransport::HttpBeginFailed);
  assert(posts==previous+1 && ends==posts+1);
  httpBegin=true;
  WiFi.state=0;
  assert(sensorTransport.upload("one,row")==SensorHttpTransport::WifiDisconnected);
  serialRows.clear(); badReading=true;
  collectSample(42); // Offline acquisition still reads and emits all records.
  assert(serialRows.size()==CHANNEL_COUNT);
  for(const auto& row:serialRows) {
    assert(row.find(",42,"+std::to_string(NODE_ID)+",")!=std::string::npos);
    assert(row.find(",nan,nan,-2.00\r\n")!=std::string::npos);
  }
}
'''


@unittest.skipUnless(shutil.which("g++"), "g++ is required for host mock compilation")
class TransportTests(unittest.TestCase):
    def compile_run(self, source, defines=""):
        with tempfile.TemporaryDirectory(prefix="sensor-http-test-") as directory:
            tmp = Path(directory)
            # Use production headers without importing a developer's credentials.
            for name in ("SensorHttpTransport.h", "TransportConfig.h", "SensorCommandServer.h"):
                shutil.copyfile(LIBRARY / name, tmp / name)
            (tmp / "Mock.h").write_text(MOCK)
            for name in ("Arduino.h", "WiFi.h", "HTTPClient.h", "DHTesp.h", "Wire.h", "BH1750.h", "WebServer.h"):
                (tmp / name).write_text('#include "Mock.h"\n')
            (tmp / "freertos").mkdir()
            for name in ("FreeRTOS.h", "task.h"):
                (tmp / "freertos" / name).write_text('#include "Mock.h"\n')
            cpp = tmp / "test.cpp"
            exe = tmp / "test.exe"
            cpp.write_text(defines + '\n#include "Mock.h"\n' + source)
            command = ["g++", "-std=c++11", "-Wall", "-Wextra", "-Werror", "-I", str(tmp),
                       "-I", str(LIBRARY), str(cpp), "-o", str(exe)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

    def test_both_sketches_and_pacing_modes(self):
        for sketch in sorted(ROOT.glob("data_log_producer*/*.ino")):
            source = sketch.read_text()
            node = 1 if "12-remaining" in sketch.name else 2
            for simulation in (True, False):
                with self.subTest(node=node, simulation=simulation):
                    code = source if simulation else source.replace("SIMULATION_MODE = true", "SIMULATION_MODE = false")
                    pins = (4, 25, 26) if node == 1 else (15, 21, 22)
                    defines = '#define SENSOR_WIFI_SSID "test"\n#define SENSOR_JETSON_HOST "192.168.1.100"\n'
                    defines += f"#define EXPECTED_NODE {node}\n#define EXPECTED_CHANNELS {12 if node == 1 else 4}\n"
                    defines += f"#define EXPECTED_DHT {pins[0]}\n#define EXPECTED_SDA {pins[1]}\n#define EXPECTED_SCL {pins[2]}\n"
                    self.compile_run(code + MAIN, defines)

    def test_command_validation_busy_and_completion(self):
        self.compile_run(r'''#include <SensorCommandServer.h>
SensorCommandServer commands;
int collected=0;
void collect(int session) {
  assert(session==123); ++collected;
  WebServer::instance->handler();
  assert(WebServer::instance->status==409); // Busy during actual acquisition.
}
int main() {
  assert(commands.begin());
  auto& server=*WebServer::instance;
  for(const char* bad: {"", "0", "-1", "1x", "2147483648"}) {
    server.form={{"session_id",bad}}; server.handler(); assert(server.status==400);
  }
  server.form={}; server.handler(); assert(server.status==400);
  server.form={{"session_id","123"}}; server.handler(); assert(server.status==202);
  assert(collected==0); // Acceptance does not run acquisition in server task.
  server.handler(); assert(server.status==409); // Pending command is also busy.
  commands.poll(collect); assert(collected==1);
  commands.poll(collect); assert(collected==1); // No automatic repeat.
  server.handler(); assert(server.status==202); // Idle after completion.
}
''')

    def test_missing_or_invalid_configuration(self):
        code = '''#include <SensorHttpTransport.h>
int main() { SensorHttpTransport t; t.begin(); t.poll();
assert(WiFi.begins==0 && WiFi.reconnects==0);
assert(t.upload("row")==SensorHttpTransport::NotConfigured);
assert(t.failedUploads()==1 && posts==0); }'''
        self.compile_run(code)
        self.compile_run(code, '#define SENSOR_WIFI_SSID "test"\n#define SENSOR_JETSON_HOST "not-a-url"\n')

    def test_no_acquisition_trigger_or_schedule(self):
        for sketch in ROOT.glob("data_log_producer*/*.ino"):
            code = sketch.read_text()
            self.assertEqual(code.count("collectSample("), 1)
            self.assertIn("commandServer.poll(collectSample)", code)
            for removed in ("sessionId++", "SAMPLE_INTERVAL", "COLLECTION_DURATION", "targetTime"):
                self.assertNotIn(removed, code)


if __name__ == "__main__":
    unittest.main()
