#include <DHTesp.h>
#include <Wire.h>
#include <BH1750.h>
#include <SensorHttpTransport.h>
#include <SensorCommandServer.h>

// ============================================================
// SENSOR
// ============================================================

const int NODE_ID = 2;
const int CHANNEL_COUNT = 4;

DHTesp dht;
BH1750 lightMeter;
SensorHttpTransport sensorTransport;
SensorCommandServer commandServer;

// Preserve the serial stream; networking and its results stay at this boundary.
void emitRecord(const String& csv) {
  Serial.println(csv);
  sensorTransport.upload(csv);
}


// ============================================================
// COLLECTION CONFIGURATION
// Change timing values ONLY in this section.
// ============================================================

// Select per-plant pacing only; this does not schedule acquisition.
// true: 100 ms (Wokwi), false: 500 ms (real system).
const bool SIMULATION_MODE = true;


// MUX timing
const unsigned long MUX_SETTLE_TIME =
  10UL;


// Time spent reading each plant
const unsigned long PLANT_READ_DELAY =
  SIMULATION_MODE
    ? 100UL      // Wokwi
    : 500UL;     // Real system


// ============================================================
// 4067 CHANNEL SELECTION
// ============================================================

void selectChannel(int channel) {

  digitalWrite(16, channel & 1);
  digitalWrite(17, (channel >> 1) & 1);
  digitalWrite(18, (channel >> 2) & 1);
  digitalWrite(19, (channel >> 3) & 1);
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  Serial.println(
    "timestamp,session_id,node_id,sampling_point,plant_id,soil,temperature,humidity,light"
  );

  // ----------------------------------------------------------
  // DHT22
  // ----------------------------------------------------------

  dht.setup(15, DHTesp::DHT22);


  // ----------------------------------------------------------
  // BH1750
  // SDA = GPIO 21
  // SCL = GPIO 22
  // ----------------------------------------------------------

  Wire.begin(21, 22);

  lightMeter.begin();


  // ----------------------------------------------------------
  // 4067 select pins
  // ----------------------------------------------------------

  pinMode(16, OUTPUT);
  pinMode(17, OUTPUT);
  pinMode(18, OUTPUT);
  pinMode(19, OUTPUT);

  sensorTransport.begin();
  commandServer.begin();
}


// ============================================================
// COLLECT ONE SCAN
//
// sampling_point identifies the physical sensor within this node.
// Initially sensor ID and plant ID both equal MUX channel + 1.
// Keep sensor IDs with their probes if the wiring or pot assignment changes.
// ============================================================

// Perform one complete blocking scan: read shared environmental sensors once,
// then visit channels 0 through CHANNEL_COUNT - 1 exactly once and emit one
// unchanged CSV record per channel. sessionId is supplied by the caller and is
// copied into every record; this function never assigns or advances session IDs.
// Call only after setup(), from one firmware context at a time (not reentrant).
// The caller owns invocation timing; no automatic collection is installed.
void collectSample(int sessionId) {

  // ----------------------------------------------------------
  // Read shared environmental conditions once.
  // These values apply to all 4 plants.
  // ----------------------------------------------------------

  TempAndHumidity data =
    dht.getTempAndHumidity();


  // ----------------------------------------------------------
  // Read ambient light once.
  //
  // BH1750 returns the value directly in lux.
  // ----------------------------------------------------------

  float lightValue =
    lightMeter.readLightLevel();


  // ----------------------------------------------------------
  // Scan all 4 plants
  // ----------------------------------------------------------

  for (int channel = 0; channel < CHANNEL_COUNT; channel++) {

    unsigned long timestamp = millis();


    // --------------------------------------------------------
    // Select plant/channel
    // --------------------------------------------------------

    selectChannel(channel);


    // Allow MUX output to settle
    delay(MUX_SETTLE_TIME);


    // --------------------------------------------------------
    // Soil moisture
    // --------------------------------------------------------

    int soilValue = map(
      analogRead(34),
      0,
      4095,
      0,
      100
    );


    // --------------------------------------------------------
    // CSV record
    // --------------------------------------------------------

    emitRecord(
      String(timestamp) + "," +
      String(sessionId) + "," +
      String(NODE_ID) + "," +
      String(channel + 1) + "," +
      String(channel + 1) + "," +
      String(soilValue) + "," +
      String(data.temperature) + "," +
      String(data.humidity) + "," +
      String(lightValue)
    );


    // Simulated/realistic time spent reading this plant
    delay(PLANT_READ_DELAY);
  }
}


// ============================================================
// Commands are the only acquisition trigger; Jetson owns the schedule.
// ============================================================

void loop() {
  // Only an accepted remote command invokes a scan.
  sensorTransport.poll();
  commandServer.poll(collectSample);
  delay(100);
}
