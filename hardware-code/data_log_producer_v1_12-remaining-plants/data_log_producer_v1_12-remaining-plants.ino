#include <DHTesp.h>
#include <Wire.h>
#include <BH1750.h>

// ============================================================
// SENSOR
// ============================================================

const int NODE_ID = 1;
const int CHANNEL_COUNT = 12;

DHTesp dht;
BH1750 lightMeter;


// ============================================================
// COLLECTION CONFIGURATION
// Change timing values ONLY in this section.
// ============================================================

// Set to true for Wokwi accelerated simulation.
// Set to false for real-time operation.
const bool SIMULATION_MODE = true;


// Real-world collection requirements
const unsigned long REAL_SAMPLE_INTERVAL =
  5UL * 60UL * 1000UL;                 // 5 minutes

const unsigned long REAL_SESSION_DURATION =
  30UL * 60UL * 1000UL;                // 30 minutes


// Wokwi simulation speed
// 1 simulated minute = 1 real second
const unsigned long SIMULATED_MINUTE =
  1000UL;


// Automatically select the correct timing
const unsigned long SAMPLE_INTERVAL =
  SIMULATION_MODE
    ? 5UL * SIMULATED_MINUTE
    : REAL_SAMPLE_INTERVAL;

const unsigned long COLLECTION_DURATION =
  SIMULATION_MODE
    ? 30UL * SIMULATED_MINUTE
    : REAL_SESSION_DURATION;


// MUX timing
const unsigned long MUX_SETTLE_TIME =
  10UL;


// Time spent reading each plant
const unsigned long PLANT_READ_DELAY =
  SIMULATION_MODE
    ? 100UL      // Wokwi
    : 500UL;     // Real system


// ============================================================
// SESSION STATE
// ============================================================

int sessionId = 1;


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

  dht.setup(4, DHTesp::DHT22);


  // ----------------------------------------------------------
  // BH1750
  // SDA = GPIO 25
  // SCL = GPIO 26
  // ----------------------------------------------------------

  Wire.begin(25, 26);

  lightMeter.begin();


  // ----------------------------------------------------------
  // 4067 select pins
  // ----------------------------------------------------------

  pinMode(16, OUTPUT);
  pinMode(17, OUTPUT);
  pinMode(18, OUTPUT);
  pinMode(19, OUTPUT);
}


// ============================================================
// COLLECT ONE SCAN
//
// sampling_point identifies the physical sensor within this node.
// Initially sensor ID and plant ID both equal MUX channel + 1.
// Keep sensor IDs with their probes if the wiring or pot assignment changes.
// ============================================================

void collectSample() {

  // ----------------------------------------------------------
  // Read shared environmental conditions once.
  // These values apply to all 12 plants.
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
  // Scan all 12 plants
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

    Serial.println(
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
// MAIN COLLECTION SESSION
// ============================================================

void loop() {

  unsigned long sessionStart = millis();


  // Number of scans in one session:
  //
  // 1 → 0 minutes
  // 2 → 5 minutes
  // 3 → 10 minutes
  // 4 → 15 minutes
  // 5 → 20 minutes
  // 6 → 25 minutes
  //
  // = 6 scans


  const int TOTAL_SAMPLES =
    COLLECTION_DURATION / SAMPLE_INTERVAL;


  // ----------------------------------------------------------
  // Run all scans in this session
  // ----------------------------------------------------------

  for (
    int sampleNumber = 0;
    sampleNumber < TOTAL_SAMPLES;
    sampleNumber++
  ) {


    // --------------------------------------------------------
    // sampleNumber is the scan index, independent of sensor identity
    // --------------------------------------------------------



    // --------------------------------------------------------
    // Determine when this scan is supposed to occur
    // --------------------------------------------------------

    unsigned long targetTime =
      sessionStart +
      (sampleNumber * SAMPLE_INTERVAL);


    // --------------------------------------------------------
    // If the previous scan finished early,
    // wait until the scheduled scan.
    //
    // If scanning already took longer than the interval,
    // do not wait.
    // --------------------------------------------------------

    unsigned long currentTime = millis();

    if (currentTime < targetTime) {

      delay(targetTime - currentTime);
    }


    // --------------------------------------------------------
    // Collect all 12 plants
    // --------------------------------------------------------

    collectSample();
  }


  // ----------------------------------------------------------
  // SESSION COMPLETE
  // ----------------------------------------------------------

  sessionId++;


  // Small pause before beginning the next session
  delay(100);
}