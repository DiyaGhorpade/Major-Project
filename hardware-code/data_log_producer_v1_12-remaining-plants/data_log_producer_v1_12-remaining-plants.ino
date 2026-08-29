#include <DHTesp.h>
#include <Wire.h>
#include <BH1750.h>

// ============================================================
// SENSOR
// ============================================================

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
    "timestamp,session_id,sampling_point,plant_id,soil,temperature,humidity,light"
  );

  // ----------------------------------------------------------
  // DHT22
  // ----------------------------------------------------------

  dht.setup(4, DHTesp::DHT22);


  // ----------------------------------------------------------
  // BH1750
  // SDA = GPIO 21
  // SCL = GPIO 22
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
// COLLECT ONE SAMPLING POINT
//
// samplingPoint identifies which 5-minute measurement
// within the current session this is.
//
// samplingPoint = 1, 2, 3, 4, 5, 6
// ============================================================

void collectSample(int samplingPoint) {

  // ----------------------------------------------------------
  // Read shared environmental conditions once.
  // These values apply to all 16 plants.
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
  // Scan all 16 plants
  // ----------------------------------------------------------

  for (int channel = 0; channel < 16; channel++) {

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
      String(samplingPoint) + "," +
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


  // Number of sampling points in one session:
  //
  // 1 → 0 minutes
  // 2 → 5 minutes
  // 3 → 10 minutes
  // 4 → 15 minutes
  // 5 → 20 minutes
  // 6 → 25 minutes
  //
  // = 6 sampling points


  const int TOTAL_SAMPLES =
    COLLECTION_DURATION / SAMPLE_INTERVAL;


  // ----------------------------------------------------------
  // Run all sampling points in this session
  // ----------------------------------------------------------

  for (
    int sampleNumber = 0;
    sampleNumber < TOTAL_SAMPLES;
    sampleNumber++
  ) {


    // --------------------------------------------------------
    // Sampling point is 1–6, not 0–5
    // --------------------------------------------------------

    int samplingPoint =
      sampleNumber + 1;


    // --------------------------------------------------------
    // Determine when this sampling point is supposed to occur
    // --------------------------------------------------------

    unsigned long targetTime =
      sessionStart +
      (sampleNumber * SAMPLE_INTERVAL);


    // --------------------------------------------------------
    // If the previous scan finished early,
    // wait until the scheduled sampling point.
    //
    // If scanning already took longer than the interval,
    // do not wait.
    // --------------------------------------------------------

    unsigned long currentTime = millis();

    if (currentTime < targetTime) {

      delay(targetTime - currentTime);
    }


    // --------------------------------------------------------
    // Collect all 16 plants
    // --------------------------------------------------------

    collectSample(samplingPoint);
  }


  // ----------------------------------------------------------
  // SESSION COMPLETE
  // ----------------------------------------------------------

  sessionId++;


  // Small pause before beginning the next session
  delay(100);
}