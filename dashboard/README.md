# Monitoring dashboard

This folder owns the browser presentation layer. `index.html` contains the HTML,
styles, JavaScript, and embedded replay records, so it can be opened directly in
a browser without a build step or Python server. Google Fonts require internet
access; the page has fallback fonts.

Open `dashboard/index.html` from the repository root, or serve it locally:

```powershell
python -m http.server 8000 --directory dashboard
```

Then visit <http://localhost:8000>.

## Demo data

`data/sample_data_updated.csv` is the source dataset for the 200 records embedded
in `index.html`. The page reads its embedded JSON, not the CSV at runtime.
Editing the CSV alone does not update the dashboard. Invalid readings are
included to demonstrate rejection and validation behavior.

The demo displays two physical nodes and 16 plants, regrouping the CSV's four
node identifiers for presentation. This mapping is part of the demo.

## Component boundary

The dashboard currently has no live connection to the ESP32, Flask receiver,
Jetson, or stress classifier. Replay and validation are browser-side demo logic.
Live ingestion belongs in `../sensor_data_reciever/`, firmware in
`../hardware-code/`, and model training in `../plant_stress/`.
