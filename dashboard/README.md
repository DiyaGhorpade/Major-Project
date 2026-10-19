# Browser dashboard demo

This dashboard replays embedded sample readings. It does not connect to the
ESP32 nodes, Jetson receiver or classifier.

## Run

Open `dashboard/index.html` directly in a browser, or serve it from the repository root:

```sh
python -m http.server 8000 --directory dashboard
```

Visit <http://localhost:8000>. Stop the server with Ctrl+C. No package installation
or build step is required. Google Fonts use internet access; fallback fonts work offline.

## Change the demo data

The page reads JSON embedded in `index.html`. The source sample file is
`data/sample_data_updated.csv`; changing that CSV alone does not change the page.
Update the embedded records to change the replay.

Some readings are intentionally invalid to demonstrate validation. The displayed
two-node/16-plant grouping is a demo mapping, not the installed hardware configuration.
For actual data collection, use the [receiver setup](../sensor_data_reciever/README.md).
