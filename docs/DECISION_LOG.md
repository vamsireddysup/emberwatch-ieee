# Decision Log

## 2026-07-22

### Detect ignition precursors, not wildfire

The monitored object is a transformer, recloser, connector, or related grid asset. The
primary output is abnormal asset heating relative to ambient and recent history.

### Use an ESN as the deployable temporal model

An echo state network has fixed recurrent weights and trains only a linear readout. It
captures temporal behavior while keeping MCU memory, compute, and implementation risk
small. Threshold detectors remain mandatory baselines.

### Use time and station-aware evaluation

Training uses 2020-2021, validation uses 2022, and test uses 2023. Normalization and model
weights are fit on training only. Results include held-out stations when configured and
event-level detection, not only row accuracy.

### Use USB from receiver to host

The competition receiver sends decoded LoRa payloads over framed USB serial. This avoids
network provisioning and unnecessary BoM/power cost. A later gateway may bridge USB to
Wi-Fi or Ethernet.

### Preserve all project material

No deletions. Superseded work is archived with an index entry. Generated datasets and
artifacts require explicit overwrite flags.

### Select the 48-unit ESN as the current software candidate

The larger bounded run used 400,000 rows in each split and improved synthetic test alert
recall over the 24-unit smoke model and leakage-safe thresholds. The candidate is not
frozen: MCU measurements, controlled rig data, station holdout, and size ablation remain
required before final deployment claims.
