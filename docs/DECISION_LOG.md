# Decision Log

## 2026-07-22

### State the energy claim as two separate claims

The battery-life advantage of transmitting only on change is large but is achieved almost
equally by a threshold detector, so it is a system property, not an ESN result. The ESN's
distinct advantage is detection quality: higher event recall and positive median lead
(alerting before the anomaly label rather than after). Reporting keeps these separate.
`src/energy_model.py` quantifies the system claim from real transmission counts with an
editable placeholder hardware block; `src/compare_models.py` shows the AI claim in one
table alongside baselines and quantized variants.

### Per-device threshold trim, not feature self-calibration, for station FPR

Swapping input normalization per station at inference makes FPR worse, because the readout
and threshold were fit against the global normalization. The accepted mitigation is a
per-device alert-threshold trim chosen on a commissioning period (validation split as
proxy), never lowered below the shipped threshold; it brings the RAWS stations under budget
without touching weights. A per-station readout retrain remains the fuller fix and is left
to the ML pipeline owner.

### Quantize weights only, keep float accumulation, pack the reservoir sparse

The MCU model is stored as int8 or int16 with per-output-row float scales and dequantized
during inference; accumulation stays in float. This captures the flash reduction, which is
the binding constraint, while keeping numerics close enough to verify by Python/C parity.
Full integer accumulation would additionally reduce compute but is deferred until measured
MCU latency justifies it. The reservoir is packed CSR because it is generated with a fixed
15 percent connectivity, so most entries are structurally zero and lossless to drop. The
int8-sparse model is 1709 bytes (6.59x smaller than float) with unchanged event recall;
int16-sparse is numerically identical to float at 2543 bytes. Final int8-vs-int16 choice
is deferred to the hardware flash budget and is not frozen. The float export and reference
C inference are retained unchanged.

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
