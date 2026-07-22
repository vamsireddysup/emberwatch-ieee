# EmberWatch Hardware Integration and Execution Guide

This is the step-by-step guide for taking the completed software to physical hardware:
wire the sensors, generate and integrate the model, write the board-specific glue, flash
the node, run the receiver and dashboard on a host, and verify end to end.

It assumes the software in this repository as of the current `master`. It does not assume
any board decision has been finalized; where a value is still owned by the hardware team it
is called out explicitly and cross-referenced to `docs/HARDWARE_INTEGRATION.md`.

Read `docs/HARDWARE_INTEGRATION.md` (measurement chain, cadence, required constants) and
`docs/VALIDATION_PLAN.md` (acceptance gates) alongside this guide. This guide is the "how",
those two are the "contract" and the "proof".

---

## 0. What is ready vs what you must write

**Ready in this repo (portable C, no board assumptions, parity-tested against Python):**

| Module | Header | Source | Role |
|---|---|---|---|
| Feature history | `firmware/include/emberwatch_features.h` | `firmware/src/emberwatch_features.c` | Builds the 6 model features from asset/ambient temperature every 5 min |
| Float inference | `firmware/include/emberwatch_inference.h` | `firmware/src/emberwatch_inference.c` | Reference ESN, float32 weights |
| Quantized inference | `firmware/include/emberwatch_inference_q.h` | `firmware/src/emberwatch_inference_q.c` | int8/int16 ESN, ~1.7 KB, recommended for flash-limited targets |
| Alert/radio policy | `firmware/include/emberwatch_policy.h` | `firmware/src/emberwatch_policy.c` | Persistence + heartbeat gating of transmissions |
| Packet codec | `firmware/include/emberwatch_protocol.h` | `firmware/src/emberwatch_protocol.c` | 22-byte telemetry encode/decode + CRC |
| Model constants | `firmware/generated/emberwatch_model.h` (float) and `emberwatch_model_q.h` (quantized) | generated | Weights, normalization, threshold |

**You (hardware/firmware team) must write the board-specific glue. It is intentionally not
in this repo because it depends on the final part choices still marked "Awaiting team" in
`docs/HARDWARE_INTEGRATION.md`:**

- MCU startup, clock, and low-power (Stop2) sleep entry/exit.
- Two-channel NTC ADC read and the ADC-code-to-Celsius conversion (formula in section 3).
- LoRa radio driver: initialize the SX126x, transmit the 22-byte packet.
- The 5-minute wake timer (RTC or LPTIM).
- Sensor open/short/disconnect detection (must bypass the model, section 6).

The main loop that ties ready modules to your glue is sketched in section 5.

---

## 1. Host prerequisites

On the development machine (macOS/Linux), from the repo root:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Confirm everything builds and passes before touching hardware:

```bash
make test
```

Expected: the Python suite passes, and the C protocol, features/policy, and inference
objects compile under `-Wall -Wextra -Werror`.

---

## 2. Generate the deployable model

The model header is generated from the trained ESN. If `artifacts/models/esn_model.npz`
already exists you can skip training and just export. To regenerate everything:

```bash
# (optional) regenerate the physics-grounded dataset, then retrain:
make generate-v2
make train-esn        # writes artifacts/models/esn_model.npz + firmware/generated/emberwatch_model.h

# quantized header (recommended for the MCU): writes firmware/generated/emberwatch_model_q.h
make export-quant
```

Pick one inference path and stay consistent:

- **Quantized (recommended):** `emberwatch_model_q.h` + `emberwatch_inference_q.c`. ~1.7 KB
  of constants (int8) or ~2.5 KB (int16). Event recall is identical to float; see
  `docs/generated/QUANTIZATION_RESULTS.md`. Use int16 (`make export-quant` then
  `python -m src.export_c_quant --bits 16`) if you want numerically-identical-to-float and
  have the flash.
- **Float reference:** `emberwatch_model.h` + `emberwatch_inference.c`. Use only if you are
  debugging a discrepancy; it is ~11 KB of constants.

The model is trained for a **5-minute** sample cadence. Changing the cadence invalidates
the slope and reservoir-memory features and requires retraining (see
`docs/HARDWARE_INTEGRATION.md`). Do not change it without regenerating the dataset.

---

## 3. Wire the sensors

Two NTC thermistors: one clamped to the monitored asset, one measuring nearby ambient air.
Each in a divider with a fixed resistor. For a low-side NTC with pull-up `R_FIXED`, ADC
full-scale code `ADC_MAX`, measured code `adc`:

```text
R_NTC = R_FIXED * adc / (ADC_MAX - adc)
1/T_K = 1/T0_K + ln(R_NTC / R0) / BETA
T_C  = T_K - 273.15
```

`R0` is the thermistor nominal resistance at `T0_K` (usually 298.15 K = 25 C); `BETA` is
the datasheet beta, or use the manufacturer Steinhart-Hart coefficients for better
accuracy. Reverse the first line for a high-side NTC. These constants are hardware-owned;
record the exact parts, divider topology, resistor value/tolerance, and ADC bits/reference
in `docs/HARDWARE_INTEGRATION.md` before freezing firmware.

Keep the two channels thermally isolated from each other so ambient does not track the
asset. Oversample and average each channel after settling (reject min/max).

---

## 4. Add the model to your firmware build

1. Copy or reference the ready C sources into your firmware project:
   `firmware/src/emberwatch_features.c`, your chosen inference source
   (`emberwatch_inference_q.c` recommended), `emberwatch_policy.c`,
   `emberwatch_protocol.c`.
2. Add both `firmware/include` and `firmware/generated` to the compiler include path.
3. The ESN uses `tanhf` and `expf` from `<math.h>`. Link the math library (host builds:
   `-lm`). On the M4 you may later swap in CMSIS-DSP or a verified approximation, but only
   after Python/C parity is re-confirmed.
4. No heap is used. Reservoir RAM is `4 * reservoir_size` bytes (192 bytes for the 48-unit
   model). Generated weights are `const` flash.

---

## 5. Main loop skeleton

The ready modules connect like this. Everything in `>>>` comments is board-specific glue
you write.

```c
#include "emberwatch_features.h"
#include "emberwatch_inference_q.h"   /* or emberwatch_inference.h for the float path */
#include "emberwatch_policy.h"
#include "emberwatch_protocol.h"

static ew_feature_history_t history;
static ew_q_state_t         model_state;   /* ew_model_state_t for the float path */
static ew_alert_policy_t    policy;

void emberwatch_init(void) {
    ew_features_reset(&history);
    ew_q_reset(&model_state);
    ew_alert_policy_reset(&policy, /*heartbeat_samples=*/72);  /* 72 * 5 min = 6 h */
}

/* Call once every 5 minutes from your RTC/LPTIM wake handler. */
void emberwatch_tick(void) {
    /* >>> read both NTC channels, convert to Celsius (section 3) */
    float asset_c   = board_read_asset_temp_c();
    float ambient_c = board_read_ambient_temp_c();

    /* >>> if a sensor is open/short/disconnected: send a SensorFault packet and
       >>> return WITHOUT running the model (acceptance gate, section 6). */

    float features[EW_FEATURE_COUNT];
    if (!ew_features_update(&history, asset_c, ambient_c, features)) {
        return;  /* history still filling; no inference yet */
    }

    ew_q_output_t out = ew_q_predict(&model_state, features);

    ew_alert_decision_t decision = ew_alert_policy_step(
        &policy, out.probabilities[1], out.probabilities[2],
        EW_Q_ALERT_THRESHOLD, /*warming_confirmations=*/2,
        /*clear_confirmations=*/3, /*heartbeat_samples=*/72);

    if (decision.transmit) {
        ew_telemetry_t t = {0};
        t.device_id = BOARD_DEVICE_ID;
        t.sequence  = next_sequence++;
        t.uptime_s  = board_uptime_s();
        t.asset_temp_c_x100   = (int16_t)(asset_c   * 100.0f);
        t.ambient_temp_c_x100 = (int16_t)(ambient_c * 100.0f);
        t.confidence_u8 = (uint8_t)(out.confidence * 255.0f);
        t.state = decision.state;
        t.battery_mv = board_battery_mv();
        t.flags = /* set EW_FLAG_ALERT / BATTERY_LOW / SENSOR_FAULT / MODEL_VALID */;

        uint8_t packet[EW_PACKET_SIZE];
        ew_encode_telemetry(&t, packet);
        /* >>> board_lora_transmit(packet, EW_PACKET_SIZE); */
    }
    /* >>> enter Stop2 sleep until the next 5-minute wake */
}
```

The `EW_Q_ALERT_THRESHOLD` macro comes from the generated header. See section 9 for the
per-device threshold trim.

---

## 6. Bring-up order (do these in sequence)

From `docs/HARDWARE_INTEGRATION.md`, expanded:

1. **Sensors first.** Stream raw ADC and converted Celsius over USB. Compare both channels
   against a reference probe across at least five stable points plus slow ramps
   (`docs/VALIDATION_PLAN.md` section 1). Confirm 0.1 C display resolution.
2. **Fault handling.** Verify open/short/disconnect on each channel is detected and produces
   an explicit `EW_STATE_SENSOR_FAULT` packet that bypasses the model. This is an acceptance
   gate, not optional.
3. **Feature parity.** Stream all six C features for a fixed temperature sequence and compare
   row-for-row against a Python fixture (`make test` covers the Python/C model path; you add
   the feature-stream comparison on hardware). Require agreement within a documented
   tolerance before trusting the model.
4. **Model parity on device.** Run the generated ESN in C on an identical feature sequence
   and confirm probabilities match Python. `tests/test_quantized_parity.py` and
   `tests/test_c_parity.py` already prove this host-side; reproduce on the target.
5. **Packet + radio.** Encode a packet in firmware, transmit, and decode it with the host
   receiver (section 7). Confirm CRC, magic, version, and sequence handling.
6. **Power.** Measure current for sleep, sense, inference, encode, and transmit; integrate
   energy (`docs/VALIDATION_PLAN.md` section 4). Replace the placeholder numbers in
   `src/energy_model.py` `HardwareAssumptions` with the measured values and rerun
   `make energy` to get real battery-life numbers.

---

## 7. Run the receiver and dashboard on the host

The custom reader (an MCU dev board or Raspberry Pi running your radio-to-USB firmware)
emits one ASCII line per valid LoRa reception:

```text
EW1,<44 lowercase hex chars>,<rssi_dbm>,<snr_db>\n
```

Log real hardware over USB serial:

```bash
./venv/bin/python -m src.receiver --port /dev/tty.usbmodemXXXX --output artifacts/telemetry/field.csv
```

Or dry-run the whole path with no hardware (simulated frames piped into the receiver):

```bash
./venv/bin/python -m src.simulate_receiver --count 20 --interval 0.2 \
  | ./venv/bin/python -m src.receiver --output artifacts/telemetry/demo.csv
```

Then view it live:

```bash
./venv/bin/python -m src.dashboard --log artifacts/telemetry/field.csv
```

The receiver validates framing, length, magic, version, and CRC; invalid lines are counted
and reported, never logged as telemetry.

---

## 8. Flashing / pushing code to the node

Board-toolchain specific, so exact commands depend on your final MCU and programmer. The
general sequence once the glue in section 5 is written:

1. Build the firmware image with your board toolchain (STM32CubeIDE, `arm-none-eabi-gcc` +
   your Makefile/CMake, or PlatformIO). Include the EmberWatch C sources and both include
   directories from section 4.
2. Connect the programmer (ST-LINK for STM32) to the node.
3. Flash (for example `STM32_Programmer_CLI -c port=SWD -w firmware.elf -rst`, or the IDE's
   flash button, or `openocd` with the matching board config).
4. Open a serial monitor at your chosen baud to watch the bring-up stream from section 6.
5. Regenerating the model later (`make train-esn` / `make export-quant`) only changes the
   generated headers; rebuild and reflash to pick up new weights. Nothing else changes.

Keep the model cadence at 5 minutes unless you retrain (section 2).

---

## 9. Per-device commissioning (recommended)

`docs/generated/STATION_ROBUSTNESS.md` found that a single fleet-wide alert threshold leaves
the higher-variance sites above the false-positive budget, and that a per-device threshold
trim fixes it without changing any model weight. In the field this means:

1. After install, run the node through a normal (non-fault) period and log its model output.
2. Raise the per-device alert threshold (the value passed to `ew_alert_policy_step`) above
   the shipped `EW_*_ALERT_THRESHOLD` until the observed Normal-period false-alert rate meets
   your budget. Never lower it below the shipped value.
3. Store that trimmed threshold in device configuration.

This is cheaper than per-device retraining and touches no weights. Do not lower the shipped
threshold; that can only add false alerts.

---

## 10. Verification checklist before the demo

- [ ] `make test` passes on the host.
- [ ] Sensor calibration report exists with bias/MAE/RMSE and 0.1 C resolution.
- [ ] Sensor faults produce an explicit fault packet and bypass the model.
- [ ] C features match Python within the documented tolerance on real ADC data.
- [ ] C model probabilities match Python on the target MCU.
- [ ] A real LoRa packet round-trips through `src/receiver.py` with valid CRC.
- [ ] Measured per-state energy recorded; `src/energy_model.py` assumptions updated; battery
      life recomputed with `make energy`.
- [ ] RF range/RSSI/SNR/packet-delivery recorded with radio settings and jurisdiction.
- [ ] Controlled-rig fault trials run, with whole trials held out of training when reporting
      detection (`docs/VALIDATION_PLAN.md` section 3).
- [ ] Synthetic, rig, and field numbers are labeled separately in all reporting.

---

## 11. Troubleshooting

- **C model disagrees with Python:** confirm the same header (float vs quantized) is used on
  both sides, the same feature order, and that history is warmed up (`ew_features_update`
  returns false until ready). Re-run `tests/test_quantized_parity.py` host-side to isolate.
- **Model always Normal or always Anomaly:** check ADC-to-Celsius conversion and channel
  order (asset vs ambient swapped inverts `delta_c`). Stream features and compare to Python.
- **Too many transmissions:** verify the heartbeat and confirmation counts match section 5,
  and apply the per-device threshold trim (section 9).
- **Receiver logs nothing:** check the serial port and baud, and that the reader emits the
  exact `EW1,<hex>,<rssi>,<snr>` frame; invalid frames are counted, not logged.
- **Model header missing:** run `make export-quant` (quantized) or `make train-esn` (float).

---

## Quick command reference

```bash
make test           # host build + tests
make train-esn      # retrain ESN, refresh float header
make export-quant   # generate quantized header + compile-check quantized C
make energy         # battery-life model (edit HardwareAssumptions first with measured values)
make compare        # one comparison table across baselines and ESN variants
make robustness     # per-station FPR analysis + threshold-trim recommendation
./venv/bin/python -m src.receiver --port <PORT> --output artifacts/telemetry/field.csv
./venv/bin/python -m src.dashboard --log artifacts/telemetry/field.csv
```
