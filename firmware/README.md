# EmberWatch Firmware Reference

This folder contains portable C feature history, ESN inference, alert/radio policy, and
packet-codec contracts. Board startup, ADC conversion, sleep, and the LoRa stack remain
board-specific. The inference module uses `generated/emberwatch_model.h`, refreshed by
`python -m src.train_esn` or `python -m src.export_c`.

Add `firmware/include` and `firmware/generated` to the MCU compiler include path. The ESN
uses `tanhf` and `expf`; link the math library in host builds. Cortex-M4 builds may select
CMSIS-DSP or a verified approximation after Python/C parity has been established.

No heap allocation is used. Reservoir state RAM is `4 * reservoir_size` bytes. Generated
weights are constant flash data. The current feature code must be called every five
minutes to match training. Board integration must validate sensors before feature/model
updates.
