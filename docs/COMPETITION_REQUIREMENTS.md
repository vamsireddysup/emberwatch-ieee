# IEEE HART Phase 2 Requirements Mapping

Source reviewed: Phase 2 Guidance Document and IEEE HART Q&A available 2026-07-22.

## Submission constraints

- Complete end-to-end sensor node and fabricated reader.
- Node includes temperature sensing, embedded processing, wireless communication, power
  management, and required signal conditioning.
- Reader must receive and decode transmissions; an MCU development board or Raspberry Pi
  is allowed, but a complete purchased reader subsystem is not.
- Up to USD 1,000 eligible hardware reimbursement, subject to competition rules.
- Ansys license cost is excluded from the BoM.
- Project description: maximum two pages.
- Video: maximum five minutes.
- Submission deadline: 2026-09-10.

## Evaluation evidence

| Criterion | Planned evidence |
|---|---|
| Cost | Reproducible BoM in USD including owned parts at current typical price |
| Energy | Measured Wh and energy breakdown for sense, inference, idle, and transmit |
| Size/weight | Finished node and receiver measurements |
| Range | Maximum legal demonstrated range with radio settings and method documented |
| Accuracy/resolution | Calibration fixture, reference instrument, error plot, 0.1 C reporting resolution |
| Modeling | Thermal/electrical simulation files and assumptions |
| AI innovation | ESN comparison, event detection, false alarms, radio transmissions avoided |
| Demonstration | Controlled heating precursor, on-device state, LoRa, reader, USB logger |

The organizers do not prescribe one test method. The report and video must explain and
show how each measurement was verified. LoRa testing must obey local law; jurisdictional
limits and any supplementary link-budget evidence should be stated explicitly.
