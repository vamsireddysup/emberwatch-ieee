"""Radio energy and battery-life model for EmberWatch.

Purpose: turn the model's transmission behavior into the energy and battery-life numbers
the competition scores, and to state the AI claim honestly.

Two distinct claims come out of this and they must not be conflated:

1. SYSTEM claim (gating vs always-on): an on-device classifier that stays silent unless
   something changes avoids the vast majority of radio transmissions and multiplies
   battery life. This is large but it is NOT unique to the ESN -- a good threshold
   detector gates almost as well. It is the value of doing any on-device decision at all.

2. AI claim (ESN vs threshold baseline): at a similar transmission budget the ESN detects
   more events and warns earlier (see docs/generated/ML_RESULTS.md and
   docs/generated/MODEL_COMPARISON.md). That, not packet count, is the ESN's edge.

This module quantifies claim 1 rigorously and points to the comparison report for claim 2.

Method: over a device-day of `SAMPLES_PER_DAY` wake/sense/inference cycles, only a
fraction end in a LoRa transmission. Daily charge draw is:

    Q_day = Q_sleep + Q_sense + Q_inference + Q_transmit

    Q_sleep      = I_sleep * seconds_asleep
    Q_sense      = SAMPLES_PER_DAY * I_active * t_sense
    Q_inference  = SAMPLES_PER_DAY * I_active * t_inference
    Q_transmit   = transmissions_per_day * I_tx * t_tx

Battery life (days) = usable_capacity / Q_day, where usable_capacity applies a derating
factor to the nameplate mAh.

EVERY hardware number below is a documented PLACEHOLDER drawn from STM32WL55 and SX126x
datasheet-typical values. They are collected in one dataclass, `HardwareAssumptions`, so
the hardware team can replace them in exactly one place once real measurements exist. The
model's transmission counts, by contrast, come from the actual pipeline reports and are
not assumptions.

Usage:
    python -m src.energy_model                 # uses artifacts/reports if present
    python -m src.energy_model --json out.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ESN_REPORT = REPO_ROOT / "artifacts" / "reports" / "esn_metrics.json"
BASELINE_REPORT = REPO_ROOT / "artifacts" / "reports" / "baseline_metrics.json"
DEFAULT_JSON = REPO_ROOT / "artifacts" / "reports" / "energy_metrics.json"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "generated" / "ENERGY_RESULTS.md"

SAMPLES_PER_DAY = 288  # one 5-minute wake/sense/inference cycle every five minutes


@dataclass(frozen=True)
class HardwareAssumptions:
    """PLACEHOLDER hardware values. Replace with measured numbers from the hardware team.

    Sources are datasheet-typical, not measured:
      - STM32WL55 Stop2 sleep ~1.5 uA; Run at ~48 MHz ~5 mA (used for sense+inference).
      - SX126x LoRa TX at +14 dBm ~45 mA; a 22-byte payload at SF9/125 kHz is ~120 ms.
      - Sense settling + oversampled ADC on two NTC channels ~20 ms.
      - ESN inference for a 48-unit reservoir is well under 5 ms on M4; 5 ms is a safe cap.
      - Battery: 2x AA Li/FeS2 ~2700 mAh nameplate; 0.75 derating for temperature,
        self-discharge, and cutoff voltage over a multi-year deployment.
    """

    i_sleep_ua: float = 1.5
    i_active_ma: float = 5.0
    i_tx_ma: float = 45.0
    t_sense_s: float = 0.020
    t_inference_s: float = 0.005
    t_tx_s: float = 0.120
    battery_capacity_mah: float = 2700.0
    battery_derating: float = 0.75

    @property
    def usable_capacity_mah(self) -> float:
        return self.battery_capacity_mah * self.battery_derating


def daily_charge_uah(transmissions_per_day: float, hw: HardwareAssumptions) -> dict:
    """Return the per-day charge budget in micro-amp-hours, broken out by state."""
    active_seconds_per_day = SAMPLES_PER_DAY * (hw.t_sense_s + hw.t_inference_s)
    tx_seconds_per_day = transmissions_per_day * hw.t_tx_s
    sleep_seconds_per_day = 86400.0 - active_seconds_per_day - tx_seconds_per_day

    # charge (uAh) = current (uA) * time (s) / 3600
    sense = SAMPLES_PER_DAY * hw.i_active_ma * 1000.0 * hw.t_sense_s / 3600.0
    inference = SAMPLES_PER_DAY * hw.i_active_ma * 1000.0 * hw.t_inference_s / 3600.0
    transmit = transmissions_per_day * hw.i_tx_ma * 1000.0 * hw.t_tx_s / 3600.0
    sleep = hw.i_sleep_ua * sleep_seconds_per_day / 3600.0
    total = sense + inference + transmit + sleep
    return {
        "sleep_uah": sleep,
        "sense_uah": sense,
        "inference_uah": inference,
        "transmit_uah": transmit,
        "total_uah": total,
    }


def battery_life_days(transmissions_per_day: float, hw: HardwareAssumptions) -> float:
    total_uah = daily_charge_uah(transmissions_per_day, hw)["total_uah"]
    return hw.usable_capacity_mah * 1000.0 / total_uah


def _transmissions_per_day_from_fraction(avoided_fraction: float) -> float:
    """Convert a 'transmissions avoided vs periodic' fraction into transmissions/day."""
    return SAMPLES_PER_DAY * (1.0 - avoided_fraction)


def _read_avoided_fraction(report_path: Path, keys: list) -> float | None:
    if not report_path.exists():
        return None
    data = json.loads(report_path.read_text())
    node = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node)


def build_scenarios() -> dict:
    """Assemble transmissions/day for always-on, baseline-gated, and ESN-gated policies.

    Always-on transmits every sample. The gated fractions come from the real pipeline
    reports when available, else fall back to the last recorded values with a note.
    """
    esn_avoided = _read_avoided_fraction(
        ESN_REPORT, ["test", "radio", "transmissions_avoided_fraction"]
    )
    baseline_avoided = _read_avoided_fraction(
        BASELINE_REPORT, ["models", "fixed_delta_3sigma", "radio", "transmissions_avoided_fraction"]
    )

    scenarios = {}
    scenarios["always_on"] = {
        "transmissions_per_day": float(SAMPLES_PER_DAY),
        "source": "every sample transmitted (reference)",
    }
    if baseline_avoided is not None:
        scenarios["baseline_gated"] = {
            "transmissions_per_day": _transmissions_per_day_from_fraction(baseline_avoided),
            "source": f"fixed_delta_3sigma, avoided={baseline_avoided:.4f} (baseline_metrics.json)",
        }
    if esn_avoided is not None:
        scenarios["esn_gated"] = {
            "transmissions_per_day": _transmissions_per_day_from_fraction(esn_avoided),
            "source": f"ESN operational policy, avoided={esn_avoided:.4f} (esn_metrics.json)",
        }
    return scenarios


def compute(hw: HardwareAssumptions) -> dict:
    scenarios = build_scenarios()
    reference = None
    results = {"assumptions": asdict(hw), "samples_per_day": SAMPLES_PER_DAY, "scenarios": {}}
    for name, scenario in scenarios.items():
        tx_per_day = scenario["transmissions_per_day"]
        charge = daily_charge_uah(tx_per_day, hw)
        life_days = battery_life_days(tx_per_day, hw)
        if name == "always_on":
            reference = life_days
        results["scenarios"][name] = {
            **scenario,
            "charge_breakdown_uah": charge,
            "battery_life_days": life_days,
            "battery_life_years": life_days / 365.25,
        }
    for name, entry in results["scenarios"].items():
        entry["battery_multiplier_vs_always_on"] = entry["battery_life_days"] / reference
    return results


def write_markdown(results: dict, path: Path) -> None:
    hw = results["assumptions"]
    lines = [
        "# Generated Energy and Battery-Life Results",
        "",
        "Radio-energy model output. The transmission counts are from the pipeline reports;",
        "the hardware currents/timings are documented PLACEHOLDERS (see the table) and must",
        "be replaced with measured values before any energy claim is final.",
        "",
        "## Two separate claims",
        "",
        "1. **System (gating vs always-on):** silent-unless-changed operation avoids most",
        "   transmissions and multiplies battery life. Large, but a threshold detector also",
        "   achieves it; it is the value of any on-device decision.",
        "2. **AI (ESN vs threshold):** at a similar transmission budget the ESN catches more",
        "   events and warns earlier. See `docs/generated/MODEL_COMPARISON.md`. That is the",
        "   ESN's real advantage, not packet count.",
        "",
        "## Battery life by policy",
        "",
        "| Policy | Transmissions/day | Daily draw (uAh) | Battery life (days) | Years | x vs always-on |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in results["scenarios"].items():
        lines.append(
            f"| {name} | {entry['transmissions_per_day']:.1f} | "
            f"{entry['charge_breakdown_uah']['total_uah']:.1f} | "
            f"{entry['battery_life_days']:.0f} | {entry['battery_life_years']:.2f} | "
            f"{entry['battery_multiplier_vs_always_on']:.1f}x |"
        )
    lines += [
        "",
        "## Placeholder hardware assumptions",
        "",
        "| Parameter | Value | Note |",
        "|---|---:|---|",
        f"| Sleep current | {hw['i_sleep_ua']} uA | STM32WL55 Stop2, datasheet-typical |",
        f"| Active current | {hw['i_active_ma']} mA | MCU run at ~48 MHz for sense+inference |",
        f"| TX current | {hw['i_tx_ma']} mA | SX126x LoRa +14 dBm |",
        f"| Sense time | {hw['t_sense_s'] * 1000:.0f} ms | two NTC channels, settled + oversampled |",
        f"| Inference time | {hw['t_inference_s'] * 1000:.0f} ms | 48-unit ESN on M4, conservative cap |",
        f"| TX time | {hw['t_tx_s'] * 1000:.0f} ms | 22-byte payload at ~SF9/125 kHz |",
        f"| Battery capacity | {hw['battery_capacity_mah']:.0f} mAh | e.g. 2x AA Li/FeS2 |",
        f"| Derating | {hw['battery_derating']} | temperature, self-discharge, cutoff |",
        "",
        "Replace these in `HardwareAssumptions` in `src/energy_model.py` with measured",
        "numbers from the hardware team, then rerun `make energy`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "./venv/bin/python -m src.energy_model",
        "```",
        "",
        "Full breakdown including per-state charge is in `artifacts/reports/energy_metrics.json`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    hw = HardwareAssumptions()
    results = compute(hw)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, args.markdown)

    print("Battery life by policy (placeholder hardware assumptions):")
    for name, entry in results["scenarios"].items():
        print(
            f"  {name:<16} {entry['transmissions_per_day']:>7.1f} tx/day  "
            f"{entry['battery_life_years']:>6.2f} yr  "
            f"{entry['battery_multiplier_vs_always_on']:>6.1f}x vs always-on"
        )
    print(f"\nWrote {args.json}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
