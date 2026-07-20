# EmberWatch

EmberWatch is a wireless temperature sensor node I'm building for the IEEE HART HardwAIre Challenge. It's coin sized, battery powered, and clamps onto grid adjacent electrical equipment in wildfire prone regions. The idea is to catch overheating on that equipment with on device AI before it turns into an ignition event.

## Why this matters

Electrical grid equipment is a documented wildfire ignition source in the Pacific Northwest. Most of the monitoring that exists today is manual or expensive to deploy at scale. A node like this is cheap enough and small enough that you could put a lot of them out on equipment that currently gets no monitoring at all, and it's smart enough to decide on its own whether what it's seeing is worth telling anyone about.

## What the AI actually does

The core model is an echo state network that runs directly on the Cortex-M4 core of the STM32WL55 microcontroller. It reads the thermal time series coming off two NTC thermistors (one on the monitored equipment, one for ambient) and classifies each window as normal, warming, or anomaly. That classification gates the LoRa radio. If the state hasn't changed and nothing looks anomalous, the node stays quiet and saves power. If something changes, it wakes the radio and sends a packet.

That single on device decision is what the whole project's value proposition rests on. Fewer unnecessary transmissions means longer battery life, which means the node can go longer between maintenance visits and can be deployed in places that are harder to reach.

## My role

I own the AI and data side. That means the dataset architecture, the feature engineering pipeline, the baseline detectors we compare the ESN against, and the evaluation harness that scores any model (baseline or ESN) on the same metrics, so results are apples to apples no matter who trained what.

## Dataset architecture

There isn't a real fielded EmberWatch sensor yet, so the dataset is assembled from existing public data that stands in for the real thing until we have hardware in the field.

Ambient temperature comes from several real weather stations in Oregon and Northern California, chosen because they sit in or near wildfire prone terrain: airport weather stations around the Columbia Gorge and Portland metro area, plus RAWS (Remote Automatic Weather Station) data from sites closer to the actual fire risk zones. Each station is kept as its own separate dataset rather than merged into one blended ambient signal, since merging would hide differences in elevation, microclimate, and data quality between sites.

The asset (equipment) temperature channel uses the NAB machine_temperature_system_failure dataset, a real, publicly labeled time series of a machine sensor that actually failed. It's not a perfect stand in for electrical equipment, but it's real sensor data with real labeled anomalies, which is more useful for a working baseline than anything synthetic would be at this stage.

There's also a transformer temperature dataset (ETT, Electricity Transformer Temperature) in the mix, used to pull realistic noise characteristics rather than as a direct signal source.

Raw data files are not included in this repo because of their size. Everything under `data/raw/` and `data/processed/` is gitignored. The scripts in `src/` are written to reproduce the full pipeline from the original public sources, so anyone cloning this repo can regenerate everything locally.

## How to run this

Clone the repo, then from the project root:

```
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Drop the required raw files into `data/raw/` (see the docstring at the top of `src/ingest.py` for the exact filenames expected). Then run:

```
./venv/bin/python src/ingest.py
```

This downloads the NAB files automatically if they're missing, processes each ambient weather station independently, and writes the processed outputs into `data/processed/`.
