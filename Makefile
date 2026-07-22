.PHONY: test smoke-esn train-esn baselines loso receiver-demo dashboard

PYTHON := ./venv/bin/python

test:
	$(PYTHON) -m py_compile src/*.py
	$(PYTHON) -m unittest discover -s tests -v
	mkdir -p build
	cc -std=c11 -Wall -Wextra -Werror -Ifirmware/include firmware/src/emberwatch_protocol.c tests/c/test_protocol.c -o build/test_protocol
	./build/test_protocol
	cc -std=c11 -Wall -Wextra -Werror -Ifirmware/include firmware/src/emberwatch_features.c firmware/src/emberwatch_policy.c tests/c/test_features_policy.c -lm -o build/test_features_policy
	./build/test_features_policy

smoke-esn:
	$(PYTHON) -m src.train_esn --max-rows-per-station 12000 --reservoir-size 24

train-esn:
	$(PYTHON) -m src.train_esn --max-rows-per-station 80000 --reservoir-size 48

baselines:
	$(PYTHON) -m src.baselines_v2 --max-rows-per-station 80000

loso:
	$(PYTHON) -m src.loso_experiment --max-rows-per-station 30000 --reservoir-size 32

receiver-demo:
	$(PYTHON) -m src.simulate_receiver --count 20 --interval 0.05 | $(PYTHON) -m src.receiver --output artifacts/telemetry/demo.csv

dashboard:
	$(PYTHON) -m src.dashboard --log artifacts/telemetry/demo.csv
