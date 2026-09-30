PYTHON ?= python
GPUS ?= 0

.PHONY: table1 figures analyze calibration env smoke clean

table1:
	$(PYTHON) scripts/reproduce_table1.py --gpus $(GPUS) --verify

figures:
	$(PYTHON) scripts/make_figure1.py
	$(PYTHON) scripts/make_figure2.py

analyze:
	$(PYTHON) scripts/analyze_results.py --root results --outdir analysis

calibration:
	$(PYTHON) scripts/launch_experiments.py --gpu-ids $(GPUS) --seeds 0 1 2 3 4 --phases calibration --calibration-sizes 128 512 2048 4096 --root results
	$(PYTHON) scripts/analyze_results.py --root results --outdir analysis
	$(PYTHON) scripts/make_calibration_occupancy.py --raw-dir results/calibration --out-prefix figures/calibration_occupancy

env:
	bash scripts/capture_environment.sh

smoke:
	$(PYTHON) tests/smoke_test.py

clean:
	rm -rf results/table1 analysis figures/*.pdf figures/*.png figures/*.csv
