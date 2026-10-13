PYTHON ?= python3

.PHONY: verify test syntax

verify:
	$(PYTHON) scripts/project_healthcheck.py

test:
	$(PYTHON) -m unittest discover -s tests -v

syntax:
	$(PYTHON) -m compileall -q src scripts tests run_expid.py run_param_tuner.py prediction.py
