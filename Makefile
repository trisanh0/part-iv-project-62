# Locate Python version
PYTHON := $(shell which python3.11 || which python3.12)

create_environment:
ifndef PYTHON
	$(error "TEMPO requires Python 3.11 or 3.12.")
endif
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	./.venv/bin/pip install -e .