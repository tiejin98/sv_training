PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test cli-help inspect check build dist-check clean

install:
	$(PIP) install -e ".[vllm]"

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

cli-help:
	$(PYTHON) -m sv_training --help

inspect:
	bash quick_start/code/00_inspect_data.sh

check: test cli-help inspect

build:
	$(PYTHON) -m build

dist-check: build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
