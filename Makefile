PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo .venv/Scripts/python.exe; fi)
PIP_EXTRA := --index-url https://download.pytorch.org/whl/cpu

.PHONY: help venv install dev test test-slow lint format type prefetch docker run-docker clean

help:
	@echo "make venv      create .venv"
	@echo "make install   install the package with UI and dev extras"
	@echo "make dev       run the Streamlit app locally"
	@echo "make test      run fast tests (no model downloads)"
	@echo "make test-slow run real-model smoke tests (downloads weights)"
	@echo "make lint      ruff check"
	@echo "make format    ruff format"
	@echo "make type      mypy"
	@echo "make prefetch  download model weights for local use"
	@echo "make docker    build the Docker image"
	@echo "make run-docker run the image on :7860"

venv:
	python -m venv .venv

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install $(PIP_EXTRA) torch torchvision
	$(PY) -m pip install -e ".[ui,dev]"

dev:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -m "not slow"

test-slow:
	VAS_RUN_SLOW=1 $(PY) -m pytest tests/slow -m slow

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

type:
	$(PY) -m mypy

prefetch:
	$(PY) -m vision_ai_studio.cli prefetch --seg ade --caption blip-base

docker:
	docker build -t vision-ai-studio:local .

run-docker:
	docker run --rm -p 7860:7860 --name vision-ai-studio vision-ai-studio:local

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
