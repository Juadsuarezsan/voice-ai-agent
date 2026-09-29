.PHONY: install test lint format typecheck eval demo-cases wer serve notebook gitleaks check

PY ?= python

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

typecheck:
	mypy --strict src/

check: lint typecheck test

eval:
	$(PY) -m eval.run

demo-cases:
	$(PY) scripts/build_demo_cases.py

wer:
	$(PY) scripts/wer_benchmark.py --references data/eval/wer_fixture/references.txt --hypotheses data/eval/wer_fixture/hypotheses.txt

serve:
	uvicorn src.api.main:app --reload --port 8000

notebook:
	$(PY) -m nbclient.cli --execute notebooks/demo.ipynb --inplace

gitleaks:
	gitleaks detect --no-banner --redact --source .
