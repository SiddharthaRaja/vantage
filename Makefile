.PHONY: help install dev demo test lint fmt clean sweep report docker-build serve-vllm

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## install the package
	pip install -e .

dev:  ## install with dev extras
	pip install -e ".[dev]"

demo:  ## run the whole pipeline on the simulated backend (no GPU, no API key)
	vantage demo

test:  ## run the test suite
	pytest -q

lint:  ## ruff check
	ruff check src tests scripts finetune

fmt:  ## ruff format + autofix
	ruff format src tests scripts finetune && ruff check --fix src tests scripts finetune

sweep:  ## run a sweep: make sweep CONFIG=configs/sweeps/quantization.yaml
	vantage sweep -c $(or $(CONFIG),configs/sweeps/demo.yaml)

report:  ## rebuild a report: make report RUN=results/demo/20260101-120000
	vantage report $(RUN)

workload:  ## regenerate the bundled sample workload
	python scripts/make_sample_workload.py

docker-build:  ## build the harness image
	docker build -t vantage:latest .

serve-vllm:  ## start a local vLLM server for the default model
	vllm serve Qwen/Qwen3-4B-Instruct --port 8000 --max-model-len 8192 \
		--gpu-memory-utilization 0.90 --enable-prefix-caching

clean:  ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
