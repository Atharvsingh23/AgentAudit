.PHONY: install test lint bench ablation corpus results demo clean all

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

corpus:
	agentaudit corpus export --n 100
	agentaudit corpus verify

bench:
	agentaudit bench --n 100

ablation:
	@echo "=== corroboration ON ==="
	@agentaudit bench --n 100 --ablation
	@echo "=== corroboration OFF ==="
	@agentaudit bench --n 100 --ablation --no-corroborate

results:
	agentaudit bench --n 100 --out results/standard.json
	agentaudit bench --n 100 --ablation --out results/ablation_on.json
	agentaudit bench --n 100 --ablation --no-corroborate --out results/ablation_off.json

demo:
	python examples/silent_failure_demo.py

all: install corpus test lint bench

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
