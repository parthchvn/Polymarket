# Common operations. `make help` lists targets.

.PHONY: help test lint check collect overnight grind clean-caches

help:
	@grep -E "^[a-z-]+:.*## " Makefile | awk -F ":.*## " \x27{printf "  %-14s %s\n", $$1, $$2}\x27

test: ## full test suite
	python -m pytest -q

lint: ## ruff over src, tests, scripts
	ruff check src tests scripts

check: lint test ## lint then tests (what CI runs)

collect: ## forward collector (edit MARKETS/QUERIES in the file)
	./scripts/collect.sh

overnight: ## snapshot-merge worker, 2h cycles
	./scripts/overnight.sh

grind: ## article bodies + LLM claim extraction loop
	./scripts/grind-news.sh

clean-caches: ## remove pytest/ruff caches
	rm -rf .pytest_cache .ruff_cache
