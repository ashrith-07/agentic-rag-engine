.PHONY: dev qdrant redis test ingest eval generate-dataset lint typecheck clean push deploy help

help:
	@echo "agentic-rag-engine"
	@echo "────────────────────────────────────────"
	@echo "make dev              Start all services (build first)"
	@echo "make qdrant           Start Qdrant only (local dev)"
	@echo "make redis            Start Redis only (local dev)"
	@echo "make test             Run pytest with coverage"
	@echo "make ingest PDF=path  Ingest a PDF into the pipeline"
	@echo "make eval             Run full benchmark suite"
	@echo "make generate-dataset Generate ground truth test dataset"
	@echo "make lint             Ruff lint check"
	@echo "make typecheck        Mypy type check"
	@echo "make clean            Stop all containers + remove volumes"
	@echo "make push             Push to GitHub + HF Spaces (triggers rebuild)"

dev:
	docker compose up --build

qdrant:
	docker compose up qdrant -d
	@echo "Qdrant → http://localhost:6333/dashboard"

redis:
	docker compose up redis -d
	@echo "Redis → localhost:6379"

test:
	.venv/bin/pytest tests/ -v \
		--cov=src/ingestion \
		--cov=src/retrieval \
		--cov=src/reranking \
		--cov=src/llm \
		--cov=src/pipeline.py \
		--cov=src/evaluation/retrieval_metrics.py \
		--cov=src/utils \
		--cov-report=term-missing

ingest:
	@test -n "$(PDF)" || (echo "Usage: make ingest PDF=./data/raw/doc.pdf" && exit 1)
	python -m src.pipeline ingest --pdf $(PDF)

eval:
	python -m src.evaluation.benchmark_runner

generate-dataset:
	python -m src.evaluation.test_dataset_generator

lint:
	.venv/bin/ruff check src/ tests/

typecheck:
	.venv/bin/mypy src/

clean:
	docker compose down -v
	@echo "Containers stopped and volumes removed."

## Push to GitHub AND HF Spaces in one command
push:
	git push origin main
	git push space main
	@echo "✅ Pushed to GitHub + HF Spaces — rebuild triggered."

deploy: push
