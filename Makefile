.PHONY: setup test lint features notebooks card dashboard all clean

setup:
	uv sync
	uv run python scripts/install_kernel.py

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

features:
	uv run python scripts/build_features.py

# Execute, then clear, as two passes. Combining them does not work: nbconvert runs
# ClearOutputPreprocessor before ExecutePreprocessor, so a single invocation clears
# the outputs and then executes straight over them. Committed notebooks carry no
# outputs; the results live in metrics/ and reports/.
notebooks:
	JUPYTER_PATH=$(CURDIR)/.jupyter uv run jupyter nbconvert --to notebook \
		--execute --inplace --ExecutePreprocessor.kernel_name=adil notebooks/*.ipynb
	uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

card:
	uv run python scripts/render_card.py

dashboard:
	uv run python scripts/build_dashboard.py

all: lint test features notebooks card dashboard

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
