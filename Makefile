.PHONY: install train test lint app clean

install:            ## install the package, the web app and the dev tools
	pip install -e ".[app,dev]"

train:              ## tune, compare and save the models
	spam-detector train

test:               ## run the unit tests
	pytest

lint:               ## style check
	ruff check .

app:                ## start the web interface on http://localhost:8501
	streamlit run app/streamlit_app.py

clean:              ## remove generated caches
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
