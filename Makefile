init:
	git submodule update --init --recursive
	pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

syntax:
	pip install pylint
	python3 -m pylint --exit-zero --rcfile=.pylintrc $(shell git ls-files '*.py')

syntax_strategy:
	pip install pylint
	python3 -m pylint --exit-zero --rcfile=.pylintrc $(shell find bridge/strategy -name "*.py")

auto_format:
	@dpkg -s pre-commit >/dev/null 2>&1 || pip install pre-commit
	pre-commit install
	pre-commit run -a

PYTHON ?= venv/bin/python

run:
	$(PYTHON) main.py

run-real:
	IS_SIMULATOR_USED=False DEBUG_MODE=False $(PYTHON) main.py

run-sim:
	IS_SIMULATOR_USED=True DEBUG_MODE=False $(PYTHON) main.py

run-sim-debug:
	IS_SIMULATOR_USED=True DEBUG_MODE=True DEBUG_GAME_STATE=RUN $(PYTHON) main.py

.PHONY: init test syntax run run-real run-sim run-sim-debug
