init:
	git submodule update --init --recursive
	pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -v

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

run:
	python3 main.py

.PHONY: init test syntax
