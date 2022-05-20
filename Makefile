# Makefile with some simple commands to make developer's life easier


install-requirements: install-build-essential
	pip install -r requirements.txt

dev/install-requirements: install-requirements
	pip install -r requirements.dev.txt

install-build-essential:
	sudo apt-get update
	sudo apt-get install build-essential

update-setuptools:
	pip install --upgrade setuptools wheel

test-unit:
	pytest tests
	@echo 'unit tests OK'

black-check:
	black --diff --line-length 120 cobra/

typecheck:
	mypy cobra --allow-redefinition --allow-untyped-globals --ignore-missing-imports
	@echo 'typecheck OK'

codestyle:
	pycodestyle cobra
	@echo 'codestyle OK'

docstyle:
	pydocstyle cobra
	@echo 'docstyle OK'

code-qa: typecheck codestyle docstyle
