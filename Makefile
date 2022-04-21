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

lint:
	pylint cobra
	@echo 'lint OK'

lint-minimal:
	pylint E cobra
	@echo 'lint minimal OK'

typecheck:
	mypy cobra
	@echo 'typecheck OK'

codestyle:
	pycodestyle cobra
	@echo 'codestyle OK'

docstyle:
	pydocstyle cobra
	@echo 'docstyle OK'

code-qa: typecheck codestyle docstyle lint-minimal
