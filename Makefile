.PHONY: staticchecks

staticchecks:
	flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics
	mypy examples

local-test:
	coverage run -m pytest -n `sysctl -n hw.ncpu` --verbose --capture=no --color=yes

workflow-test:
	coverage run -m pytest -n `sysctl -n hw.ncpu` --color=yes