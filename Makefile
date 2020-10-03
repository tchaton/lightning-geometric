.PHONY: staticchecks

staticchecks:
	flake8 . --count --select=E9,F402,F6,F7,F5,F8,F9 --show-source --statistics
	mypy examples

local-test:
	coverage run -m pytest -n `python -c 'import multiprocessing as mp; print(mp.cpu_count())'` --verbose --capture=no --color=yes

workflow-test:
	coverage run -m pytest -n `python -c 'import multiprocessing as mp; print(mp.cpu_count())'` --color=yes