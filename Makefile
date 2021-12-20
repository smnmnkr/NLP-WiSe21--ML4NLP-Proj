module := lib

run:
	@python3 -m ${module}

baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt