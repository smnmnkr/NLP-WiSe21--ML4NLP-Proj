module := lib

run:
	@python3 -m ${module}

baseline:
	@python3 ./scripts/baseline.py