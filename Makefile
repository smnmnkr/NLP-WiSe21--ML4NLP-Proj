module := lib
config := config.json

run:
	@python3 -m ${module} -C ${config}

baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt