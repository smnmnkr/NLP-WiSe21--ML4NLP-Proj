module := lib

train_untrained:
	@python3 -m ${module} -C config/train_untrained.json

train_fasttext:
	@python3 -m ${module} -C config/train_fasttext.json

baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt