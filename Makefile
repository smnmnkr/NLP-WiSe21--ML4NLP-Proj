module := lib
config_path := config_examples

train_base:
	@python3 -m ${module} -C ${config_path}/train_base.json

train_fasttext:
	@python3 -m ${module} -C ${config_path}/train_fasttext.json

train_bert:
	@python3 -m ${module} -C ${config_path}/train_bert.json

baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt