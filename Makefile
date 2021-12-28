module := challenge

#
# Run config examples:
config_path := config_examples

example_base:
	@python3 -m ${module} -C ${config_path}/train_base.json

example_fasttext:
	@python3 -m ${module} -C ${config_path}/train_fasttext.json

example_bert:
	@python3 -m ${module} -C ${config_path}/train_bert.json

# --- --- --- --- ---

#
# Run experiments:
experiment_fasttext:
	@python3 -m ${module} -C experiments/fasttext/config.json

# --- --- --- --- ---

#
# Run miscellaneous:
baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt