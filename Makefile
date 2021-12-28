module := challenge

#
# Run config examples:
examples_path := config_examples

example_base:
	@python3 -m ${module} -C ${examples_path}/base.json

example_fasttext:
	@python3 -m ${module} -C ${examples_path}/fasttext.json

example_bert:
	@python3 -m ${module} -C ${examples_path}/bert.json

# --- --- --- --- ---

#
# Run experiments:
experiment_fasttext:
	@python3 -m ${module} -C experiments/fasttext/config.json

experiment_bert:
	@python3 -m ${module} -C experiments/bert/config.json

# --- --- --- --- ---

#
# Run miscellaneous:
baseline:
	@python3 ./scripts/baseline.py

install:
	@python3 -m pip install -r requirements.txt