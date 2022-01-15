from train import train_model_on_train_data
from evaluate import evaluate_on_test_data


# Parameters
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/eval.csv"
MODEL_NAME = "bert-base-cased"
BATCH_SIZE = 32
NUM_EPOCHS = 2
SEED = 42

model, training_stats = train_model_on_train_data(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, SEED)

testing_stats = evaluate_on_test_data(model, TEST_DATA_PATH, MODEL_NAME, BATCH_SIZE, SEED)

print("\nTraining results: ", training_stats)
print("\nTesting results: ", testing_stats)