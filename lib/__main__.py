import torch
from transformers import AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

from train import train_model_on_train_data, train_model_on_full_train_data
from evaluate import evaluate_on_test_data


# Parameters
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/eval.csv"
MODEL_NAME = "microsoft/deberta-base"
BATCH_SIZE = 16
NUM_EPOCHS = 1
SEED = 42
TRAIN_MODEL_ON_FULL_TRAINING_DATA = True
SAVE_NEW_MODEL = True
USE_PRETRAINED_MODEL = True


if USE_PRETRAINED_MODEL:
    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
        )
    model.load_state_dict(torch.load("pretrained_model/model_state_dict.pt"))

else:
    if TRAIN_MODEL_ON_FULL_TRAINING_DATA:
        model, training_stats = train_model_on_full_train_data(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, SEED)
    else:
        model, training_stats = train_model_on_train_data(TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, SEED)
    print("\nTraining results: ", training_stats)

    if SAVE_NEW_MODEL:
        torch.save(model.state_dict(), "pretrained_model/model_state_dict.pt")
        torch.save(model, "pretrained_model/entire_model.pt")



testing_stats = evaluate_on_test_data(model, TEST_DATA_PATH, MODEL_NAME, BATCH_SIZE, SEED)
print("\nTesting results: ", testing_stats)

