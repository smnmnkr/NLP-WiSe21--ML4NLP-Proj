from tqdm.auto import tqdm
from model_preparation import Model
from helper_functions import get_device
from data_preparation import get_dataloaders


# Parameters
TRAIN_DATA_PATH = "data/train.csv"
MODEL_NAME = "bert-base-cased"
BATCH_SIZE = 16
NUM_EPOCHS = 4


train_dataloader, validation_dataloader = get_dataloaders(TRAIN_DATA_PATH,  MODEL_NAME, batch_size = 16, create_validation_set = True)


model_class = Model(MODEL_NAME, 4, len(train_dataloader))
model, optimizer, lr_scheduler = model_class.get_model_optimizer_scheduler()
device = get_device()
model = model.to(device)

progress_bar = tqdm(range(model_class.get_num_training_steps()))

try:
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            model.zero_grad()
            parameters = {
                "input_ids" : batch[0].to(device),
                "attention_mask" :  batch[1].to(device), 
                "labels" : batch[2].to(device)
            }
            outputs = model(**parameters)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

except RuntimeError as e:
    print(e)