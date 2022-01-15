from helper_functions import get_device
from data_preparation import get_dataloaders
from datasets import load_metric
import torch
from model_preparation import set_seed

def evaluate_on_test_data(model, TEST_DATA_PATH, MODEL_NAME, BATCH_SIZE, SEED=42):

    set_seed(SEED)

    test_dataloader = get_dataloaders(TEST_DATA_PATH, 
                                            MODEL_NAME, 
                                            BATCH_SIZE, 
                                            create_validation_set = False)

    device = get_device()
    model = model.to(device)

    testing_stats = []

    try:
        total_test_loss = 0
        metric = load_metric("f1")

        model.eval()

        for batch in test_dataloader:

            parameters = {
                "input_ids" : batch[0].to(device),
                "attention_mask" :  batch[1].to(device), 
                "labels" : batch[2].to(device)
            }
            with torch.no_grad():
                outputs = model(**parameters)
            
            logits = outputs.logits
            loss = outputs.loss
            total_test_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=parameters["labels"])

        testing_stats.append({
            "test_loss": total_test_loss/len(test_dataloader),
            "test_f1_score": metric.compute()
        })

        print(f"\nAvg test loss:  {testing_stats[0]['test_loss']}")
        print(f"F1 test score:  {testing_stats[0]['test_f1_score']}\n")

    except RuntimeError as e:
        print(e)

    return testing_stats