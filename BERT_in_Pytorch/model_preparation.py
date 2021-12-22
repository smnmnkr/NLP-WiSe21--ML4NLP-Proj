from transformers import    AutoModelForSequenceClassification,\
                            AdamW,\
                            get_scheduler


class Model():
    def __init__(self, model_name, num_epochs, length_dataloader):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-cased", # Use the 12-layer BERT model, with a cased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        self.optimizer = AdamW(self.model.parameters(),lr = 2e-5)
        self.num_epochs = num_epochs
        self.num_training_steps = self.num_epochs * length_dataloader
        self.lr_scheduler = get_scheduler(
                                    "linear",
                                    optimizer=self.optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=self.num_training_steps
                                    )
    
    def get_model_optimizer_scheduler(self):
        return self.model, self.optimizer, self.lr_scheduler

    def get_num_training_steps(self):
        return self.num_training_steps


