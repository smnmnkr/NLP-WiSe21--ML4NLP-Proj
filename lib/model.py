import torch
import torch.nn as nn

from lib.nn import MLP, GRU
from lib.util import Metric, flatten, get_device


class Model(nn.Module):

    #
    #
    #  -------- init -----------
    #
    def __init__(self, config: dict, embedding):
        super().__init__()

        # save config
        self.config = config
        self.metric = Metric()
        self.embedding = embedding

        self.emb_dropout = nn.Dropout(p=self.embedding.dropout, inplace=False)
        self.acf_out = nn.Softmax(dim=1)

        # RNN to calculate contextualized word embedding
        self.context = GRU(
            in_size=self.embedding.dimension,
            hid_size=self.config["rnn"]["hid_size"],
            depth=self.config["rnn"]["depth"],
            dropout=self.config["rnn"]["dropout"],
        )

        # MLP to calculate the POS tags
        self.score = MLP(
            in_size=self.config["rnn"]["hid_size"] * 2,
            hid_size=self.config["score"]["hid_size"],
            out_size=2,
            dropout=self.config["score"]["dropout"],
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, batch: list) -> list:

        # embed batch and apply dropout
        embed_batch: list = [self.emb_dropout(row) for row in self.embedding.forward_batch(batch)]

        # Contextualize embedding with BiLSTM
        pad_context, mask, hidden = self.context(embed_batch)

        # Calculate the score using the sum of all context prediction:
        # return [torch.sum(pred, dim=0, keepdim=True) for pred in unpad(self.score(pad_context), mask)]

        # Calculate the score using last hidden context state:
        score = self.score(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return torch.split(self.acf_out(score), 1)

    #
    #
    #  -------- predict -----------
    #
    def predict(
            self,
            batch: list,
    ) -> tuple:

        word_batch: list = []
        label_batch: list = []

        for row in batch:

            # append bert input: ['input_ids', 'attention_mask]
            if type(self.embedding).__name__ == "Bert":
                word_batch.append({'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']})

            # append regular input: 'text tokens'
            else:
                word_batch.append(row['text'])

            label_batch.append([row['label']])

        return self.forward(word_batch), label_batch

    #
    #
    #  -------- loss -----------
    #
    def loss(
            self,
            batch: list,
    ) -> nn.CrossEntropyLoss:

        predictions, target_ids = self.predict(batch)

        return nn.CrossEntropyLoss()(
            torch.cat(predictions),
            torch.LongTensor(flatten(target_ids)).to(get_device()),
        )

    #
    #
    #  -------- evaluate -----------
    #
    @torch.no_grad()
    def evaluate(
            self,
            batch: list,
            reset: bool = True,
            category: str = None,
    ) -> float:
        self.eval()

        if reset:
            self.metric.reset()

        predictions, target_ids = self.predict(batch)

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(predictions, target_ids):
            for (p, g) in zip(pred, gold):

                p_idx: int = torch.argmax(p).item()

                if p_idx == g:
                    self.metric.add_tp(p_idx)

                    for c in self.metric.get_classes():
                        if c != p_idx:
                            self.metric.add_tn(p_idx)

                if p_idx != g:
                    self.metric.add_fp(p_idx)
                    self.metric.add_fn(g)

        return self.metric.f_score(class_name=category)

    #  -------- save -----------
    #
    def save(self, path: str) -> None:

        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "metric": self.metric,
                "embedding": self.embedding
            },
            path + ".pickle",
        )

    #  -------- load -----------
    #
    @classmethod
    def load(cls, path: str) -> nn.Module:

        data = torch.load(path + ".pickle")

        model: nn.Module = cls(data["config"], data["embedding"]).to(get_device())
        model.load_state_dict(data["state_dict"])

        return model

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return sum(p.numel() for p in self.parameters())
