class EarlyStopping:

    def __init__(
            self,
            delta: float = 0.02,
            patience: int = 20):

        self.delta = delta
        self.patience = patience

        self.counter: int = 0
        self.smallest_loss: float = float("inf")

        self.should_save: bool = False
        self.should_stop: bool = False

    def step(self, val_loss: float) -> None:

        # new loss is smaller then smallest recorded
        if val_loss < self.smallest_loss:

            self.counter = 0
            self.smallest_loss = val_loss
            self.should_save = True

        # new loss is smaller then smallest plus delta deviation
        elif val_loss < self.smallest_loss + self.delta:
            self.should_save = False

        # new loss is larger then smallest plus delta deviation
        else:
            self.counter += 1
            self.should_save = False

            if self.counter == self.patience:
                self.should_stop = True
