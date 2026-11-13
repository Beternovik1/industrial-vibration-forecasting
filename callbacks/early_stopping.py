import copy


class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, model, validation_metric):
        if validation_metric < self.best_metric - self.min_delta:
            self.best_metric = validation_metric
            self.counter = 0
            # Freeze the best weights before later optimizer steps change them.
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
