import torch
import torch.nn as nn
import torch.nn.functional as F


class CropYieldModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        self._input_size = input_size
        self._output_size = output_size
        # self.linear = nn.Linear(self._input_size, self._output_size)

    def forward(self, xb):
        return self.network(xb)

    def training_step(self, batch):
        inputs, targets = batch

        # Generating predictions
        out = self(inputs)
        # calculate loss
        loss = F.l1_loss(out, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.l1_loss(out, targets)
        return {"val_loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {"val_loss": epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch[{}], val_loss: {:.4f}".format(epoch + 1, result["val_loss"]))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)

    return history


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input: ", input)
    print("Target: ", target)
    print("Prediction: ", prediction)


from sklearn.metrics import r2_score


def calc_r2_score(model, X_scaled, y_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.Tensor(X_scaled)
        predictions = model(inputs)

    y_true = y_test
    y_pred = predictions.numpy().ravel()

    nn_r2 = r2_score(y_true, y_pred)
    print(f"r2_score: {nn_r2:.4f}")
