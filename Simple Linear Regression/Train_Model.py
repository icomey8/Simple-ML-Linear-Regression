import torch
import matplotlib.pyplot as plt
from Build_Model import X_test, X_train, y_train, y_test



### TRAINING OUR MODEL
torch.manual_seed(100)
epochs = 300
num_epochs = []
training_losses = []
test_losses = []

def training_loop(model_1, loss_function, optimizer, see_training):
    for epoch in range(epochs):
        # TRAIN
        model_1.train()
        y_predictions = model_1(X_train)
        loss = loss_function(y_predictions, y_train)  # comparing our predictions with the actual testing values
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TEST
        if epoch % 20 == 0 and see_training == "Y":
            model_1.eval()
            with torch.inference_mode():
                test_predictions = model_1(X_test)
                test_loss = loss_function(test_predictions, y_test.type(torch.float))    
                num_epochs.append(epoch)
                training_losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())
                print(f"EPOCH NUMBER: {epoch} | MAE TRAINING LOSS: {loss:.3f} | MAE TESTING LOSS: {test_loss:.3f}")
        else:
            model_1.eval()
            with torch.inference_mode():
                test_predictions = model_1(X_test)
                test_loss = loss_function(test_predictions, y_test.type((torch.float)))    
                num_epochs.append(epoch)
                training_losses.append(loss.detach().numpy())
                test_losses.append(test_loss.detach().numpy())



def plot_training():
    plt.plot(num_epochs, training_losses, label="Train loss")
    plt.plot(num_epochs, test_losses, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show()

