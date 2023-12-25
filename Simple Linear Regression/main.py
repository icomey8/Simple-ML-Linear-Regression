import time
import Build_Model, Train_Model, Predictions, Save_and_Load


### MAIN
input("\nHello, this is a program that generates predictions for randomly created datasets using a Linear Regression Model. \nEnter any key to begin.\n\n")

model_1 = Build_Model.LinearRegressionModel()
loss_function = Build_Model.nn.L1Loss()
optimizer = Build_Model.torch.optim.SGD(params=model_1.parameters(), lr=0.01)

see_training = input("\nWould you like to see data from the model's training (epoch count, MAE Training Loss, MAE Testing Loss? Press Y or N. \n\n")
Train_Model.training_loop(model_1, loss_function, optimizer, see_training)

time.sleep(0.75)
print("\n\nNote: The ML Model's results will be displayed on a graph.  Click the exit button on the graph when ready to move on.\n\n")
time.sleep(3)
Predictions.make_predictions(model_1)

Save_and_Load.save_and_load(model_1)