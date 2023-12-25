import torch
import time
from pathlib import Path
from Train_Model import test_losses, training_losses
from Build_Model import LinearRegressionModel
from Predictions import make_predictions


### SAVE AND LOAD (only if model is accurate)
def save_and_load(model_1):
    if (test_losses[-1]) < 0.15 and (training_losses[-1] < 0.5):   
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        MODEL_NAME = "01_pytorch_workflow_model_1.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME   

        print(f"\n\nThis model is accurate, so it will be saved.\n Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the model's learned parameters
                f=MODEL_SAVE_PATH)
        time.sleep(1.5)

        load_or_no = input("\n\nWould you like to load the model that was just saved? Press Y if so.\n\n")
        if load_or_no == "Y":
            global loaded_model_2
            loaded_model_2 = LinearRegressionModel()
            loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
            loaded_model_2.eval()
            with torch.inference_mode():
                make_predictions(loaded_model_2, loaded=True)
                

