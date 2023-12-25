import torch
import time
from Build_Model import plot_input, X_test


### MAKE PREDICTIONS USING MODEL
def make_predictions(model, loaded=False):
    with torch.inference_mode():
        model.eval()
        global y_predict
        y_predict = model(X_test)
        if loaded == True:
            global loaded_model_preds
            loaded_model_preds = model(X_test)
            if (torch.equal(y_predict, loaded_model_preds)):
                    print("\nthe model has succesfully loaded.")
                    time.sleep(1)
    plot_input(predictions=y_predict)

