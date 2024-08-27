# -*- coding: utf-8 -*-
"""Pred function FISA A2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EZ53TFhwaGjXT35OK_Loc3zWKylx8YP4
"""

def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import torch.nn as nn
    import joblib
    data = data.fillna(method='ffill')
    data_valid_close = data.reset_index()['Close']
    scaler = joblib.load('./FISA A2/scaler.gz')
    data_valid_close = scaler.transform(np.array(data_valid_close).reshape(-1,1))
    X = np.expand_dims(data_valid_close,0)
    X = torch.from_numpy(X).float()
    class LSTMmodel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True, bidirectional = True)
            self.linear1 = nn.Linear(32, 16)
            self.linear2 = nn.Linear(16, 2)
        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear1(x[:,-1,:])
            x = self.linear2(x)
            return x
    model = LSTMmodel()
    model.load_state_dict(torch.load("./best_model_GRU_bidirectional_2linear_layer_no_skip_connection"))
    model.eval()
    with torch.no_grad():
      Y_pred = model(X)
    Y_pred_rescaled = np.squeeze(scaler.inverse_transform(Y_pred.numpy().reshape(-1,1)))
    #print(Y_pred_rescaled.tolist())

    return Y_pred_rescaled.tolist()