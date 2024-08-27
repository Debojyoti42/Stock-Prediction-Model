Our predict_func() takes a pandas dataframe as input and returns a list of length 2 which are the predictions of the next two days.

We take only the close values from previous 50days to predict the close price of next two days.

To make our model robust, we use a sliding window on the the given training csv  file to generate a set of (sequence,label) pairs. Each sequence contains previous 50 days closing prices and the label contains the next twodays closing prices.

Then we randomly choose 80% data from the dataset for training and keep 20% data for validation.

We use a Bidirectional GRU followed by two linear layers to predict the prices. 

The model weights are saved with the name "best_model_GRU_bidirectional_2linear_layer_no_skip_connection"

We use a minmax scaler for prosessing the data before training as RNN models perform poorly with high value of data because of exploding gradients.
The minmax scalar can be found with the name "scaler.gz".This is also used in rescaling the output of the neural network.

Our predict_func() uses both "best_model_GRU_bidirectional_2linear_layer_no_skip_connection" weights and "scaler.gz" to predict the next two days of value.

*I.File 22104402_Debojyoti.py is edited version of sample.py in which I edited the pred_func() only.

*II.File pred_function.py contains only the prediction function.

Preprocessing-Training Script.ipynb is provided to show the model I trained with some curves. 