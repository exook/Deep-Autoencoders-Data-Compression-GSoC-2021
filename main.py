from data_preprocessing import preprocess
from autoencoders import standard_AE
from create_plots import plot_initial_data, plot_test_pred_data
import pandas as pd
from data_loader import load_cms_data

#cms_data_df = load_cms_data(filename="open_cms_data.root")
data_df = pd.read_csv('27D_openCMS_data.csv')

# Plot the original data
plot_initial_data(data_df)

# Preprocessing
data_df, train_data, test_data = preprocess(data_df)
# Plot the normalized data
plot_initial_data(data_df)

# Run the Autoencoder and obtain the reconstructed data
reconstructed_data = standard_AE.train(train_data=train_data, test_data=test_data)
# Plot the original along with the reconstructed data
plot_test_pred_data(test_data, reconstructed_data)
