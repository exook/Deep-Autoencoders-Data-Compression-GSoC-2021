from data_processing import preprocess_28D
import autoencoders.autoencoder as ae
import autoencoders.variational_autoencoder as vae
import autoencoders.sparse_autoencoder as sae
from create_plots import plot_initial_data, plot_test_pred_data, correlation_plot
from evaluate import evaluate_model
import pandas as pd
import argparse
from data_loader import load_cms_data
import sys
import pickle

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # constructing argument parsers
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--epochs', type=int, default=50,
                    help='number of epochs to train our autoencoder for')

    ap.add_argument('-v', '--num_variables', type=int, default=24,
                    help='Number of variables we want to compress (either 19 or 24)')

    ap.add_argument('-cn', '--custom_norm', type=bool, default=False,
                    help='Whether to normalize all variables with min_max scaler or also use custom normalization for 4-momentum')

    ap.add_argument('-vae', '--use_vae', type=bool, default=True,
                    help='Whether to use Variational AE')
    ap.add_argument('-sae', '--use_sae', type=bool, default=False,
                    help='Whether to use Sparse AE')
    ap.add_argument('-l1', '--l1', type=bool, default=True,
                    help='Whether to use L1 loss or KL-divergence in the Sparse AE')
    ap.add_argument('-p', '--plot', type=bool, default=False,
                    help='Whether to make plots')
    ap.add_argument('-mode', '--mode')
                    
    args = vars(ap.parse_args()) 
    epochs = args['epochs']
    use_vae = args['use_vae']
    use_sae = args['use_sae']
    custom_norm = args['custom_norm']
    num_of_variables = args['num_variables']
    create_plots = args['plot']
    l1 = args['l1']
    mode = args["mode"]
    
    if mode == "train":
        reg_param = 0.001
        # sparsity parameter for KL loss in SAE
        RHO = 0.05
        # learning rate
        lr = 0.001

        #cms_data_df = load_cms_data(filename="open_cms_data.root")
        #data_df = pd.read_csv('27D_openCMS_data.csv')

        # Preprocess data
        #data_df, train_data, test_data, scaler = preprocess_28D(data_df=data_df, num_variables=num_of_variables, custom_norm=custom_norm)
        
        data_df = pd.read_csv('george.csv')
        data_df.drop(columns=data_df.columns[0], axis=1, inplace=True)
        train_data, test_data = train_test_split(data_df, test_size=0.15, random_state=1)

        print("\nNumber of input variables",len(list(data_df.columns)))
        print("List of input variables",list(data_df.columns))
        print("\n")

        # Run the Sparse Autoencoder and obtain the reconstructed data
        test_data, reconstructed_data = sae.train(variables=num_of_variables, train_data=train_data,
                                                      test_data=test_data, learning_rate=lr, reg_param=reg_param, epochs=epochs, RHO=RHO, l1=l1)

        with open("test_data.pickle", "wb") as handle:
            pickle.dump(test_data, handle)      
        with open("reconstructed_data.pickle", "wb") as handle:
            pickle.dump(reconstructed_data, handle)                                                           
    elif mode == "plot":
        with open("test_data.pickle", 'rb') as handle:
            test_data = pickle.load(handle)
        with open("reconstructed_data.pickle", 'rb') as handle:
            reconstructed_data = pickle.load(handle)
            
        # Plot the reconstructed along with the initial data
        plot_test_pred_data(test_data, reconstructed_data, num_variables=num_of_variables, sae=True)
        # Evaluate the reconstructions of the network based on various metrics
        #evaluate_model(y_true=test_data, y_predicted=reconstructed_data)
