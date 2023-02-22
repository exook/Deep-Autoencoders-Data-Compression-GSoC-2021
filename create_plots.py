import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="white")


def plot_initial_data(input_data, num_variables, normalized=False):
    input_data = input_data.sort_values(by=['ak5PFJets_pt_'])

    if normalized:
        save_dir = "D:\Desktop\GSoC-ATLAS\preprocessed_data_plots"
    else:
        save_dir = "D:\Desktop\GSoC-ATLAS\initial_data_plots"

    prefix = 'ak5PFJets_'

    if num_variables == 24:
        save_dir = "D:\Desktop\preprocessed_data_plots\d24"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy',
                         'mElectronEnergy', 'mMuonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mMuonMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mChargedEmEnergy',
                         'mChargedMuEnergy', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

        branches = [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_',
                    prefix + 'mJetArea', prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy',
                    prefix + 'mPhotonEnergy',
                    prefix + 'mElectronEnergy', prefix + 'mMuonEnergy', prefix + 'mHFHadronEnergy',
                    prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                    prefix + 'mNeutralHadronMultiplicity',
                    prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                    prefix + 'mMuonMultiplicity',
                    prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity', prefix + 'mChargedEmEnergy',
                    prefix + 'mChargedMuEnergy', prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                    prefix + 'mNeutralMultiplicity']
    else:
        save_dir = "D:\Desktop\GSoC-ATLAS\preprocessed_data_plots\d19"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

        branches = [prefix + 'pt_', prefix + 'eta_', prefix + 'phi_', prefix + 'mass_',
                    prefix + 'mJetArea', prefix + 'mChargedHadronEnergy', prefix + 'mNeutralHadronEnergy',
                    prefix + 'mPhotonEnergy', prefix + 'mHFHadronEnergy',
                    prefix + 'mHFEMEnergy', prefix + 'mChargedHadronMultiplicity',
                    prefix + 'mNeutralHadronMultiplicity',
                    prefix + 'mPhotonMultiplicity', prefix + 'mElectronMultiplicity',
                    prefix + 'mHFHadronMultiplicity', prefix + 'mHFEMMultiplicity',
                    prefix + 'mNeutralEmEnergy', prefix + 'mChargedMultiplicity',
                    prefix + 'mNeutralMultiplicity']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_bins = 100
    save = True  # Option to save figure

    for kk in range(0, num_variables):
        if branches[kk] == prefix + 'pt_' or branches[kk] == prefix + 'mass_':
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=n_bins, log=True)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        elif branches[kk] == prefix + 'phi_':
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=50)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        else:
            n_hist_data, bin_edges, _ = plt.hist(input_data[branches[kk]], color='orange', label='Input', alpha=1,
                                                 bins=n_bins)
            plt.xlabel(xlabel=variable_list[kk])
            plt.ylabel('# of jets')
            plt.suptitle(variable_list[kk])
            if save:
                plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        plt.figure()
        plt.show()


def plot_test_pred_data(test_data, predicted_data, num_variables, vae=False, sae=False):
    if num_variables == 24:
        if sae:
            save_dir = "D:\Desktop\GSoC-ATLAS\SAE_plots\d24"
        elif vae:
            save_dir = "D:\Desktop\GSoC-ATLAS\VAE_plots\d24"
        else:
            save_dir = "D:\Desktop\GSoC-ATLAS\AE_plots\d24"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy',
                         'mElectronEnergy', 'mMuonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mMuonMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mChargedEmEnergy',
                         'mChargedMuEnergy', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']
    else:
        if sae:
            save_dir = "D:\Desktop\GSoC-ATLAS\SAE_plots\d19"
        elif vae:
            save_dir = "D:\Desktop\GSoC-ATLAS\VAE_plots\d19"
        else:
            save_dir = "D:\Desktop\GSoC-ATLAS\AE_plots\d19"

        variable_list = ['pt_', 'eta_', 'phi_', 'mass_', 'mJetArea',
                         'mChargedHadronEnergy', 'mNeutralHadronEnergy',
                         'mPhotonEnergy', 'mHFHadronEnergy',
                         'mHFEMEnergy', 'mChargedHadronMultiplicity',
                         'mNeutralHadronMultiplicity',
                         'mPhotonMultiplicity', 'mElectronMultiplicity',
                         'mHFHadronMultiplicity', 'mHFEMMultiplicity', 'mNeutralEmEnergy', 'mChargedMultiplicity',
                         'mNeutralMultiplicity']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    colors = ['pink', 'green']
    n_bins = 100
    save = True  # Option to save figure
    
    df_test = pd.DataFrame(test_data, columns=variable_list)
    df_predicted = pd.DataFrame(predicted_data, columns=variable_list)
    
    plot_response(df_test,df_predicted,variable_list)
    
    # plot the input data along with the reconstructed from the AE
    for kk in np.arange(num_variables):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(test_data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(predicted_data[:, kk], color=colors[0], label='Output', alpha=0.8, bins=bin_edges)
        plt.suptitle(variable_list[kk])
        plt.xlabel(xlabel=variable_list[kk])
        plt.ylabel('Number of jets')
        plt.yscale('log')
        plt.legend()
        if save:
            plt.savefig(os.path.join(save_dir, variable_list[kk] + '.png'))
        #plt.show()

def RMS_function(response_norm):
    square = np.square(response_norm)
    MS = square.mean()
    RMS = np.sqrt(MS)
    return RMS
        
def plot_response(df_test,df_predicted,variable_list):
    
    before = df_test
    after = df_predicted

    columns = variable_list
    number_of_columns = len(columns)
    output_path = "./response/"

    with PdfPages(output_path + "comparison.pdf") as pdf:
        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])
        #figure1, (ax1, ax2) = plt.subplots(
        #    1, 2, figsize=(18.3 * (1 / 2.54) * 1.7, 13.875 * (1 / 2.54) * 1.32)
        
        axsLeft = subfigs[0].subplots(2, 1, sharex=True)
        ax1=axsLeft[0]
        ax3=axsLeft[1]

        axsRight = subfigs[1].subplots()
        ax2=axsRight

        for index, column in enumerate(columns):
            print(f"{index} of {number_of_columns}")

            response = (after - before) / before
            response_list = list(filter(lambda p: -20 <= p <= 20, response[column]))
            response_RMS = RMS_function(response_norm=response_list)
            #            minimum = int(min(before[column]+after[column]))
            #            maximum = int(max(before[column]+after[column]))
            #            diff = maximum - minimum
            #            if diff == np.inf or diff == 0:#FIXME: We have to skip some variables
            #                pdf.savefig()
            #                ax2.clear()
            #                ax1.clear()
            #                continue
            #            step = diff/100
            # counts_before, bins_before = np.histogram(before[column],bins=np.arange(minimum,maximum,step))
            
            x_min = min([min(before[column]),min(after[column])])
            x_max = max([max(before[column]),max(after[column])])
            diff = x_max - x_min
            
            counts_before, bins_before = np.histogram(
                before[column], bins=np.linspace(x_min, x_max,101)
            )
            hist_before = ax1.hist(
                bins_before[:-1], bins_before, weights=counts_before, label="Before"
            )
            # counts_after, bins_after = np.histogram(after[column],bins=np.arange(minimum,maximum,step))
            counts_after, bins_after = np.histogram(
                after[column], bins=np.linspace(x_min, x_max,101)
            )
            hist_after = ax1.hist(
                bins_after[:-1],
                bins_after,
                weights=counts_after,
                label="After",
                histtype="step",
            )
            ax1.set_title(f"{column} Distribution")
            ax1.set_xlabel(f"{column}", ha="right", x=1.0)
            ax1.set_xticks([])
            ax1.set_ylabel("Counts", ha="right", y=1.0)
            ax1.set_yscale("log")
            ax1.legend(loc="best")
            diff = x_max - x_min
            ax1.set_xlim(x_min-(diff/2)*0.1,x_max+abs(x_max*0.1))

            # Residual subplot in comparison
            #divider = make_axes_locatable(ax1)
            #ax3 = divider.append_axes("bottom", size="20%", pad=0.25)
            #ax1.figure.add_axes(ax3)
            #ax3.bar(bins_after[:-1], height=(hist_after[0] - hist_before[0])/hist_before[0])
            data_bin_centers = bins_after[:-1]+(bins_after[1:]-bins_after[:-1])/2
            ax3.scatter(data_bin_centers, ((counts_after - counts_before)/counts_before)*100) # FIXME: Dividing by zero
            ax3.axhline(y=0, linewidth=0.2, color="black")
            ax3.set_ylim(-200, 200)
            #ax3.set_ylabel("(after - before)/before")
            ax3.set_ylabel("Relative Difference [%]")
            #ax3.set_xlim(x_min-abs(x_min*0.1),x_max+abs(x_max*0.1))

            #            minimum = min(response[column])
            #            maximum = max(response[column])
            #            diff = maximum - minimum
            #            if diff == np.inf or diff == 0:
            #                pdf.savefig()
            #                ax2.clear()
            #                ax1.clear()
            #                continue
            #            step = diff/100
            # counts_response, bins_response = np.histogram(response[column],bins=np.arange(minimum,maximum,step))
            counts_response, bins_response = np.histogram(
                response[column], bins=np.arange(-2, 2, 0.1)
            )
            ax2.hist(
                bins_response[:-1],
                bins_response,
                weights=counts_response,
                label="Response",
            )
            ax2.axvline(
                np.mean(response_list),
                color="k",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean {round(np.mean(response_list),8)}",
            )
            ax2.plot([], [], " ", label=f"RMS: {round(response_RMS,8)}")

            # To have percent on the x-axis
            # formatter = mpl.ticker.FuncFormatter(to_percent)
            # ax2.xaxis.set_major_formatter(formatter)
            ax2.set_title(f"{column} Response")
            ax2.set_xlabel(f"{column} Response", ha="right", x=1.0)
            ax2.set_ylabel("Counts", ha="right", y=1.0)
            ax2.legend(loc="best")

            pdf.savefig()
            ax2.clear()
            ax1.clear()
            ax3.clear()        

def plot_residuals(test_data, predicted_data, kk):
    save_dir = "./response/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Calculate the residuals
    print(predicted_data)
    response_test = np.absolute(predicted_data - test_data)/test_data
    # residual_train = np.absolute(train_data - prediction_train)
    plt.figure()

    ## Plotting the scatter plots
    #print("These are the scatter plots")
    #plt.scatter(test_data, residual_test)
    #plt.title(f"Test data:{kk}")
    #plt.show()

    #plt.figure()
    # Plotting Histograms
    print("These are the histograms")
    plt.hist(response_test, 50)
    plt.title(f"Residuals on test data:{kk}")
    #plt.show()
    plt.savefig(f'./response/{kk}.png')

def plot_4D_data(test_data, predicted_data):
    save_dir = "D:\Desktop\GSoC-ATLAS\AE_4D_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    variable_list = [r'$E$', r'$p_T$', r'$\eta$', r'$\phi$']
    colors = ['pink', 'green']

    alph = 0.8
    n_bins = 200

    for kk in np.arange(4):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(test_data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        n_hist_pred, _, _ = plt.hist(predicted_data[:, kk], color=colors[0], label='Output', alpha=alph, bins=bin_edges)
        plt.suptitle(variable_list[kk])
        plt.xlabel(xlabel=variable_list[kk])
        plt.ylabel('Number of events')

        plt.yscale('log')
        plt.show()


def correlation_plot(data):
    data = data.drop(['entry', 'subentry'], axis=1)
    data.columns = data.columns.str.lstrip("ak5PFJets")
    data.columns = data.columns.str.lstrip("_")

    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
