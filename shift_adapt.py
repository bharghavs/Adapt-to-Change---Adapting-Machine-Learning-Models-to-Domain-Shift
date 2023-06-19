# Name: Bharghav Srikhakollu
# Date: 02-23-2023
#######################################################################################################
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import label_shift_adaptation as ls
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reference Citation:
# RandomForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# Gaussian Process: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
# K-Neighbor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Dummy Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
# Confusion Matrix Display: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
# Matplotlib: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
# Feature Importance: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#######################################################################################################
# Read CSV file using Pandas: dataframe
#######################################################################################################
# matplotlib - settings and plotting
plt.rcParams['font.family'] = "serif"
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# csv files names
list_of_csv = ['train-TX', 'val-TX', 'test1-TX', 'test2-FL', 'test3-FL']
df_list = []
 
# append csv files into the df_list
cols = ['Severity', 'Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

for i in range(len(list_of_csv)):
    df = pd.read_csv(list_of_csv[i] + ".csv", usecols = cols)
    df_list.append(df)
  
# building x and y : train-tx
XY_train_tx = df_list[0].to_numpy()
X_train_tx, Y_train_tx = XY_train_tx[:, 1:], XY_train_tx[:, 0]
Y_train_tx = Y_train_tx.astype('int')
val_classes = np.unique(Y_train_tx)

# building x and y : val-tx
XY_val_tx = df_list[1].to_numpy()
X_val_tx, Y_val_tx = XY_val_tx[:, 1:], XY_val_tx[:, 0]
Y_val_tx = Y_val_tx.astype('int')

# building x and y : test1-tx
XY_test1_tx = df_list[2].to_numpy()
X_test1_tx, Y_test1_tx = XY_test1_tx[:, 1:], XY_test1_tx[:, 0]
Y_test1_tx = Y_test1_tx.astype('int')

# building x and y : test2-fl
XY_test2_fl = df_list[3].to_numpy()
X_test2_fl, Y_test2_fl = XY_test2_fl[:, 1:], XY_test2_fl[:, 0]
Y_test2_fl = Y_test2_fl.astype('int')

# building x and y : test3-fl
XY_test3_fl = df_list[4].to_numpy()
X_test3_fl, Y_test3_fl = XY_test3_fl[:, 1:], XY_test3_fl[:, 0]
Y_test3_fl = Y_test3_fl.astype('int')

#######################################################################################################
# Step 3 - train 4 classifiers + 2 dummy classifiers
# Step 4 - predictions for all the classifiers with validation and test data sets
#######################################################################################################
# initializations

val_acc_tx = []
test1_acc_tx = []
test2_acc_fl = []
test3_acc_fl = []
test1_wt_tx = []
test2_wt_fl = []
test3_wt_fl = []
test1_new_tx = []
test2_new_fl = []
test3_new_fl = []
gp_val_pred = []
nn3_val_pred = []
nn3_test1_tx_bbsc = []
gp_test2_fl_bbsc = []
gp_test1_tx_bbsc = []

kernel = 1.0 * RBF(10.0)
clf_names = ['Baseline: most_frequent', 'Baseline: stratified', 'RandomForest', 'GaussianProcess', '3-NearestNeighbor', '9-NearestNeighbor']
# classifiers
classifiers = [DummyClassifier(strategy = 'most_frequent', random_state = 100), 
               DummyClassifier(strategy = 'stratified', random_state = 200),
               RandomForestClassifier(criterion = 'gini', random_state = 300, n_estimators = 100), 
               GaussianProcessClassifier(kernel = kernel, random_state = 400),
               KNeighborsClassifier(n_neighbors = 3, weights = 'uniform'),
               KNeighborsClassifier(n_neighbors = 9, weights = 'uniform')]
j = 0
# fit and predicting - val, test1-tx, test2-fl, test3-fl
for clf in classifiers:
    print('Fit and Predicting Val-TX, Test1-TX, Test2-FL, Test3-FL accuracies and BBSC accuracies for: ', clf_names[j])
    # fit the classifier
    clf.fit(X_train_tx, Y_train_tx)
    
    #************************************************************************************#
    # Step 10: Extra Credit - Feature Importance (Random Forest)
    #************************************************************************************#

    if j == 2:
        feature_names = cols[1:]
        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)
        clf_importances = pd.Series(importances, index = feature_names)

        fig, ax = plt.subplots()
        clf_importances.plot.bar(yerr = std, ax = ax)
        ax.set_title('Feature Importances')
        ax.set_ylabel('Mean Decrease (impurity)')
        plt.savefig('Feature_Importance.jpg', bbox_inches = 'tight')

    #************************************************************************************#
    # val-tx
    #************************************************************************************#    
    # val-tx set accuracy
    Y_pred_val = clf.predict(X_val_tx)
    acc_val = round(accuracy_score(Y_val_tx, Y_pred_val)*100, 2)
    val_acc_tx.append(acc_val)
    # just saving the val pred for checking on question - 9(D) - gaussian and 3-nn
    if j == 3:
        gp_val_pred = Y_pred_val
    if j == 4:
        nn3_val_pred = Y_pred_val

    #************************************************************************************#
    # test1-tx
    #************************************************************************************#
    # test1-tx set accuracy
    Y_pred_test1 = clf.predict(X_test1_tx)
    # classifier - predicted probabilities for test1-tx
    Y_prob_test1 = clf.predict_proba(X_test1_tx)
    # accuracy - test1-tx
    acc_test1 = round(accuracy_score(Y_test1_tx, Y_pred_test1)*100, 2)
    test1_acc_tx.append(acc_test1)
    # 'j' is the switch to apply adapt wts/pred for only classifiers, not baselines
    if j > 1:
        # compute adaptation weights - test1-tx
        wt_test1 = ls.analyze_val_data(Y_val_tx, Y_pred_val, Y_pred_test1)
        wt_test1 = list(np.around(np.array(wt_test1), 2))
        test1_wt_tx.append(wt_test1)
        # adapt classifier predictions - test1-tx
        wt_test1_pred, wt_test1_prob = ls.update_probs(val_classes, wt_test1, Y_pred_test1, Y_prob_test1)
        # calculate new accuracy - test1-tx
        acc_new_test1 = round(accuracy_score(Y_test1_tx, wt_test1_pred)*100, 2)
        # just saving the bbsc pred for checking on question - 9(D) - 3-nn
        if j == 4:
            nn3_test1_tx_bbsc = wt_test1_pred
        if j == 3:
            gp_test1_tx_bbsc = wt_test1_pred
    
        test1_new_tx.append(acc_new_test1)
        
    #************************************************************************************#
    # test2-fl
    #************************************************************************************#
    # test2-fl set accuracy
    Y_pred_test2 = clf.predict(X_test2_fl)
    # classifier - predicted probabilities for test2-fl
    Y_prob_test2 = clf.predict_proba(X_test2_fl)
    # accuracy - test2-fl
    acc_test2 = round(accuracy_score(Y_test2_fl, Y_pred_test2)*100, 2)
    test2_acc_fl.append(acc_test2)
    # 'j' is the switch to apply adapt wts/pred for only classifiers, not baselines
    if j > 1:
        # compute adaptation weights - test2-fl
        wt_test2 = ls.analyze_val_data(Y_val_tx, Y_pred_val, Y_pred_test2)
        wt_test2 = list(np.around(np.array(wt_test2), 2))
        test2_wt_fl.append(wt_test2)
        # adapt classifier predictions - test2-fl
        wt_test2_pred, wt_test2_prob = ls.update_probs(val_classes, wt_test2, Y_pred_test2, Y_prob_test2)
        # calculate new accuracy - test2-fl
        acc_new_test2 = round(accuracy_score(Y_test2_fl, wt_test2_pred)*100, 2)
        # just saving the bbsc pred for checking on question - 9(D) - gp
        if j == 3:
            gp_test2_fl_bbsc = wt_test2_pred
            
        test2_new_fl.append(acc_new_test2)
    
    #************************************************************************************#
    # test3-fl
    #************************************************************************************#
    # test3-fl set accuracy
    Y_pred_test3 = clf.predict(X_test3_fl)
    # classifier - predicted probabilities for test3-fl
    Y_prob_test3 = clf.predict_proba(X_test3_fl)
    # accuracy - test3-fl
    acc_test3 = round(accuracy_score(Y_test3_fl, Y_pred_test3)*100, 2)
    test3_acc_fl.append(acc_test3)
    # 'j' is the switch to apply adapt wts/pred for only classifiers, not baselines
    if j > 1:
        # compute adaptation weights - test3-fl
        wt_test3 = ls.analyze_val_data(Y_val_tx, Y_pred_val, Y_pred_test3)
        wt_test3 = list(np.around(np.array(wt_test3), 2))
        test3_wt_fl.append(wt_test3)
        # adapt classifier predictions - test3-fl
        wt_test3_pred, wt_test3_prob = ls.update_probs(val_classes, wt_test3, Y_pred_test3, Y_prob_test3)
        # calculate new accuracy - test3-fl
        acc_new_test3 = round(accuracy_score(Y_test3_fl, wt_test3_pred)*100, 2)
        test3_new_fl.append(acc_new_test3)
       
    j = j + 1
   
#######################################################################################################
# Step 6 - test accuracy of original predictions and test accuracy after using BBSC
#######################################################################################################

acc_table_names = ['Accuracy', 'Val-TX', 'Test1-TX', 'Test2-FL', 'Test3-FL']
acc_table_data = [acc_table_names,
                  ['Baseline: most_frequent', val_acc_tx[0], test1_acc_tx[0], test2_acc_fl[0], test3_acc_fl[0]],
                  ['Baseline: stratified', val_acc_tx[1], test1_acc_tx[1], test2_acc_fl[1], test3_acc_fl[1]],
                  ['RandomForest', val_acc_tx[2], [test1_acc_tx[2], test1_new_tx[0]], [test2_acc_fl[2], test2_new_fl[0]], [test3_acc_fl[2], test3_new_fl[0]]],
                  ['GaussianProcess', val_acc_tx[3], [test1_acc_tx[3], test1_new_tx[1]], [test2_acc_fl[3], test2_new_fl[1]], [test3_acc_fl[3], test3_new_fl[1]]],
                  ['3-NearestNeighbor', val_acc_tx[4], [test1_acc_tx[4], test1_new_tx[2]], [test2_acc_fl[4], test2_new_fl[2]], [test3_acc_fl[4], test3_new_fl[2]]],
                  ['9-NearestNeighbor', val_acc_tx[5], [test1_acc_tx[5], test1_new_tx[3]], [test2_acc_fl[5], test2_new_fl[3]], [test3_acc_fl[5], test3_new_fl[3]]]]
print(tabulate(acc_table_data, headers = 'firstrow', tablefmt = 'grid'))

#######################################################################################################
# Step 7 - adaptation weights calculated for each classifier and for each test set
#######################################################################################################

adp_wt_names = ['Adaptation Weights', 'Test1-TX', 'Test2-FL', 'Test3-FL']
adp_wt_data = [adp_wt_names,
               ['RandomForest', test1_wt_tx[0], test2_wt_fl[0], test3_wt_fl[0]],
               ['GaussianProcess', test1_wt_tx[1], test2_wt_fl[1], test3_wt_fl[1]],
               ['3-NearestNeighbor', test1_wt_tx[2], test2_wt_fl[2], test3_wt_fl[2]],
               ['9-NearestNeighbor', test1_wt_tx[3], test2_wt_fl[3], test3_wt_fl[3]]]
print(tabulate(adp_wt_data, headers = 'firstrow', tablefmt = 'grid'))

#######################################################################################################
# Step 8 - true class label distribution of all four data sets
#######################################################################################################
label_1 = []
label_2 = []
label_3 = []
label_4 = []
datasets = [Y_val_tx, Y_test1_tx, Y_test2_fl, Y_test3_fl]
# calculate true class label distribution - normalize
for dataset in datasets:
    unique, freq = np.unique(dataset, return_counts = True)
    freq = np.divide(freq, len(dataset))
    label_1.append(freq[0])
    label_2.append(freq[1])
    label_3.append(freq[2])
    label_4.append(freq[3])

df = pd.DataFrame([['Label 1', label_1[0], label_1[1], label_1[2], label_1[3],], 
                   ['Label 2', label_2[0], label_2[1], label_2[2], label_2[3],], 
                   ['Label 3', label_3[0], label_3[1], label_3[2], label_3[3],],
                   ['Label 4', label_4[0], label_4[1], label_4[2], label_4[3],]],
                  columns=['Severity', 'Val-TX', 'Test1-TX', 'Test2-FL', 'Test3-FL'])
  
df.plot(x = 'Severity',
        kind = 'bar',
        stacked = False,
        title='True Class Label Distribution')
plt.savefig('True_Label_Dist.jpg', bbox_inches = 'tight')
#######################################################################################################
# Step 9(D): Confusion Matricies and distribution of predictions
#######################################################################################################
inp_true = [Y_val_tx, Y_val_tx]
inp_pred = [gp_val_pred, nn3_val_pred]
titles = ['Confusion Matrix: GP: Val-TX', 'Confusion Matrix: 3-NN: Val-TX']
fig_names = ['cm_gp_val_tx.jpg', 'cm_3nn_val_tx']

for i in range(len(inp_true)):
    conf_mat = confusion_matrix(inp_true[i], inp_pred[i])
    disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = val_classes)
    disp.plot()
    plt.title(titles[i])
    plt.savefig(fig_names[i])
    plt.show()
    
# gaussian process - distribution of predictions of test set (test2_fl) after BBSC
unique, freq = np.unique(gp_test2_fl_bbsc, return_counts = True)
print("Distribution of predictions of test2_fl: gaussian process is: ", freq, 'for labels ', unique)
# 3-nn - distribution of predictions of test set (test1_tx) after BBSC
unique, freq = np.unique(nn3_test1_tx_bbsc, return_counts = True)
print("Distribution of predictions of test1_tx: 3-nn is: ", freq, 'for labels ', unique)
# gaussian process - distribution of predictions of test set (test1_tx) after BBSC
unique, freq = np.unique(gp_test1_tx_bbsc, return_counts = True)
print("Distribution of predictions of test1_tx: gaussian process is: ", freq, 'for labels ', unique)
#######################################################################################################    