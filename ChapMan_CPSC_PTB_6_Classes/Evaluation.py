

# Libraries
import os
import sys
import math
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/ECG_3block_classification'))

from functions import functions

# ------------------------------------------Reading existing files---------------------------------------------------- 

Dirction_and_labels = pd.read_excel('/kaggle/working/ECG_3block_classification/Direction_And_folds/Chap_CPSC_PTB_Direction_SingleLabels_CT-Code_v3.xlsx')

# for i,dir in enumerate(Dirction_and_labels['Ecg_dir']):
#     Dirction_and_labels['Ecg_dir'][i]= dir.replace(".../", "/kaggle/input/")

with open('/kaggle/working/ECG_3block_classification/Direction_And_folds/Train_Test_Split_8Class_Chap_CPSC_PTB.pickle', 'rb') as handle:
    Folds_splited_data = pickle.load(handle)
# ------------------------------------------Constant values and Empty lists-------------------------------------------

NumOfFold = 6
NumOfClass = 6
NumOfEpochs = 100
train_split = 0
test_split = 1
batchsize = 24

PAC_Rhythm = []
PVC_Rhythm = []
SIN_Rhythm = []
Chap_CPSC_PTB_df_With_PAC = []  
Chap_CPSC_PTB_df_Without_PAC_PVC = []

# -------------------------------------------Preprocessing part-------------------------------------------------------

# Make a Data Frame with direction for K-fold of ChapMan, CPSC, and PTB dataset using Dirction_and_labels file

# Extract the indexes of PVC rhythm to remove them from our Data Frame
for fold in range(NumOfFold):

    PVC = []
    SIN = []
    for TrTe in range(2):

        PVC.append(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type[Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type=='PVC'].index)
        SIN.append(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type[Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type=='SIN'].index)
        
    PVC_Rhythm.append([PVC[0],PVC[1]])
    SIN_Rhythm.append([SIN[0],SIN[1]])


# Make train and test data frame without PVC and a fraction of Normal
for fold in range(NumOfFold):
    Chap_CPSC_PTB = []
    Chap_CPSC_PTB_df_Without_PVC_infold = []

    for TrTe in range(2):
        
        Chap_CPSC_PTB_df = Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].drop(PVC_Rhythm[fold][TrTe].tolist())
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.sample(frac = 1,random_state=42)
        Chap_CPSC_PTB.append(Chap_CPSC_PTB_df.set_index(pd.Index(np.arange(0,len(Chap_CPSC_PTB_df)))))


    Chap_CPSC_PTB_df_With_PAC.append([Chap_CPSC_PTB[0],Chap_CPSC_PTB[1]])

# Reduce number of All 6 rhythms with a constant coefficient to number of PAC rhythm 
for fold in range(NumOfFold):
    Chap_CPSC_PTB_df_Without_PAC_PVC_infold = []
    for TrTe in range(2):
        
        PAC_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='PAC'].index
        SIN_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='SIN'].index
        SVT_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='SVT'].index
        SB_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='SB'].index
        STach_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='STach'].index
        Afib_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='Afib'].index
        AF_Rhythm = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type[Chap_CPSC_PTB_df_With_PAC[fold][TrTe].Labs_Type=='Af'].index


        # PAC_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[PAC_Rhythm]

        SIN_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[SIN_Rhythm]
        SIN_Rhythm_Sampled = SIN_Rhythm_Sampled.sample(n=len(PAC_Rhythm)-10, random_state=42)

        SVT_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[SVT_Rhythm]
        SVT_Rhythm_Sampled = SVT_Rhythm_Sampled.sample(n=len(SVT_Rhythm), random_state=42,replace=True)

        SB_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[SB_Rhythm]
        SB_Rhythm_Sampled = SB_Rhythm_Sampled.sample(n=len(PAC_Rhythm)-10, random_state=42)

        STach_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[STach_Rhythm]
        STach_Rhythm_Sampled = STach_Rhythm_Sampled.sample(n=len(PAC_Rhythm)-10, random_state=42)

        Afib_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[Afib_Rhythm]
        Afib_Rhythm_Sampled = Afib_Rhythm_Sampled.sample(n=len(PAC_Rhythm)-10, random_state=42)

        AF_Rhythm_Sampled = Chap_CPSC_PTB_df_With_PAC[fold][TrTe].iloc[AF_Rhythm]
        AF_Rhythm_Sampled = AF_Rhythm_Sampled.sample(n=len(AF_Rhythm), random_state=42,replace=True)
        
        Chap_CPSC_PTB_df = pd.concat([SIN_Rhythm_Sampled,SVT_Rhythm_Sampled,SB_Rhythm_Sampled,STach_Rhythm_Sampled,
                                                      Afib_Rhythm_Sampled,AF_Rhythm_Sampled],ignore_index=True)
        
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.sample(frac = 1,random_state=42)
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.set_index(pd.Index(np.arange(0,len(Chap_CPSC_PTB_df))))

        Chap_CPSC_PTB_df_Without_PAC_PVC_infold.append(Chap_CPSC_PTB_df)


    Chap_CPSC_PTB_df_Without_PAC_PVC.append(Chap_CPSC_PTB_df_Without_PAC_PVC_infold)


# Binarizing our k-fold labels
for fold in range(NumOfFold):
    
    for TrTe in range(2): 

        one_hot = functions.MultiLabelBinarizer()
        y_=one_hot.fit_transform(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][TrTe].CT_code.str.split(pat=','))
        print("The classes we will look at are encoded as SNOMED CT codes:")
        print(one_hot.classes_)
        y_1 = np.delete(y_, -1, axis=1)
        print("classes: {}".format(y_1.shape[1]))

        Chap_CPSC_PTB_df_Without_PAC_PVC[fold][TrTe]['Labs'] = list(y_1)
        snomed_classes = one_hot.classes_[0:-1]

model = functions.residual_network_1d(NumOfClass,trainable=True,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3)

# General report of classification of each folds
for fold in range(NumOfFold):

    model.load_weights(f'/kaggle/working/ECG_3block_classification/ChapMan_CPSC_PTB_6_Classes/Weights/Model_weights_{fold}_6Class_Chap_CPSC_PTB_PAC.weights.h5')
    print(f'Fold {fold}')
    prediction = model.predict(functions.generate_validation_data(np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Ecg_dir.tolist()),
    	np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Labs.tolist()))[0], batch_size=32)
    a = [ 'Afib','Af','SB','SVT','SIN','STach' ]
    seven_class_valu = np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Labs.tolist())
    print(classification_report(np.argmax(seven_class_valu,axis=1), np.argmax(prediction,axis=1), target_names=a),'\n')


# Confusion Matrix of each folds
"""

According to this that, our data is imbalanced,
and also to have a good analysis we have to report them on the balance form.
So, the below code able to carry out this processing

below code for carrying out this process, first of all,
find the length of the smallest class and then divide the other classes into
bunchs of the smallest class length and find a confusion matrix for each bunch
and ultimately, the mean of these confusion matrices is reported.

"""
for fold in range(NumOfFold):

    model.load_weights(f'/kaggle/working/ECG_3block_classification/ChapMan_CPSC_PTB_6_Classes/Weights/Model_weights_{fold}_6Class_Chap_CPSC_PTB_PAC.weights.h5')

    print(f'Fold {fold}')
    y_pred = model.predict(functions.generate_validation_data(np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Ecg_dir.tolist()),
    	np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Labs.tolist()))[0], batch_size=32)
    a = [ 'Afib','Af','SB','SVT','SIN','STach' ]

    actual = np.asarray(Chap_CPSC_PTB_df_Without_PAC_PVC[fold][1].Labs.tolist())

    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []

    seven_class = []
    for i in range(actual.shape[0]):
            if actual[i,0] == 1:
                c1.append(i)
            if actual[i,1] == 1:
                c2.append(i)
            if actual[i,2] == 1:
                c3.append(i)
            if actual[i,3] == 1:
                c4.append(i)
            if actual[i,4] == 1:
                c5.append(i)
            if actual[i,5] == 1:
                c6.append(i)


    for i in range(actual.shape[0]):
        for g in range(6):
            if actual[i,g] == 0:
                pass
            else:
                seven_class.append(i)


    classes = [c1,c2,c3,c4,c5,c6]
    classes_len = []

    for i in range(len(classes)):
        classes_len.append(len(classes[i]))
    classes_len

    min_num_class = min(classes_len)
    min_num_class_index = classes_len.index(min_num_class)

    devs = []
    for i in range(NumOfClass):
        d = classes_len[i]//min_num_class
        if d == 1:
            devs.append(classes_len[i]/min_num_class)
        else:
            devs.append(d)
    print(devs)

    class_arr = np.array(classes[4])
    dev_class = []
    import random
    for i in range(len(devs)):
        if devs[i]>=2:
            dev_class.append(np.random.randint(classes_len[i], size=(devs[i],min_num_class)))
        if devs[i] == 1:
            dev_class.append(np.arange(0,min_num_class).reshape(1,min_num_class))
        if 1 < devs[i] < 2 :
            dev_class.append(np.random.randint(classes_len[i], size=(1,min_num_class)))
    len(dev_class)
    for i in range(len(dev_class)):
        print(dev_class[i].shape)

    import numpy as np
    from sklearn import metrics

    mat_devd = []

    sup_mat = np.array([[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1],
                       [0,0,0,0,0,1],[0,0,0,0,0,1]])

    sup_mat_2 = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],
                       [0,0,0,0,1,0],[0,0,0,0,0,1]])


    conf_mat = np.zeros((NumOfClass,NumOfClass))

    for g in range(NumOfClass):
        cm = np.zeros((dev_class[g].shape[0],NumOfClass,NumOfClass))

        for i in range(dev_class[g].shape[0]):

            pred_without_undifined = y_pred[np.array(classes[g])[dev_class[g][i,:]]]
            pred_without_undifined_dif = y_pred[np.array(classes[-1])[dev_class[-1]]]

            conc_pred = np.concatenate((pred_without_undifined,sup_mat_2),axis=0)

            actual_without_undifined = actual[np.array(classes[g])[dev_class[g][i,:]]]
            actual_without_undifined_dif = actual[np.array(classes[-1])[dev_class[-1]]]
            conc_actual = np.concatenate((actual_without_undifined,sup_mat),axis=0)

            mat_devd.append([conc_pred,conc_actual])

            y_preed=np.argmax(conc_pred, axis=1)
            y_test=np.argmax(conc_actual, axis=1)
            cm[i,:,:] = metrics.confusion_matrix(y_test, y_preed)

            if g==min_num_class_index:
                cm = cm - 1

        cm_mean = np.rint(cm.mean(axis=0))
        conf_mat[g,:] = cm_mean[g,:]

    print(conf_mat)
    print(len(mat_devd),',',cm.shape,',',cm_mean.shape)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = a)

    cm_display.plot()
    plt.show()