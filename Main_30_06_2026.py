import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from numpy import matlib
from DOA import DOA
from Global_Vars import Global_Vars
from Model_CNN_SVM import Model_CNN_SVM
from Model_FuDNN import Model_FuDNN
from Model_IFD_ASTN import Model_IFD_ASTN
from Model_SSAE_LReLU import Model_SSAE_LReLU
from Model_XceptionNet import Model_XceptionNet
from NBOA import NBOA
from POA import POA
from PROPOSED import PROPOSED
from Plot_Results import *
from RBMO import RBMO
from objfun import objfun_cls


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the Dataset
an = 0
if an == 1:
    Dataset_folders = './Datasets/kidney_disease.csv'
    df = pd.read_csv(Dataset_folders, on_bad_lines='skip')
    Target = df['classification']
    df.drop(['classification'], inplace=True, axis=1)
    columns_to_convert = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    Datas = convert_to_numeric(df, columns_to_convert)
    df = df.replace(r'^\s+|\s+$', '', regex=True)
    df = df.replace(['?', '\t?', '', 'nan', 'None'], np.nan)
    df = df.replace(r'\t', '', regex=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)
    Datas = df.astype(np.float32).values

    Datas = np.asarray(Datas).astype('float')
    Target = np.asarray(Target).reshape(-1, 1)
    uni, count = np.unique(Target, return_counts=True)
    print('Unique : ' + str(uni) + ' Counts : ' + str(count))
    mapping = {'notckd': 0, 'ckd': 1, 'ckd\t': 1}
    Target_mapped = np.vectorize(mapping.get)(Target)

    index = np.arange(len(Datas))
    np.random.shuffle(index)
    Shuffled_Datas = Datas[index]
    Shuffled_Target = Target_mapped[index]
    np.save('Index.npy', index)
    np.save('Datas.npy', Shuffled_Datas)
    np.save('Targets.npy', Shuffled_Target)

# feature extraction
an = 0
if an == 1:
    Datas = np.load('Datas.npy', allow_pickle=True)
    Targets = np.load('Targets.npy', allow_pickle=True)
    Feature = Model_SSAE_LReLU(Datas, Targets)
    np.save('Features.npy', Feature)

# Optimization for classification
an = 0
if an == 1:
    Feat = np.load('Features.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron count, Learning Rate, Step per Epoches in STN
    xmin = matlib.repmat([5, 0.01, 100], Npop, 1)
    xmax = matlib.repmat([255, 0.99, 500], Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("NBOA...")
    [bestfit1, fitness1, bestsol1, time1] = NBOA(initsol, fname, xmin, xmax, Max_iter)  # NBOA

    print("RBMO...")
    [bestfit2, fitness2, bestsol2, time2] = RBMO(initsol, fname, xmin, xmax, Max_iter)  # RBMO

    print("DOA...")
    [bestfit3, fitness3, bestsol3, time3] = DOA(initsol, fname, xmin, xmax, Max_iter)  # DOA

    print("POA...")
    [bestfit4, fitness4, bestsol4, time4] = POA(initsol, fname, xmin, xmax, Max_iter)  # POA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))  # Save the Fitness
    np.save('BestSol.npy', np.asarray(BestSol_CLS))  # Save the BestSol

# Classification by Varying Activation Function
an = 0
if an == 1:
    Feat = np.load('Features.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # Load the Best Solution
    EVAL = []
    Activation = ["linear", "relu", "tanh", "softmax", "sigmoid"]
    for ACT in range(len(Activation)):
        Eval = np.zeros((10, 25))
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        for j in range(BestSol.shape[0]):
            sol = BestSol[j, :]
            Eval[j, :], pred = Model_IFD_ASTN(Train_Data, Train_Target, Test_Data, Test_Target, ACT=Activation[ACT], sol=sol)
        Eval[5, :], pred1 = Model_CNN_SVM(Train_Data, Train_Target, Test_Data, Test_Target, ACT=Activation[ACT])
        Eval[6, :], pred2 = Model_FuDNN(Train_Data, Train_Target, Test_Data, Test_Target, ACT=Activation[ACT])
        Eval[7, :], pred3 = Model_XceptionNet(Train_Data, Train_Target, Test_Data, Test_Target, ACT=Activation[ACT])
        Eval[8, :], pred4 = Model_IFD_ASTN(Train_Data, Train_Target, Test_Data, Test_Target, ACT=Activation[ACT])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_ACT.npy', np.asarray(EVAL))  # Save the Eval all EP

Plot_Conv()
ROC_curve()
Plot_Confusion()
PLot_ACTIVATION()
Plot_KFold()
