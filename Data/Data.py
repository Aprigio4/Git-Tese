import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def full_brain_AD_data(dir_read):
    
    # Test set
    x = np.load((dir_read+'PET_npy/FDG_PET_AD_Baseline_VI_1.npy'))
    X_test = x.reshape((1,60,128,128,1))
    
    for i in range(2,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_AD_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
        
    for i in range(1,11):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    # Train set
    x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Baseline_VI_11.npy'))
    X_train = x.reshape((1,60,128,128,1))
    
    for i in range(12,96):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X = np.concatenate((X_train, x), axis=0)
        X = X_train.reshape((-1,60,128,128,1))
    
    for i in range(11,87):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X = np.concatenate((X_train, x), axis=0)
        X = X_train.reshape((-1,60,128,128,1))
    
    for i in range(11,75):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
        
    for i in range(11,60):
        x = np.load((dir_read+'/PET_npy/FDG_PET_AD_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    return X_test, X_train
    
def full_brain_CN_data(dir_read):
    # Test set
    x = np.load((dir_read+'/PET_npy/FDG_PET_CN_Baseline_VI_1.npy'))
    X_test = x.reshape((1,60,128,128,1))
    
    for i in range(2,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
        
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
        
    # Train set    
    x = np.load((dir_read+'PET_npy/FDG_PET_CN_Baseline_VI_11.npy'))
    X_train = x.reshape((1,60,128,128,1))
    
    
    for i in range(2,100):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    for i in range(1,95):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    for i in range(1,86):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
        
    for i in range(1,85):
        x = np.load((dir_read+'PET_npy/FDG_PET_CN_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    return X_test,X_train

def full_brain_MCI_data(dir_read):
    
    # Test set
    x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Baseline_VI_1.npy'))
    X_test = x.reshape((1,60,128,128,1))
    
    for i in range(2,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
        
    for i in range(1,11):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_test = np.concatenate((X_test, x), axis=0)
        X_test = X_test.reshape((-1,60,128,128,1))
    
    # Train set
    x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Baseline_VI_11.npy'))
    X_train = x.reshape((1,60,128,128,1))
    
    
    for i in range(12,208):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Baseline_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    for i in range(11,189):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month6_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    for i in range(11,178):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month12_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X_train = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
        
    for i in range(11,143):
        x = np.load((dir_read+'PET_npy/FDG_PET_MCI_Month24_VI_'+str(i)+'.npy'))
        x = x.reshape((1,60,128,128,1))     
        X = np.concatenate((X_train, x), axis=0)
        X_train = X_train.reshape((-1,60,128,128,1))
    
    return X_test, X_train


def group_data_full_brain(dir_save, dir_read):
    norm = 32700  
    AD_X_test, AD_X_train = full_brain_AD_data(dir_read)
    CN_X_test, CN_X_train = full_brain_CN_data(dir_read)
    MCI_X_test, MCI_X_train = full_brain_MCI_data(dir_read)
    
    AD_X_train = AD_X_train/norm
    AD_X_test = AD_X_test/norm
    CN_X_train = CN_X_train/norm
    CN_X_test = CN_X_test/norm
    MCI_X_train = MCI_X_train/norm
    MCI_X_test = MCI_X_test/norm
    # CN & AD (0,1)
    
    AD_Y_test = np.ones(AD_X_test.shape[0])
    AD_Y_train = np.ones(AD_X_train.shape[0])
    
    CN_Y_test = np.zeros(CN_X_test.shape[0])
    CN_Y_train = np.zeros(CN_X_train.shape[0])
    
    CN_AD_X_test = np.concatenate((AD_X_test, CN_X_test), axis=0)/norm
    CN_AD_Y_test = np.concatenate((AD_Y_test, CN_Y_test), axis=0)
    
    CN_AD_X_train = np.concatenate((AD_X_train, CN_X_train), axis=0)/norm
    CN_AD_Y_train = np.concatenate((AD_Y_train, CN_Y_train), axis=0)
    
    CN_AD_X_train, CN_AD_Y_train = shuffle(CN_AD_X_train, CN_AD_Y_train)
    
    np.save(dir_save+"FullBrain/CN_AD_X_train", CN_AD_X_train)
    np.save(dir_save+"FullBrain/CN_AD_Y_train", CN_AD_Y_train)
    
    np.save(dir_save+"FullBrain/CN_AD_X_test", CN_AD_X_test)
    np.save(dir_save+"FullBrain/CN_AD_Y_test", CN_AD_Y_test)
    
    # CN & MCI (0,1)
    
    MCI_Y_test = np.ones(MCI_X_test.shape[0])
    MCI_Y_train = np.ones(MCI_X_train.shape[0])
    
    CN_MCI_X_test = np.concatenate((MCI_X_test, CN_X_test), axis=0)
    CN_MCI_Y_test = np.concatenate((MCI_Y_test, CN_Y_test), axis=0)
    
    CN_MCI_X_train = np.concatenate((MCI_X_train, CN_X_train), axis=0)
    CN_MCI_Y_train = np.concatenate((MCI_Y_train, CN_Y_train), axis=0)
    
    CN_MCI_X_train, CN_MCI_Y_train = shuffle(CN_MCI_X_train, CN_MCI_Y_train)
    
    np.save(dir_save+"FullBrain/CN_MCI_X_train", CN_MCI_X_train)
    np.save(dir_save+"FullBrain/CN_MCI_Y_train", CN_MCI_Y_train)
    
    np.save(dir_save+"FullBrain/CN_MCI_X_test", CN_MCI_X_test)
    np.save(dir_save+"FullBrain/CN_MCI_Y_test", CN_MCI_Y_test)
    
    # MCI & AD (0,1)
        
    MCI_Y_test = np.zeros(MCI_X_test.shape[0])
    MCI_Y_train = np.zeros(MCI_X_train.shape[0])
    
    MCI_AD_X_test = np.concatenate((AD_X_test, MCI_X_test), axis=0)
    MCI_AD_Y_test = np.concatenate((AD_Y_test, MCI_Y_test), axis=0)
    
    MCI_AD_X_train = np.concatenate((AD_X_train, MCI_X_train), axis=0)
    MCI_AD_Y_train = np.concatenate((AD_Y_train, MCI_Y_train), axis=0)
    
    MCI_AD_X_train, MCI_AD_Y_train = shuffle(MCI_AD_X_train, MCI_AD_Y_train)
    
    np.save(dir_save+"FullBrain/MCI_AD_X_train", MCI_AD_X_train)
    np.save(dir_save+"FullBrain/MCI_AD_Y_train", MCI_AD_Y_train)
    
    np.save(dir_save+"FullBrain/MCI_AD_X_test", MCI_AD_X_test)
    np.save(dir_save+"FullBrain/MCI_AD_Y_test", MCI_AD_Y_test)
    
    
    # CN & MCI & AD ((0,0,1), (0,1,0), (1,0,0))
    
    AD_Y_test = np.ones(AD_X_test.shape[0]) * 3
    AD_Y_train = np.zeros((AD_X_train.shape[0],3))
    for i in range(AD_X_train.shape[0]):
        AD_Y_train[i,0] = 1
    
    MCI_Y_test = np.ones(MCI_X_test.shape[0]) * 2
    MCI_Y_train = np.zeros((MCI_X_train.shape[0],3))
    for i in range(MCI_X_train.shape[0]):
        MCI_Y_train[i,1] = 1
    
    CN_Y_test = np.ones(CN_X_test.shape[0]) * 1
    CN_Y_train = np.zeros((CN_X_train.shape[0],3))
    for i in range(CN_X_train.shape[0]):
        CN_Y_train[i,2] = 1
        
    CN_AD_X_test = np.concatenate((AD_X_test, CN_X_test), axis=0)
    CN_AD_Y_test = np.concatenate((AD_Y_test, CN_Y_test), axis=0)
    CN_MCI_AD_X_test = np.concatenate((CN_AD_X_test, MCI_X_test), axis=0)
    CN_MCI_AD_Y_test = np.concatenate((CN_AD_Y_test, MCI_Y_test), axis=0)
    
    CN_AD_X_train = np.concatenate((AD_X_train, CN_X_train), axis=0)
    CN_AD_Y_train = np.concatenate((AD_Y_train, CN_Y_train), axis=0)
    CN_MCI_AD_X_train = np.concatenate((CN_AD_X_train, MCI_X_train), axis=0)
    CN_MCI_AD_Y_train = np.concatenate((CN_AD_Y_train, MCI_Y_train), axis=0)
    
    CN_MCI_AD_X_train, CN_MCI_AD_Y_train = shuffle(CN_MCI_AD_X_train, CN_MCI_AD_Y_train)
    
    np.save(dir_save+"FullBrain/CN_MCI_AD_X_train", CN_MCI_AD_X_train)
    np.save(dir_save+"FullBrain/CN_MCI_AD_Y_train", CN_MCI_AD_Y_train)
    
    np.save(dir_save+"FullBrain/CN_MCI_AD_X_test", CN_MCI_AD_X_test)
    np.save(dir_save+"FullBrain/CN_MCI_AD_Y_test", CN_MCI_AD_Y_test)
  
  
def spilt_flip(X):
    
    #separate hemisfiers and flip
    X_HL = X[:,:,:64,:,:]
    X_HR = X[:,:,:64,:,:]
    X_HR = X_HR[:,:,::-1,:,:]
    
    return X_HL, X_HR

def flip(X):
  return X[:,:,::-1,:,:]

def simple_brain_split(classe, dir_save):

    #Train
    X = np.load(dir_save+"FullBrain/"+classe+"_X_train.npy")
    Y = np.load(dir_save+"FullBrain/"+classe+"_Y_train.npy")
    
    X_HL, X_HR = spilt_flip(X)
    
    np.save(dir_save+"SplitBrain/"+classe+"_X_train_HL", X_HL)
    np.save(dir_save+"SplitBrain/"+classe+"_X_train_HR", X_HR)
    np.save(dir_save+"SplitBrain/"+classe+"_Y_train", Y)
    
    #Test
    X = np.load(dir_save+"FullBrain/"+classe+"_X_test.npy")
    Y = np.load(dir_save+"FullBrain/"+classe+"_Y_test.npy")
    
    X_HL, X_HR = spilt_flip(X)
    
    np.save(dir_save+"SplitBrain/"+classe+"_X_test_HL", X_HL)
    np.save(dir_save+"SplitBrain/"+classe+"_X_test_HR", X_HR)
    np.save(dir_save+"SplitBrain/"+classe+"_Y_test", Y)


def group_data_split_brain(dir_save):
        
    simple_brain_split("CN_AD", dir_save)
    simple_brain_split("CN_MCI", dir_save)
    simple_brain_split("MCI_AD", dir_save)
    simple_brain_split("CN_MCI_AD", dir_save)
    
def augmentation(classe, dir_save):
    #Train
    X = np.load(dir_save+"FullBrain/"+classe+"_X_train.npy")
    Y = np.load(dir_save+"FullBrain/"+classe+"_Y_train.npy")
    
    X_aug = flip(X)
    X = np.concatenate((X, X_aug), axis=0)
    Y = np.concatenate((Y, Y), axis=0)

    np.save(dir_save+"FullBrain_aug/"+classe+"_X_train", X)
    np.save(dir_save+"FullBrain_aug/"+classe+"_Y_train", Y)
    

def aug_all(dir_save):
    augmentation("CN_AD", dir_save)
    augmentation("CN_MCI", dir_save)
    augmentation("MCI_AD", dir_save)
    augmentation("CN_MCI_AD", dir_save)
