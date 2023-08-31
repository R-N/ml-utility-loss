import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
# from model.synthesizer.transformer import ImageTransformer,DataTransformer
from tqdm import tqdm


def random_choice_prob_index_sampling(probs,col_idx):
    
    """
    Used to sample a specific category within a chosen one-hot-encoding representation 

    Inputs:
    1) probs -> probability mass distribution of categories 
    2) col_idx -> index used to identify any given one-hot-encoding
    
    Outputs:
    1) option_list -> list of chosen categories 
    
    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding 
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Condvec(object):
    
    """
    This class is responsible for sampling conditional vectors to be supplied to the generator

    Variables:
    1) model -> list containing an index of highlighted categories in their corresponding one-hot-encoded represenations
    2) interval -> an array holding the respective one-hot-encoding starting positions and sizes     
    3) n_col -> total no. of one-hot-encoding representations
    4) n_opt -> total no. of distinct categories across all one-hot-encoding representations
    5) p_log_sampling -> list containing log of probability mass distribution of categories within their respective one-hot-encoding representations
    6) p_sampling -> list containing probability mass distribution of categories within their respective one-hot-encoding representations

    Methods:
    1) __init__() -> takes transformed input data with respective column information to compute class variables
    2) sample_train() -> used to sample the conditional vector during training of the model
    3) sample() -> used to sample the conditional vector for generating data after training is finished
    
    """


    def __init__(self, data, output_info):
              
        self.model = []
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0 
        self.p_log_sampling = []  
        self.p_sampling = [] 
        
        # iterating through the transformed input data columns 
        st = 0
        for item in output_info:
            # ignoring columns that do not represent one-hot-encodings
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                # using starting (st) and ending (ed) position of any given one-hot-encoded representation to obtain relevant information
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)  
                log_freq = np.log(freq + 1)  
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed
           
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator during training

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> a matrix containing a conditional vector for each data point to be generated 
        2) mask -> a matrix to identify chosen one-hot-encodings across the batch
        3) idx -> list of chosen one-hot encoding across the batch
        4) opt1prime -> selected categories within chosen one-hot-encodings

        """

        if self.n_col == 0:
            return None
        batch = batch
        
        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # matrix of shape (batch x total no. of one-hot-encoded representations) with 1 in indexes of chosen representations and 0 elsewhere
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1
        
        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling, idx) 
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator after training is finished

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> an array containing a conditional vector for each data point to be generated 
        """

        if self.n_col == 0:
            return None
        
        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        
        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):   
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec

def cond_loss(data, output_info, c, m):
    
    """
    Used to compute the conditional loss for ensuring the generator produces the desired category as specified by the conditional vector

    Inputs:
    1) data -> raw data synthesized by the generator 
    2) output_info -> column informtion corresponding to the data transformer
    3) c -> conditional vectors used to synthesize a batch of data
    4) m -> a matrix to identify chosen one-hot-encodings across the batch

    Outputs:
    1) loss -> conditional loss corresponding to the generated batch 

    """
    
    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            print(torch.argmax(c[:, st_c:ed_c], dim=1))
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none'
            )
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask 
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

class Sampler(object):
    
    """
    This class is used to sample the transformed real data according to the conditional vector 

    Variables:
    1) data -> real transformed input data
    2) model -> stores the index values of data records corresponding to any given selected categories for all columns
    3) n -> size of the input data

    Methods:
    1) __init__() -> initiates the sampler object and stores class variables 
    2) sample() -> takes as input the number of rows to be sampled (n), chosen column (col)
                   and category within the column (opt) to sample real records accordingly
    """

    def __init__(self, data, output_info):
        
        super(Sampler, self).__init__()
        
        self.data = data
        self.model = []
        self.n = len(data)
        
        # counter to iterate through columns
        st = 0
        # iterating through column information
        for item in output_info:
            # ignoring numeric columns
            if item[1] == 'tanh':
                st += item[0]
                continue
            # storing indices of data records for all categories within one-hot-encoded representations
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                # iterating through each category within a one-hot-encoding
                for j in range(item[0]):
                    # storing the relevant indices of data records for the given categories
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        
        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        
        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []
        
        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen category and one-hot-encoding
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        
        return self.data[idx]

    def sample_idx(self, n, col, opt):
        
        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        
        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []
        
        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen category and one-hot-encoding
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        
        return idx

def get_st_ed(target_col_index,output_info):
    
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) output_info -> column information corresponding to the data after applying the data transformer

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c= 0
    # counter to iterate through column information
    tc= 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    
    
    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed= st+output_info[tc][0] 
    
    return (st,ed)
