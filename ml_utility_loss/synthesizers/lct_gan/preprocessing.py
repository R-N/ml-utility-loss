import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import torch
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm

EMPTY = "empty"
EMPTY_VALUE = -9999999

def log_transform(series, eps=1, EMPTY_VALUE=EMPTY_VALUE):
    # Value added to apply log to non-positive numeric values
    # Missing values indicated with EMPTY_VALUE are skipped
    lower = np.min(series[series!=EMPTY_VALUE].values) 
    if lower>0: 
        result = series.apply(lambda x: np.log(x) if x!=EMPTY_VALUE else EMPTY_VALUE)
    elif lower == 0:
        result = series.apply(lambda x: np.log(x+eps) if x!=EMPTY_VALUE else EMPTY_VALUE) 
    else:
        # Negative values are scaled to become positive to apply log
        result = series.apply(lambda x: np.log(x-lower+eps) if x!=EMPTY_VALUE else EMPTY_VALUE)
    return result, lower

def inverse_log_transform(series, lower_bound, eps=1, EMPTY_VALUE=EMPTY_VALUE):
    if lower_bound>0:
        result = series.apply(lambda x: np.exp(x) if x!=EMPTY_VALUE else EMPTY_VALUE) 
    elif lower_bound==0:
        result = series.apply(lambda x: np.ceil(np.exp(x)-eps) if ((x!=EMPTY_VALUE) & ((np.exp(x)-eps) < 0)) else (np.exp(x)-eps if x!=EMPTY_VALUE else EMPTY_VALUE))
    else: 
        result = series.apply(lambda x: np.exp(x)-eps+lower_bound if x!=EMPTY_VALUE else EMPTY_VALUE)
    return result

class DataPrep(object):
    
    """
    Data preparation class for pre-processing input data and post-processing generated data

    Variables:
    1) raw_df -> dataframe containing input data
    2) categorical -> list of categorical columns
    3) log -> list of skewed exponential numerical columns
    4) mixed -> dictionary of "mixed" column names with corresponding categorical modes 
    5) integer -> list of numeric columns without floating numbers
    6) type -> dictionary of problem type (i.e classification/regression) and target column
    7) test_ratio -> ratio of size of test to train dataset

    Methods:
    1) __init__() -> instantiates DataPrep object and handles the pre-processing steps for feeding it to the training algorithm
    2) inverse_prep() -> deals with post-processing of the generated data to have the same format as the original dataset


    """


    def __init__(self, raw_df: pd.DataFrame, categorical: list, log:list, mixed:dict, integer:list, type:dict, test_ratio:float):
        
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []
        
        self.prep(raw_df, type=type, test_ratio=test_ratio)
        
    def prep(self, raw_df: pd.DataFrame, type:dict, test_ratio:float):

        # Spliting the input data to obtain training dataset
        target_col = list(type.values())[0]
        y_real = raw_df[target_col]
        X_real = raw_df.drop(columns=[target_col])
        X_train_real, _, y_train_real, _ = model_selection.train_test_split(X_real ,y_real, test_size=test_ratio, stratify=y_real,random_state=42)        
        X_train_real[target_col]= y_train_real

        # Replacing empty strings with na if any and replace na with empty
        self.df = X_train_real
        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')

        # Dealing with empty values in numeric columns by replacing it with EMPTY_VALUE and treating it as categorical mode 
        all_columns= set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)
        for i in relevant_missing_columns:
            if i in list(self.mixed_columns.keys()):
                if EMPTY in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(lambda x: EMPTY_VALUE if x==EMPTY else x )
                    self.mixed_columns[i].append(EMPTY_VALUE)
            else:
                if EMPTY in list(self.df[i].values):   
                    self.df[i] = self.df[i].apply(lambda x: EMPTY_VALUE if x==EMPTY else x)
                    self.mixed_columns[i] = [EMPTY_VALUE]
        
        # Dealing with skewed exponential numeric distributions by applying log transformation
        if self.log_columns:
            for log_column in self.log_columns:
                self.df[log_column], self.lower_bounds[log_column] = log_transform(self.df[log_column])
        
        # Encoding categorical column using label encoding to assign each category within a column with an integer value
        for column_index, column in enumerate(self.df.columns):
            
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)
            
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]
        
        super().__init__()
        
    def inverse_prep(self, data, eps=1):
        
        # Converting generated data into a dataframe and assign column names as per original dataset
        if isinstance(data, pd.DataFrame):
            df_sample = data
        else:
            df_sample = pd.DataFrame(data, columns=self.df.columns)
        
        # Reversing the label encoding assigned to categorical columns according to the original dataset 
        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = df_sample[self.label_encoder_list[i]["column"]].astype(int)
            df_sample[self.label_encoder_list[i]["column"]] = le.inverse_transform(df_sample[self.label_encoder_list[i]["column"]])

        # Reversing log by applying exponential transformation with appropriate scaling for non-positive numeric columns 
        # EMPTY_VALUE used to denote missing values are similarly ignored
        if self.log_columns:
            for i in df_sample:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    df_sample[i] = inverse_log_transform(df_sample[i], lower_bound, eps=eps)
        
        # Rounding numeric columns without floating numbers in the original dataset
        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column]= (np.round(df_sample[column].values))
                df_sample[column] = df_sample[column].astype(int)

        # Converting back EMPTY_VALUE and EMPTY to na
        df_sample.replace(EMPTY_VALUE, np.nan,inplace=True)
        df_sample.replace('empty', np.nan,inplace=True)

        return df_sample
    
def fit_bgm(
    n_clusters,
    series=None,
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.001, # lower values result in lesser modes being active
    max_iter=100,n_init=1, random_state=42
):
    gm = BayesianGaussianMixture(
        n_components=n_clusters, 
        weight_concentration_prior_type=weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior, # lower values result in lesser modes being active
        max_iter=max_iter,n_init=n_init, random_state=random_state)
    if series is not None:
        gm.fit(series)
    return gm

# keeping only relevant modes that have higher weight than eps and are used to fit the data
def keep_modes(gm, series, n_clusters=10, eps=0.005):
    old_comp = gm.weights_ > eps
    mode_freq = (pd.Series(gm.predict(series)).value_counts().keys())
    comp = []
    for i in range(n_clusters):
        if (i in (mode_freq)) & old_comp[i]:
            comp.append(True)
        else:
            comp.append(False)
    return comp

class DataTransformer():
    
    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={}, n_clusters=10, eps=0.005):
        
        self.meta = None
        self.train_data = train_data
        self.categorical_columns= categorical_list
        self.mixed_columns= mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        self.meta = self.get_metadata()
        
    def get_metadata(self):
        
        meta = []
    
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:,index]
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                })
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })            

        return meta

    def fit(self):
        
        data = self.train_data.values
        
        # stores the corresponding bgm models for processing numeric data
        self.model = []
        
        # iterating through column information
        for id_, info in tqdm(enumerate(self.meta)):
            if info['type'] == "continuous":
                series = data[:, id_].reshape([-1, 1])
                self.fit_continuous(series)
            elif info['type'] == "mixed":
                series1 = data[:, id_].reshape([-1, 1])
            
                filter_arr = []
                for element in data[:, id_]:
                    if element not in info['modal']:
                        filter_arr.append(True)
                    else:
                        filter_arr.append(False)
                self.filter_arr.append(filter_arr)
                
                series2 = data[:, id_][filter_arr].reshape([-1, 1])
                self.fit_mixed(series1, series2, info['modal'])
            else:
                # in case of categorical columns, bgm model is ignored
                self.model.append(None)
                self.components.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']
        
    def fit_mixed(self, series1, series2, modal):
        # in case of mixed columns, two bgm models are used
        
        # first bgm model is fit to the entire data only for the purposes of obtaining a normalized value of any particular categorical 
        gm1 = self.fit_bgm(series1)
        
        # main bgm model used to fit the continuous component and serves the same purpose as with purely numeric columns
        gm2 = self.fit_bgm(series2)
        
        self.model.append((gm1,gm2))

        comp = keep_modes(gm2, series2, self.n_clusters, self.eps)
        self.components.append(comp)
        
        # modes of the categorical component are appended to modes produced by the main bgm model
        self.output_info += [(1, 'tanh'), (np.sum(comp) + len(modal), 'softmax')]
        self.output_dim += 1 + np.sum(comp) + len(modal)

    def fit_bgm(self, series=None):
        return fit_bgm(self.n_clusters, series)


    def fit_continuous(self, series):
        # fitting bgm model  
        gm = self.fit_bgm(series)
        self.model.append(gm)
        comp = keep_modes(gm, series, self.n_clusters, self.eps)
        self.components.append(comp) 
        self.output_info += [(1, 'tanh'), (np.sum(comp), 'softmax')]
        self.output_dim += 1 + np.sum(comp)

    def _transform_continuous(self, current, model, components, data):
        # mode-specific normalization occurs here
        # means and stds of the modes are obtained from the corresponding fitted bgm model
        means = model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self.n_clusters))
        # values are then normalized and stored for all modes
        features = np.empty(shape=(len(current),self.n_clusters))
        # note 4 is a multiplier to ensure values lie between -1 to 1 but this is not always guaranteed
        features = (current - means) / (4 * stds) 

        # number of distict modes
        n_opts = sum(components)                
        # storing the mode for each data point by sampling from the probability mass distribution across all modes based on fitted bgm model 
        opt_sel = np.zeros(len(data), dtype='int')
        probs = model.predict_proba(current.reshape([-1, 1]))
        probs = probs[:, components]
        for i in range(len(data)):
            pp = probs[i] + 1e-6
            pp = pp / sum(pp)
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
        
        # creating a one-hot-encoding for the corresponding selected modes
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1

        # obtaining the normalized values based on the appropriately selected mode and clipping to ensure values are within (-1,1)
        idx = np.arange((len(features)))
        features = features[:, components]
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99) 
        
        return features, probs_onehot
    
    def transform_continuous(self, id_, data):
        model = self.model[id_]
        components = self.components[id_]

        current = data[:, id_]
        current = current.reshape([-1, 1])
        
        features, probs_onehot = self._transform_continuous(current, model, components, data)

        # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
        re_ordered_phot = np.zeros_like(probs_onehot)  
        col_sums = probs_onehot.sum(axis=0)
        n = probs_onehot.shape[1]
        largest_indices = np.argsort(-1*col_sums)[:n]
        for id,val in enumerate(largest_indices):
            re_ordered_phot[:,id] = probs_onehot[:,val]

        # storing the original ordering for invoking inverse transform
        self.ordering.append(largest_indices)
        

    def transform_mixed(self, id_, info, data):
        current = data[:, id_]
        
        # means and standard deviation of modes obtained from the first fitted bgm model
        means_0 = self.model[id_][0].means_.reshape([-1])
        stds_0 = np.sqrt(self.model[id_][0].covariances_).reshape([-1])

        # list to store relevant bgm modes for categorical components
        zero_std_list = []
        
        # means and stds needed to normalize relevant categorical components
        means_needed = []
        stds_needed = []

        # obtaining the closest bgm mode to the categorical component
        for mode in info['modal']:
            # skipped for mode representing missing values
            if mode!=-9999999:
                dist = []
                for idx,val in enumerate(list(means_0.flatten())):
                    dist.append(abs(mode-val))
                index_min = np.argmin(np.array(dist))
                zero_std_list.append(index_min)
            else: continue

        
        # stores the appropriate normalized value of categorical modes
        mode_vals = []
        
        # based on the means and stds of the chosen modes for categorical components, their respective values are similarly normalized
        for idx in zero_std_list:
            means_needed.append(means_0[idx])
            stds_needed.append(stds_0[idx])
        
        for i,j,k in zip(info['modal'],means_needed,stds_needed):
            this_val  = np.clip(((i - j) / (4*k)), -.99, .99) 
            mode_vals.append(this_val)
        
        # for categorical modes representing missing values, the normalized value associated is simply 0
        if -9999999 in info["modal"]:
            mode_vals.append(0)
        
        # transforming continuous component of mixed columns similar to purely numeric columns using second fitted bgm model
        model = self.model[id_][1]
        components = self.components[id_][1]

        current = current.reshape([-1, 1])
        filter_arr = self.filter_arr[mixed_counter]
        current = current[filter_arr]

        features, probs_onehot = self._transform_continuous(current, model, components, data)
        
        # additional modes are appended to represent categorical component
        extra_bits = np.zeros([len(current), len(info['modal'])])
        temp_probs_onehot = np.concatenate([extra_bits,probs_onehot], axis = 1)
        
        # storing the final normalized value and one-hot-encoding of selected modes
        final = np.zeros([len(data), 1 + probs_onehot.shape[1] + len(info['modal'])])

        # iterates through only the continuous component
        features_curser = 0

        for idx, val in enumerate(data[:, id_]):
            
            if val in info['modal']:
                # dealing with the modes of categorical component
                category_ = list(map(info['modal'].index, [val]))[0]
                final[idx, 0] = mode_vals[category_]
                final[idx, (category_+1)] = 1
            
            else:
                # dealing with the modes of continuous component
                final[idx, 0] = features[features_curser]
                final[idx, (1+len(info['modal'])):] = temp_probs_onehot[features_curser][len(info['modal']):]
                features_curser = features_curser + 1

        # re-ordering the one-hot-encoding of modes in descending order as per their frequency of being selected
        just_onehot = final[:,1:]
        re_ordered_jhot= np.zeros_like(just_onehot)
        n = just_onehot.shape[1]
        col_sums = just_onehot.sum(axis=0)
        largest_indices = np.argsort(-1*col_sums)[:n]
        
        for id,val in enumerate(largest_indices):
                re_ordered_jhot[:,id] = just_onehot[:,val]
        
        final_features = final[:,0].reshape([-1, 1])
        
        # storing the original ordering for invoking inverse transform
        self.ordering.append(largest_indices)

        return final_features, re_ordered_jhot
        

    def transform(self, data):
        
        # stores the transformed values
        values = []

        # used for accessing filter_arr for transforming mixed columns
        mixed_counter = 0
        
        # iterating through column information
        for id_, info in tqdm(enumerate(self.meta)):
            print(f"Building transformer: {info}")
            current = data[:, id_]
            if info['type'] == "continuous":
                features, re_ordered_phot = self.transform_continuous(id_, data)
                # storing transformed numeric column represented as normalized values and corresponding modes 
                values += [features, re_ordered_phot]
            elif info['type'] == "mixed":
                final_features, re_ordered_jhot = self.transform_mixed(id_, info, data)
                
                values += [final_features, re_ordered_jhot]
                
                mixed_counter = mixed_counter + 1
    
            else:
                # for categorical columns, standard one-hot-encoding is applied where categories are in descending order of frequency by default
                self.ordering.append(None)
                col_t = np.zeros([len(data), info['size']])
                idx = list(map(info['i2s'].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)
                
        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        
        # stores the final inverse transformed generated data 
        data_t = np.zeros([len(data), len(self.meta)])
        
        # used to iterate through the columns of the raw generated data
        st = 0

        # iterating through original column information
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                
                
                data_t[:, id_] = result
                
                # moving to the next set of columns in the raw generated data in correspondance to original column information
                st += 1 + np.sum(self.components[id_])
                
            elif info['type'] == "mixed":
                
                result = self.inverse_transform_mixed(id_, info, data, st)
            
                data_t[:, id_] = result 

                st += 1 + np.sum(self.components[id_]) + len(info['modal'])
                
            else:
                # reversing one hot encoding back to label encoding for categorical columns 
                current = data[:, st:st + info['size']]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))
                st += info['size']
        return data_t
    
    def inverse_transform_continuous(self, id_, info, data, st):
        # obtaining the generated normalized values and clipping for stability
        u = data[:, st]
        u = np.clip(u, -1, 1)
        
        # obtaining the one-hot-encoding of the modes representing the normalized values
        v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]
        
        # re-ordering the modes as per their original ordering
        order = self.ordering[id_] 
        v_re_ordered = np.zeros_like(v)
        for id,val in enumerate(order):
            v_re_ordered[:,val] = v[:,id]
        v = v_re_ordered

        # ensuring un-used modes are represented with -100 such that they can be ignored when computing argmax
        v_t = np.ones((data.shape[0], self.n_clusters)) * -100
        v_t[:, self.components[id_]] = v
        v = v_t
        
        # obtaining approriate means and stds as per the appropriately selected mode for each data point based on fitted bgm model
        means = self.model[id_].means_.reshape([-1])
        stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        
        # executing the inverse transformation 
        result = u * 4 * std_t + mean_t

        return result
    
    def inverse_transform_mixed(self, id_, info, data, st):
        # obtaining the generated normalized values and corresponding modes
        u = data[:, st]
        u = np.clip(u, -1, 1)
        full_v = data[:,(st+1):(st+1)+len(info['modal'])+np.sum(self.components[id_])]
        
        # re-ordering the modes as per their original ordering
        order = self.ordering[id_]
        full_v_re_ordered = np.zeros_like(full_v)
        for id,val in enumerate(order):
            full_v_re_ordered[:,val] = full_v[:,id]
        full_v = full_v_re_ordered                
        
        # modes of categorical component
        mixed_v = full_v[:,:len(info['modal'])]
        
        # modes of continuous component
        v = full_v[:,-np.sum(self.components[id_]):]

        # similarly ensuring un-used modes are represented with -100 to be ignored while computing argmax
        v_t = np.ones((data.shape[0], self.n_clusters)) * -100
        v_t[:, self.components[id_]] = v
        v = np.concatenate([mixed_v,v_t], axis=1)       
        p_argmax = np.argmax(v, axis=1)

        # obtaining the means and stds of the continuous component using second fitted bgm model
        means = self.model[id_][1].means_.reshape([-1]) 
        stds = np.sqrt(self.model[id_][1].covariances_).reshape([-1]) 

        # used to store the inverse-transformed data points
        result = np.zeros_like(u)

        for idx in range(len(data)):
            # in case of categorical mode being selected, the mode value itself is simply assigned 
            if p_argmax[idx] < len(info['modal']):
                argmax_value = p_argmax[idx]
                result[idx] = float(list(map(info['modal'].__getitem__, [argmax_value]))[0])
            else:
                # in case of continuous mode being selected, similar inverse-transform for purely numeric values is applied
                std_t = stds[(p_argmax[idx]-len(info['modal']))]
                mean_t = means[(p_argmax[idx]-len(info['modal']))]
                result[idx] = u[idx] * 4 * std_t + mean_t
        return result