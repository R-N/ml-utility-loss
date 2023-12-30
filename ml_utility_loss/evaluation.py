import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

def stat_sim(real_path,fake_path,cat_cols=None):
    
  """
  Returns statistical similarity metrics

  Inputs:
  1) real_path -> path to real data
  2) fake_path -> path to synthetic data
  3) cat_cols -> list of categorical column names
    

  Outputs:
  1) List containing the difference in avg (normalized) wasserstein distance across numeric columns, avg jensen shannon divergence 
  across categorical columns and euclidean norm of the difference in pair-wise correlations between real and synthetic datasets
    
  """
  Stat_dict={}
  # Loading real and synthetic data
  real = pd.read_csv(real_path)
  fake = pd.read_csv(fake_path)

  really = real.copy()
  fakey = fake.copy()

  # Computing the real and synthetic pair-wise correlations
  real_corr = associations(real, nominal_columns=cat_cols, compute_only=True)['corr']
  fake_corr = associations(fake, nominal_columns=cat_cols, compute_only=True)['corr']

  # Computing the squared norm of the difference between real and synthetic pair-wise correlations
  corr_dist = np.linalg.norm(real_corr - fake_corr)
    
  # Lists to store the results of statistical similarities for categorical and numeric columns respectively
  cat_stat = []
  num_stat = []
    
  for column in real.columns:
        
    if column in cat_cols:

      # Computing the real and synthetic probabibility mass distributions (pmf) for each categorical column
      real_pmf=(really[column].value_counts()/really[column].value_counts().sum())
      fake_pmf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
      categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            
      # Ensuring the pmfs of real and synthetic data have the categories within a column in the same order
      sorted_categories = sorted(categories)
            
      real_pmf_ordered = [] 
      fake_pmf_ordered = []

      for i in sorted_categories:
        real_pmf_ordered.append(real_pmf[i])
        fake_pmf_ordered.append(fake_pmf[i])
            
      # If a category of a column is not generated in the synthetic dataset, pmf of zero is assigned
      if len(real_pmf)!=len(fake_pmf):
        zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
        for z in zero_cats:
          real_pmf_ordered.append(real_pmf[z])
          fake_pmf_ordered.append(0)

      # Computing the statistical similarity between real and synthetic pmfs 
      Stat_dict[column]=(distance.jensenshannon(real_pmf_ordered,fake_pmf_ordered, 2.0))
      cat_stat.append(Stat_dict[column])  
    else:
      # Scaling the real and synthetic numerical column values between 0 and 1 to obtained normalized statistical similarity
      scaler = MinMaxScaler()
      scaler.fit(real[column].values.reshape(-1,1))
      l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
      l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            
      # Computing the statistical similarity between scaled real and synthetic numerical distributions 
      Stat_dict[column]= (wasserstein_distance(l1,l2))
      num_stat.append(Stat_dict[column])

  return [np.mean(num_stat),np.mean(cat_stat),corr_dist]

def privacy_metrics(real_path,fake_path,data_percent=15):

  """
  Returns privacy metrics
  
  Inputs:
  1) real_path -> path to real data
  2) fake_path -> path to corresponding synthetic data
  3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

  Outputs:
  1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
  along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
  
  """
  
  # Loading real and synthetic datasets and removing duplicates if any
  real = pd.read_csv(real_path).drop_duplicates(keep=False)
  fake = pd.read_csv(fake_path).drop_duplicates(keep=False)

  # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
  real_sampled = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
  fake_sampled = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()

  # Scaling real and synthetic data samples
  scalerR = StandardScaler()
  scalerR.fit(real_sampled)
  scalerF = StandardScaler()
  scalerF.fit(fake_sampled)
  df_real_scaled = scalerR.transform(real_sampled)
  df_fake_scaled = scalerF.transform(fake_sampled)
  
  # Computing pair-wise distances between real and synthetic 
  dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
  # Computing pair-wise distances within real 
  dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
  # Computing pair-wise distances within synthetic
  dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1) 
  
  # Removes distances of data points to themselves to avoid 0s within real and synthetic 
  rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
  rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
  
  # Computing first and second smallest nearest neighbour distances between real and synthetic
  smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
  smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
  # Computing first and second smallest nearest neighbour distances within real
  smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
  smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
  # Computing first and second smallest nearest neighbour distances within synthetic
  smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
  smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
  

  # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
  min_dist_rf = np.array([i[0] for i in smallest_two_rf])
  fifth_perc_rf = np.percentile(min_dist_rf,5)
  min_dist_rr = np.array([i[0] for i in smallest_two_rr])
  fifth_perc_rr = np.percentile(min_dist_rr,5)
  min_dist_ff = np.array([i[0] for i in smallest_two_ff])
  fifth_perc_ff = np.percentile(min_dist_ff,5)
  nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
  nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
  nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
  nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
  nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
  nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    
  return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 


def privacy_metrics_2(real_data, fake_data, task_type, data_percent=15):

    """
    Returns privacy metrics
    
    Inputs:
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    """
    X_num_real, X_cat_real, y_real = real_data
    X_num_fake, X_cat_fake, y_fake = fake_data

    if task_type == 'regression':
        X_num_real = np.concatenate([X_num_real, y_real[:, np.newaxis]], axis=1)
        X_num_fake = np.concatenate([X_num_fake, y_fake[:, np.newaxis]], axis=1)
    else:
        if X_cat_fake is None:
            X_cat_real = y_real[:, np.newaxis].astype(int).astype(str)
            X_cat_fake = y_fake[:, np.newaxis].astype(int).astype(str)
        else:
            X_cat_real = np.concatenate([X_cat_real, y_real[:, np.newaxis].astype(int).astype(str)], axis=1)
            X_cat_fake = np.concatenate([X_cat_fake, y_fake[:, np.newaxis].astype(int).astype(str)], axis=1)

    if len(y_real) > 50000:
        ixs = np.random.choice(len(y_real), 50000, replace=False)
        X_num_real = X_num_real[ixs]
        X_cat_real = X_cat_real[ixs] if X_cat_real is not None else None
    
    if len(y_fake) > 50000:
        ixs = np.random.choice(len(y_fake), 50000, replace=False)
        X_num_fake = X_num_fake[ixs]
        X_cat_fake = X_cat_fake[ixs] if X_cat_fake is not None else None


    mm = MinMaxScaler().fit(X_num_real)
    X_real = mm.transform(X_num_real)
    X_fake = mm.transform(X_num_fake)
    if X_cat_real is not None:
        ohe = OneHotEncoder().fit(X_cat_real)
        X_cat_real = ohe.transform(X_cat_real) / np.sqrt(2)
        X_cat_fake = ohe.transform(X_cat_fake) / np.sqrt(2)

        X_real = np.concatenate([X_real, X_cat_real.todense()], axis=1)
        X_fake = np.concatenate([X_fake, X_cat_fake.todense()], axis=1)

    # Computing pair-wise distances between real and synthetic 
    dist_rf = pairwise_distances(X_fake, Y=X_real, metric='l2', n_jobs=-1)
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]

    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    return min_dist_rf 
