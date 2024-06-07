import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance


def jsd_single(real, fake):
  # Computing the real and synthetic probabibility mass distributions (pmf) for each categorical column
  real_pmf=(real.value_counts()/real.value_counts().sum())
  fake_pmf=(fake.value_counts()/fake.value_counts().sum())
  categories = (fake.value_counts()/fake.value_counts().sum()).keys().tolist()
        
  # Ensuring the pmfs of real and synthetic data have the categories within a column in the same order
  sorted_categories = sorted(categories)
        
  real_pmf_ordered = [] 
  fake_pmf_ordered = []

  for i in sorted_categories:
    real_pmf_ordered.append(real_pmf[i])
    fake_pmf_ordered.append(fake_pmf[i])
        
  # If a category of a column is not generated in the synthetic dataset, pmf of zero is assigned
  if len(real_pmf)!=len(fake_pmf):
    zero_cats = set(real.value_counts().keys())-set(fake.value_counts().keys())
    for z in zero_cats:
      real_pmf_ordered.append(real_pmf[z])
      fake_pmf_ordered.append(0)

  # Computing the statistical similarity between real and synthetic pmfs 
  return distance.jensenshannon(real_pmf_ordered,fake_pmf_ordered, 2.0)


def jsd(real,fake,cat_cols=None,Stat_dict=None,mean=True):
  Stat_dict = Stat_dict if Stat_dict is not None else {} 

  real = real.copy()
  fake = fake.copy()

  cat_stat = []
    
  for column in real.columns:
    if cat_cols is None or column in cat_cols:
      # Computing the statistical similarity between real and synthetic pmfs 
      Stat_dict[column]= jsd_single(real[column], fake[column])
      cat_stat.append(Stat_dict[column])  

  if not mean:
    return cat_stat
  return np.mean(cat_stat)

def wasserstein_single(real, fake):
  # Scaling the real and synthetic numerical column values between 0 and 1 to obtained normalized statistical similarity
  scaler = MinMaxScaler()
  scaler.fit(real.values.reshape(-1,1))
  l1 = scaler.transform(real.values.reshape(-1,1)).flatten()
  l2 = scaler.transform(fake.values.reshape(-1,1)).flatten()
        
  # Computing the statistical similarity between scaled real and synthetic numerical distributions 
  return wasserstein_distance(l1,l2)

def wasserstein(real,fake,cat_cols=None,Stat_dict=None,mean=True):
  Stat_dict = Stat_dict if Stat_dict is not None else {} 
    
  # Lists to store the results of statistical similarities for categorical and numeric columns respectively
  num_stat = []
    
  for column in real.columns:
    if cat_cols is None or column not in cat_cols:
      dist = wasserstein_single(real[column], fake[column])
      # Computing the statistical similarity between scaled real and synthetic numerical distributions 
      Stat_dict[column]= dist
      num_stat.append(dist)
  if not mean:
    return num_stat
  return np.mean(num_stat)

def corr(df, cat_cols=None):
  return associations(df, nominal_columns=cat_cols, theil_u=True, compute_only=True)['corr'].astype("float")

def diff_corr(real,fake,cat_cols=None):

  # Computing the real and synthetic pair-wise correlations
  real_corr = corr(real, cat_cols=cat_cols)
  fake_corr = corr(fake, cat_cols=cat_cols)

  # Computing the squared norm of the difference between real and synthetic pair-wise correlations
  corr_dist = np.linalg.norm(real_corr - fake_corr)

  return corr_dist

def encode(X, enc):
  cols = enc.feature_names_in_
  df = pd.concat([
      X.drop(cols, axis=1),
      pd.DataFrame(enc.transform(X[cols]), columns=enc.get_feature_names_out())
  ], axis=1, join="inner")
  return df

def privacy_dist(a, b=None, cat_cols=None, frac=1.0, random_state=42, return_detail=False):
  assert cat_cols is not None

  ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
  ohe.fit(a[cat_cols])
  a = encode(a, ohe)
  scaler = StandardScaler().fit(a)
     
  # Sampling smaller sets of a and synthetic data to reduce the time complexity of the evaluation
  a.loc[:, :] = scaler.transform(a)
  if frac==1.0:
    a_sampled = a
  else:
    a_sampled = a.sample(n=int(len(a)*(frac)), random_state=random_state).to_numpy()
  assert len(a_sampled) > 0

  if b is not None:
    b = encode(b, ohe)
    b.loc[:, :] = scaler.transform(b)
    if frac==1.0:
      b_sampled = b
    else:
      b_sampled = b.sample(n=int(len(b)*(frac)), random_state=random_state).to_numpy()
    # Computing pair-wise distances between a and b 
    dist = metrics.pairwise_distances(a_sampled, Y=b_sampled, metric='minkowski', n_jobs=-1)
  else:
    # Computing pair-wise distances between a
    dist = metrics.pairwise_distances(a_sampled, Y=None, metric='minkowski', n_jobs=-1) 
    dist = dist[~np.eye(dist.shape[0],dtype=bool)].reshape(dist.shape[0],-1)
  assert len(dist) > 1
  
  # Computing first and second smallest nearest neighbour distances
  smallest_two_indexes = [dist[i].argsort()[:2] for i in range(len(dist))]
  smallest_two = [dist[i][smallest_two_indexes[i]] for i in range(len(dist))] 
  assert len(smallest_two) > 1      

  # Computing 5th percentiles for DCR and NNDR between and within a and synthetic datasets
  min_dist = np.array([i[0] for i in smallest_two])
  fifth_perc = np.percentile(min_dist,5)
  nn_ratio = np.array([i[0]/i[1] for i in smallest_two])
  nn_ratio = np.nan_to_num(nn_ratio, nan=0)
  nn_fifth_perc = np.percentile(nn_ratio,5)

  if return_detail:
    return smallest_two, smallest_two_indexes
  return fifth_perc, nn_fifth_perc #dcr, nndr
