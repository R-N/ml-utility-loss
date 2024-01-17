import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

def jsd(real,fake,cat_cols,Stat_dict=None):
  Stat_dict = Stat_dict if Stat_dict is not None else {} 

  really = real.copy()
  fakey = fake.copy()

  cat_stat = []
    
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

  return np.mean(cat_stat)


def wasserstein(real,fake,cat_cols,Stat_dict=None):
  Stat_dict = Stat_dict if Stat_dict is not None else {} 
    
  # Lists to store the results of statistical similarities for categorical and numeric columns respectively
  num_stat = []
    
  for column in real.columns:
    if column not in cat_cols:
      # Scaling the real and synthetic numerical column values between 0 and 1 to obtained normalized statistical similarity
      scaler = MinMaxScaler()
      scaler.fit(real[column].values.reshape(-1,1))
      l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
      l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            
      # Computing the statistical similarity between scaled real and synthetic numerical distributions 
      Stat_dict[column]= (wasserstein_distance(l1,l2))
      num_stat.append(Stat_dict[column])

  return np.mean(num_stat)

def diff_corr(real,fake,cat_cols=None):

  # Computing the real and synthetic pair-wise correlations
  real_corr = associations(real, nominal_columns=cat_cols, compute_only=True)['corr']
  fake_corr = associations(fake, nominal_columns=cat_cols, compute_only=True)['corr']

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

def privacy_dist(a, b=None, cat_cols=None, frac=1.0, random_state=42):
  assert cat_cols is not None

  ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
  ohe.fit(a[cat_cols])
  a = encode(a, ohe)
  scaler = StandardScaler().fit(a)
     
  # Sampling smaller sets of a and synthetic data to reduce the time complexity of the evaluation
  a.loc[:, :] = scaler.transform(a)
  a_sampled = a.sample(n=int(len(a)*(frac)), random_state=random_state).to_numpy()
  assert len(a_sampled) > 0

  if b is not None:
    b = encode(b, ohe)
    b.loc[:, :] = scaler.transform(b)
    b_sampled = b.sample(n=int(len(b)*(frac)), random_state=random_state).to_numpy()
    # Computing pair-wise distances between a and b 
    dist = metrics.pairwise_distances(a_sampled, Y=b_sampled, metric='minkowski', n_jobs=-1)
  else:
    # Computing pair-wise distances between a
    dist = metrics.pairwise_distances(a_sampled, Y=None, metric='minkowski', n_jobs=-1) 
    dist = dist[~np.eye(dist.shape[0],dtype=bool)].reshape(dist.shape[0],-1)
  assert len(dist) > 0
  
  # Computing first and second smallest nearest neighbour distances
  smallest_two_indexes = [dist[i].argsort()[:2] for i in range(len(dist))]
  smallest_two = [dist[i][smallest_two_indexes[i]] for i in range(len(dist))] 
  assert len(smallest_two) > 0      

  # Computing 5th percentiles for DCR and NNDR between and within a and synthetic datasets
  min_dist = np.array([i[0] for i in smallest_two])
  print(set(min_dist))
  fifth_perc = np.percentile(min_dist,5)
  nn_ratio = np.array([i[0]/i[1] for i in smallest_two])
  print(set(nn_ratio))
  nn_fifth_perc = np.percentile(nn_ratio,5)

  return fifth_perc, nn_fifth_perc #dcr, nndr
