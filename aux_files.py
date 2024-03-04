import tensorflow_datasets as tfds
import numpy as np
import os
import random
# from torch.utils.data import random_split


def get_tfds_dataset(name):

  #Check if data folder exists
  cur_path = os.path.dirname(os.path.abspath(__file__))
  save_path=os.path.join(cur_path, 'data')
  filename=name+".npz"

  if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("Data folder does not exist, creating it...")

    ds_train, ds_test = tfds.as_numpy(
        tfds.load(
            name,
            split=['train', 'test'],
            batch_size=-1,
            as_dataset_kwargs={'shuffle_files': False}))
    print("Saving downloaded dataset")
    np.savez(os.path.join(save_path, filename), train_images=ds_train['image'], test_images=ds_test['image'], train_labels=ds_train['label'], test_labels=ds_test['label'])
    train_images, train_labels, test_images, test_labels = ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label'] 
  else:
    if os.path.isfile(os.path.join(save_path, filename)):
      print("Loading data...")
      data = np.load(os.path.join(save_path, filename))
      train_images, train_labels, test_images, test_labels = data['train_images'], data['train_labels'], data['test_images'], data['test_labels']
    else:
      print("Data folder exists but the data does not, downloading data...")
      ds_train, ds_test = tfds.as_numpy(
        tfds.load(
            name,
            split=['train', 'test'],
            batch_size=-1,
            as_dataset_kwargs={'shuffle_files': False}))
      print("Saving downloaded mnist dataset")
      np.savez(os.path.join(save_path, filename), train_images=ds_train['image'], test_images=ds_test['image'], train_labels=ds_train['label'], test_labels=ds_test['label'])
      train_images, train_labels, test_images, test_labels = ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label'] 

  return train_images, train_labels, test_images, test_labels

def class_balanced_sample(sample_size: int, 
                          labels: np.ndarray,
                          *arrays: np.ndarray, **kwargs: int):
  """Get random sample_size unique items consistently from equal length arrays.

  The items are class_balanced with respect to labels.

  Args:
    sample_size: Number of elements to get from each array from arrays. Must be
      divisible by the number of unique classes
    labels: 1D array enumerating class label of items
    *arrays: arrays to sample from; all have same length as labels
    **kwargs: pass in a seed to set random seed

  Returns:
    A tuple of indices sampled and the corresponding sliced labels and arrays
  """
  if labels.ndim != 1:
    raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
  n = len(labels)
  if not all([n == len(arr) for arr in arrays[1:]]):
    raise ValueError(f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
  classes = np.unique(labels)
  n_classes = len(classes)
  n_per_class, remainder = divmod(sample_size, n_classes)
  if remainder != 0:
    raise ValueError(
        f'Number of classes {n_classes} in labels must divide sample size {sample_size}.'
    )
  if kwargs.get('seed') is not None:
    np.random.seed(kwargs['seed'])
  inds = np.concatenate([
      np.random.choice(np.where(labels == c)[0], n_per_class, replace=False)
      for c in classes
  ])
  return (inds, labels[inds].copy()) + tuple(
      [arr[inds].copy() for arr in arrays])


def get_normalization_data(arr):
  channel_means = np.mean(arr, axis=(0, 1))
  channel_stds = np.std(arr, axis=(0, 1))
  return channel_means, channel_stds

def normalize(arr, mean, std):
  return (arr - mean) / std


def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
  assert len(x.shape) == 1
  one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
  if center:
    one_hot_vectors = one_hot_vectors - 1. / num_classes
  return one_hot_vectors

def data_split(proportions,indexes,idx_batch,N,n_nets):
    np.random.shuffle(indexes)
    
    ## Balance
    proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
    proportions = proportions/proportions.sum()
    
    proportions = (np.cumsum(proportions)*len(indexes)).astype(int)[:-1]
    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(indexes,proportions))]
    return idx_batch

def print_number_of_samples(labels):
        
    freq = np.bincount((labels.squeeze()).astype(int),minlength = 8)

    print('number_of_samples:',freq)
    return  

def generate_imbalanced_data(X_train_total, Y_train_total, X_test_total, Y_test_total, seed, n_nets, alpha = 2):
  np.random.seed(seed)
  random.seed(seed)
  # LABELS_TRAIN = jnp.argmax(Y_train_total, axis=1)

  # X_train_total, Y_train_total, X_test_total, Y_test_total = load_mnist()
  print(Y_train_total.shape)

  print_number_of_samples(Y_train_total)
  print_number_of_samples(Y_test_total)
  # print(X_train_total.shape)
  labels = np.unique(Y_train_total)

  indexes = [[] for i in range(len(labels))]
  indexes_test = [[] for i in range(len(labels))]



  for i in labels:
      indexes[i] = np.where(Y_train_total==i)[0]
      indexes_test[i] = np.where(Y_test_total==i)[0]

      
  min_size = 0
  N = Y_train_total.shape[0]
  N_test = Y_test_total.shape[0]
  net_dataidx_map = {}
  # n_nets = len(labels)
  # alpha = 2
  while min_size < 100:
      idx_batch = [[] for _ in range(n_nets)]
      idx_batch_test = [[] for _ in range(n_nets)]
      # for each class in the dataset
      for k in labels:
          proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
          print('k: ', k, 'proportions: ', proportions)
          proportions_test = [1 / n_nets for _ in range(n_nets)]
          idx_batch = data_split(proportions,indexes[k],idx_batch,N,n_nets)
          idx_batch_test = data_split(proportions_test,indexes_test[k],idx_batch_test,N_test,n_nets)
          # for idx_j in idx_batch:
              # print(len(idx_j))
          min_size = min([len(idx_j) for idx_j in idx_batch]) 

  client_train_X = []
  client_train_y = []
  client_train_labels = []
  for i,i_s in enumerate(idx_batch):
      # print(i)
      print_number_of_samples(Y_train_total[i_s])
      label_client = Y_train_total[i_s]
      client_train_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_train_total[i_s]
      client_train_X.append(data_client)
      client_train_y.append(label_client)
      # np.savez(outfile+'chunk_'+str(i), x=data_client, y=label_client)
  # outfile = './data/BloodMnist11/test/'
  client_test_X = []
  client_test_y = []
  client_test_labels = []
  for i,i_s in enumerate(idx_batch_test):
      # print(i)
      print_number_of_samples(Y_test_total[i_s])
      label_client = Y_test_total[i_s]
      client_test_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_test_total[i_s]
      client_test_X.append(data_client)
      client_test_y.append(label_client)

  return client_train_X, client_train_y, client_train_labels, client_test_X, client_test_y, client_test_labels

def generate_balanced_data(X_train_total, Y_train_total, X_test_total, Y_test_total, seed, n_nets):
  np.random.seed(seed)
  random.seed(seed)
  # LABELS_TRAIN = jnp.argmax(Y_train_total, axis=1)

  # X_train_total, Y_train_total, X_test_total, Y_test_total = load_mnist()
  print(Y_train_total.shape)

  print_number_of_samples(Y_train_total)
  print_number_of_samples(Y_test_total)
  # print(X_train_total.shape)
  labels = np.unique(Y_train_total)

  indexes = [[] for i in range(len(labels))]
  indexes_test = [[] for i in range(len(labels))]



  for i in labels:
      indexes[i] = np.where(Y_train_total==i)[0]
      indexes_test[i] = np.where(Y_test_total==i)[0]

      
  min_size = 0
  N = Y_train_total.shape[0]
  N_test = Y_test_total.shape[0]
  net_dataidx_map = {}
  # n_nets = len(labels)
  while min_size < 100:
      idx_batch = [[] for _ in range(n_nets)]
      idx_batch_test = [[] for _ in range(n_nets)]
      # for each class in the dataset
      for k in labels:
          proportions = [1 / n_nets for _ in range(n_nets)]
          print('k: ', k, 'proportions: ', proportions)
          idx_batch = data_split(proportions,indexes[k],idx_batch,N,n_nets)
          idx_batch_test = data_split(proportions,indexes_test[k],idx_batch_test,N_test,n_nets)
          # for idx_j in idx_batch:
              # print(len(idx_j))
          min_size = min([len(idx_j) for idx_j in idx_batch]) 

  client_train_X = []
  client_train_y = []
  client_train_labels = []
  for i,i_s in enumerate(idx_batch):
      # print(i)
      print_number_of_samples(Y_train_total[i_s])
      label_client = Y_train_total[i_s]
      client_train_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_train_total[i_s]
      client_train_X.append(data_client)
      client_train_y.append(label_client)
      # np.savez(outfile+'chunk_'+str(i), x=data_client, y=label_client)
  # outfile = './data/BloodMnist11/test/'
  client_test_X = []
  client_test_y = []
  client_test_labels = []
  for i,i_s in enumerate(idx_batch_test):
      # print(i)
      print_number_of_samples(Y_test_total[i_s])
      label_client = Y_test_total[i_s]
      client_test_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_test_total[i_s]
      client_test_X.append(data_client)
      client_test_y.append(label_client)

  return client_train_X, client_train_y, client_train_labels, client_test_X, client_test_y, client_test_labels

def generate_seperate_data(X_train_total, Y_train_total, X_test_total, Y_test_total, seed, n_nets):
  np.random.seed(seed)
  random.seed(seed)
  # LABELS_TRAIN = jnp.argmax(Y_train_total, axis=1)

  # X_train_total, Y_train_total, X_test_total, Y_test_total = load_mnist()
  print(Y_train_total.shape)

  print_number_of_samples(Y_train_total)
  print_number_of_samples(Y_test_total)
  # print(X_train_total.shape)
  labels = np.unique(Y_train_total)

  indexes = [[] for i in range(len(labels))]
  indexes_test = [[] for i in range(len(labels))]



  for i in labels:
      indexes[i] = np.where(Y_train_total==i)[0]
      indexes_test[i] = np.where(Y_test_total==i)[0]

      
  min_size = 0
  N = Y_train_total.shape[0]
  N_test = Y_test_total.shape[0]
  net_dataidx_map = {}
  # n_nets = len(labels)
  while min_size < 100:
      idx_batch = [[] for _ in range(n_nets)]
      idx_batch_test = [[] for _ in range(n_nets)]
      # for each class in the dataset
      for k in labels:
          proportions = [0 for _ in range(n_nets)]
          proportions[k] = 1
          proportions_test = [1 / n_nets for _ in range(n_nets)]
          print('k: ', k, 'proportions: ', proportions)
          idx_batch = data_split(proportions,indexes[k],idx_batch,N,n_nets)
          idx_batch_test = data_split(proportions_test,indexes_test[k],idx_batch_test,N_test,n_nets)
          # for idx_j in idx_batch:
              # print(len(idx_j))
          min_size = min([len(idx_j) for idx_j in idx_batch]) 

  client_train_X = []
  client_train_y = []
  client_train_labels = []
  for i,i_s in enumerate(idx_batch):
      # print(i)
      print_number_of_samples(Y_train_total[i_s])
      label_client = Y_train_total[i_s]
      client_train_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_train_total[i_s]
      client_train_X.append(data_client)
      client_train_y.append(label_client)
      # np.savez(outfile+'chunk_'+str(i), x=data_client, y=label_client)
  # outfile = './data/BloodMnist11/test/'
  client_test_X = []
  client_test_y = []
  client_test_labels = []
  for i,i_s in enumerate(idx_batch_test):
      # print(i)
      print_number_of_samples(Y_test_total[i_s])
      label_client = Y_test_total[i_s]
      client_test_labels.append(label_client)
      label_client = one_hot(label_client, len(labels))
      data_client = X_test_total[i_s]
      client_test_X.append(data_client)
      client_test_y.append(label_client)

  return client_train_X, client_train_y, client_train_labels, client_test_X, client_test_y, client_test_labels

  

