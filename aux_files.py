import tensorflow_datasets as tfds
import numpy as np
import os

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


