import itertools
import time
import warnings

from absl import app
from absl import flags

import jax
from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.example_libraries import optimizers
#from jax.example_libraries import stax
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp
#from examples import datasets
import examples as datasets
import numpy.random as npr

import neural_tangents as nt
from neural_tangents import stax

import numpy as np

import functools

from aux_files import class_balanced_sample, one_hot, get_tfds_dataset, get_normalization_data, normalize
from zca_preprocess import apply_preprocess

from jax import scipy as sp

from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score


from autodp.calibrator_zoo import generalized_eps_delta_calibrator
from autodp.mechanism_zoo import SubsampleGaussianMechanism

#Try to avoid memory (OOM)  problems
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" #Seems that avoid OOM problems
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import matplotlib.pyplot as plt

import gc

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('reg', 1e-6, 'Regularization parameter')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('dataset', 'mnist', 'Dataset')
flags.DEFINE_string('architecture', 'FC', 'choice of neural network architecture yielding the corresponding NTK')
flags.DEFINE_integer('support_size', 10, 'Support dataset size')
flags.DEFINE_integer('width', 200, 'NTK width')
flags.DEFINE_float('delta', 1e-5, 'Delta param for DP')
flags.DEFINE_float('epsilon', 1. , 'Epsilon param for DP')
flags.DEFINE_boolean(
    'zca', False, 'If True apply zca_preprocessing on RGB data.')
flags.DEFINE_boolean(
    'random_init', False, 'Init support data as random images.')

# Define the KIP loss
def loss(params, batch, y_support, kernel_fn, reg=1e-6):
  
  inputs, targets = batch
 
  print("loss params.shape=", params.shape)
  print("loss inputs.shape=", inputs.shape)


  k_ss = kernel_fn(params, params)
  k_ts = kernel_fn(inputs, params)

  print("k_ts.shape=", k_ts.shape)
  print("k_ss.shape=", k_ss.shape)

  k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
  pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
  mse_loss = 0.5*jnp.mean((pred - targets) ** 2)
  del k_ss, k_ts, k_ss_reg, pred

  return mse_loss

def eval_acc(params, y_support, x_test, y_test, kernel_fn, reg=1e-6):
  k_ss = kernel_fn(params, params)
  k_ts = kernel_fn(x_test, params)

  k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
  pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
  mse_loss = 0.5*jnp.mean((pred - y_test) ** 2)
  acc = jnp.mean(jnp.argmax(pred, axis=1) == jnp.argmax(y_test, axis=1))
  return mse_loss, acc


def clipped_grad(params, l2_norm_clip, single_example_batch, y_support, kernel_fn, reg):
  """Evaluate gradient for a single-example batch and clip its grad norm."""
  grads = grad(loss)(params, single_example_batch, y_support, kernel_fn, reg)
  nonempty_grads, tree_def = tree_flatten(grads)

  total_grad_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads]))
  
  divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
  normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
  del grads, nonempty_grads, total_grad_norm, divisor
  return tree_unflatten(tree_def, normalized_nonempty_grads)


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size, y_support, kernel_fn, small_batch_size, reg):
  """Return differentially private gradients for params, evaluated on batch."""
  if small_batch_size:
    print("Computing with small batches")
    global_clips = []
    inputs, targets = batch
    n_splits = int(inputs.shape[0] / small_batch_size)

    for split_id in range(n_splits):
      small_inputs = inputs[split_id *small_batch_size : (split_id + 1) *small_batch_size]
      small_targets = targets[split_id *small_batch_size : (split_id + 1) *small_batch_size]
      small_batch = small_inputs, small_targets

      clipped_grads = vmap(clipped_grad, (None, None, 0, None, None, None))(params, l2_norm_clip, small_batch, y_support, kernel_fn, reg)
      clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
      small_aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
      global_clips.append(small_aggregated_clipped_grads)
      del small_inputs, small_targets, small_batch, clipped_grads, clipped_grads_flat, small_aggregated_clipped_grads

    aggregated_clipped_grads = [sum(i) for i in zip(*global_clips)]
    #print("aggregated_clipped_grads=", aggregated_clipped_grads)
  else: #Full batch update
    clipped_grads = vmap(clipped_grad, (None, None, 0, None, None, None))(params, l2_norm_clip, batch, y_support, kernel_fn, reg)  
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    del clipped_grads, clipped_grads_flat

  rngs = random.split(rng, len(aggregated_clipped_grads))
  noised_aggregated_clipped_grads = [
      g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
      for r, g in zip(rngs, aggregated_clipped_grads)]
  normalized_noised_aggregated_clipped_grads = [
      g / batch_size for g in noised_aggregated_clipped_grads]

  del aggregated_clipped_grads, noised_aggregated_clipped_grads, rngs

  return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)


def shape_as_image(images, labels, dataset, dummy_dim=False):
  if dataset=='mnist' or dataset=='fashion_mnist':
    target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
    images_reshaped = jnp.reshape(images, target_shape)
  elif dataset=='cifar10' or dataset=='svhn_cropped' or dataset=='cifar100':
    target_shape = (-1, 1, 32, 32, 3) if dummy_dim else (-1, 32, 32, 3)
    images_reshaped = jnp.reshape(images, target_shape) 

  return images_reshaped, labels

def delete_live_buffers():
  for device in jax.local_devices():
    lbs = device.live_buffers()
    for lb in lbs:
      lb.delete()

def main(_):

  # define NTK architecture

  if FLAGS.architecture == 'FC':
    if FLAGS.dataset == 'cifar100':
        num_classes=100
    else:
        num_classes=10
    init_random_params, predict, kernel_ntk = stax.serial(
    stax.Flatten(),
    stax.Dense(FLAGS.width, W_std = np.sqrt(2), b_std = 0.1, parameterization = 'ntk'),
    stax.Relu(),
    stax.Dense(num_classes,  W_std = np.sqrt(2), b_std = 0.1, parameterization = 'ntk')
    )

    print("NTK with FC network")
  elif FLAGS.architecture == 'CNN':
    if FLAGS.dataset == 'cifar100':
        num_classes=100
    else:
        num_classes=10
    #Following https://github.com/google-research/google-research/blob/master/kip/nn_training.ipynb
    init_random_params, predict, kernel_ntk = stax.serial(
    stax.Conv(FLAGS.width, (3, 3), padding='SAME'),
    stax.Relu(),
    stax.Conv(FLAGS.width, (3, 3), W_std = np.sqrt(2), b_std = 0.1, padding='SAME', parameterization = 'ntk'),
    stax.Identity(),
    stax.Relu(),
    stax.AvgPool((2,2), strides=(2,2)),
    stax.Conv(FLAGS.width, (3, 3), W_std = np.sqrt(2), b_std = 0.1, padding='SAME', parameterization = 'ntk'),
    stax.Identity(),
    stax.Relu(),
    stax.AvgPool((2,2), strides=(2,2)),
    stax.Conv(FLAGS.width, (3, 3), W_std = np.sqrt(2), b_std = 0.1, padding='SAME', parameterization = 'ntk'),
    stax.Identity(),
    stax.Relu(),
    stax.AvgPool((2,2), strides=(2,2)),
    stax.Flatten(),
    stax.Dense(num_classes, W_std = np.sqrt(2), b_std = 0.1, parameterization = 'ntk')
    )
    print("NTK with CNN network")
  else:
    raise NotImplementedError(f'Unrecognized architecture {FLAGS.architecture}')
  kernel_fn = jax.jit(functools.partial(kernel_ntk, get='ntk'))

  if FLAGS.dataset == 'cifar100':
    num_classes=100
    zca_param=0.1
  elif FLAGS.dataset == 'cifar10':
    num_classes=10
    zca_param=0.1
  elif FLAGS.dataset == 'svhn_cropped':
    num_classes=10
    zca_param=100
  else:
    num_classes=10
    zca_param=0

  if FLAGS.dataset == 'mnist':
    train_images, train_labels, test_images, test_labels = datasets.mnist()
    print("train_labels=", train_labels)
    LABELS_TRAIN=jnp.argmax(train_labels, axis=1)
    Y_TRAIN=one_hot(LABELS_TRAIN, num_classes)
  else:
    X_TRAIN_RAW, LABELS_TRAIN, X_TEST_RAW, LABELS_TEST = get_tfds_dataset(FLAGS.dataset)
    if FLAGS.dataset == 'fashion_mnist':
      channel_means, channel_stds = get_normalization_data(X_TRAIN_RAW)
      train_images, test_images = normalize(X_TRAIN_RAW, channel_means, channel_stds), normalize(X_TEST_RAW, channel_means, channel_stds)
    else:
      if FLAGS.zca:
        print("Apply zca preprocessing")
        train_images, test_images = apply_preprocess(X_TRAIN_RAW, X_TEST_RAW, zca_param, False)
      else: 
        channel_means, channel_stds = get_normalization_data(X_TRAIN_RAW)
        train_images, test_images = normalize(X_TRAIN_RAW, channel_means, channel_stds), normalize(X_TEST_RAW, channel_means, channel_stds) 
    Y_TRAIN, test_labels = one_hot(LABELS_TRAIN, num_classes), one_hot(LABELS_TEST, num_classes) 


  print("train_images.shape=", train_images.shape)

  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
  num_batches = num_complete_batches + bool(leftover)
  key = random.PRNGKey(FLAGS.seed)

 
  #DP version with providing epsilon and computing the corresponding sigma
  #Note: This uses autodp package
  if FLAGS.dpsgd:
    #code from: https://github.com/yuxiangw/autodp/blob/master/tutorials/tutorial_calibrator.ipynb
    dp_params = {}
    general_calibrate = generalized_eps_delta_calibrator()
    dp_params['sigma'] = None
    dp_params['coeff'] = FLAGS.epochs * num_train / FLAGS.batch_size
    dp_params['prob'] = FLAGS.batch_size / num_train
    mech = general_calibrate (SubsampleGaussianMechanism, FLAGS.epsilon, FLAGS.delta, [0,1000], params=dp_params, para_name='sigma', name='Subsampled_Gaussian')
  #  print("For epsilon={} and delta={}:  sigma={}".format{FLAGS.epsilon, FLAGS.delta, dp_params['sigma']})
    print("SIGMA WITH AUTODP: ", dp_params['sigma'])

  def data_stream():
    rng = npr.RandomState(FLAGS.seed)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
        yield train_images[batch_idx], Y_TRAIN[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.adam(FLAGS.learning_rate)

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch, y_support):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(
        i,
        private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                     dp_params['sigma'], FLAGS.batch_size, y_support, kernel_fn, FLAGS.microbatches, FLAGS.reg), opt_state)

  """Initialize distilled images as random original ones"""
  _, labels_init, init_params, y_init = class_balanced_sample(FLAGS.support_size, LABELS_TRAIN, train_images, Y_TRAIN, seed=FLAGS.seed)
  """Initialize distilled images as N(0,1)"""
  if FLAGS.dataset == 'mnist':
    init_params, y_init = shape_as_image(init_params, y_init, FLAGS.dataset)
    #print('init_params.shape after reshape=',init_params.shape)
  elif  FLAGS.dataset == 'fashion_mnist':
    init_params=random.normal(key, (FLAGS.support_size, train_images.shape[1], train_images.shape[2], train_images.shape[3])) 
  else:
    if FLAGS.random_init:
      """Initialize distilled images as random images"""
      print("Initialize from random original images")
    else: 
      """Initialize images from N(0,1)"""
      print("Initialize from N(0,1)")
      init_params=random.normal(key, (FLAGS.support_size, train_images.shape[1], train_images.shape[2], train_images.shape[3])) 
  print('init_params.shape=',init_params.shape)
 
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  print('\nStarting training...')
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    for _ in range(num_batches):
      if FLAGS.dpsgd:
        #delete_live_buffers()
        gc.collect()
        opt_state = \
            private_update(
                key, next(itercount), opt_state,
                shape_as_image(*next(batches), FLAGS.dataset, dummy_dim=True), y_init)

      else:
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(*next(batches)).shape)
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch} in {epoch_time:0.2f} sec')

    # evaluate test accuracy
    params = get_params(opt_state)
    print("params.shape=", params.shape)
    mse_loss, acc = eval_acc(params, y_init, test_images, test_labels, kernel_fn, FLAGS.reg)
    print("test loss:", mse_loss)
    print("test acc: ", acc)

  print("params.shape=", params.shape)
  """plot generated images"""

  params_final_x, params_init_raw_y=shape_as_image(params, labels_init, FLAGS.dataset)

  print("params_final_x=", params_final_x.shape)
  print("params_init_raw_y=", params_init_raw_y)

  _, _, sample_raw, sample_init, sample_final = class_balanced_sample(10,  params_init_raw_y, init_params, init_params, params_final_x, seed=FLAGS.seed)

  if FLAGS.dataset == 'mnist':
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    sample_raw= sample_raw * np.float32(255.) #undo preprocess step in data loading
  else: 
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  cur_path = os.path.dirname(os.path.abspath(__file__))
  save_path=os.path.join(cur_path, 'result_images')
  save_acc_path=os.path.join(cur_path, 'accuracy')
  save_data_path=os.path.join(cur_path, 'distilled_images')
  
  #Save accuracy results
  filename_acc="acc_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_zca={}_initimgs={}.txt".format(FLAGS.dataset, FLAGS.architecture, FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, FLAGS.reg, FLAGS.seed, FLAGS.zca, FLAGS.random_init)
  if not os.path.exists(save_acc_path):
    os.makedirs(save_acc_path)
  with open(os.path.join(save_acc_path, filename_acc), 'w') as f:
    f.writelines(str(acc))

  #Save distilled images
  filename_data="distilled_data_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_zca={}_initimgs={}.npz".format(FLAGS.dataset, FLAGS.architecture, FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, FLAGS.reg, FLAGS.seed,  FLAGS.zca, FLAGS.random_init)
  if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)
  np.savez(os.path.join(save_data_path, filename_data), data=params_final_x, labels=params_init_raw_y)

  #Save plotted images
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  fig = plt.figure(figsize=(33,10))
  fig.suptitle('Image comparison.\n\nRow 1: Original uint8.  Row2: Original normalized.  Row 3: KIP learned images.', fontsize=16, y=1.02)
  for i, img in enumerate(sample_raw):
    ax = plt.subplot(3, 10, i+1)
    ax.set_title(class_names[i])
    plt.imshow(np.squeeze(img))

  for i, img in enumerate(sample_init, 1):
    plt.subplot(3, 10, 10+i)
    plt.imshow(np.squeeze(img))

  for i, img in enumerate(sample_final, 1):
    plt.subplot(3, 10, 20+i)
    plt.imshow(np.squeeze(img))

  filename="distilled_img_{}_{}_supp_size={}_eps={}_delta={}_lr={}_c={}_bs={}_epochs={}_reg={}_seed={}_zca={}_initimgs={}.png".format(FLAGS.dataset, FLAGS.architecture, FLAGS.support_size, FLAGS.epsilon, FLAGS.delta, FLAGS.learning_rate, FLAGS.l2_norm_clip, FLAGS.batch_size, FLAGS.epochs, FLAGS.reg, FLAGS.seed,  FLAGS.zca, FLAGS.random_init)
  plt.savefig(os.path.join(save_path, filename), format="png")

if __name__ == '__main__':
  app.run(main)
