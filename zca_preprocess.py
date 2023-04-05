import jax
from absl import logging
import jax.numpy as jnp

def apply_preprocess(x_train, x_test, zca_reg, zca_reg_abs_scale):
  """Apply ZCA preprocessing on the standard normalized data."""
  
  preprocess_type = "zca_normalize"
  if preprocess_type == 'standard':
    # Normalization is already done.
    pass
  else:

    if preprocess_type == 'zca_normalize':
        preprocess_op = _get_preprocess_op(
          x_train,
          layer_norm=True,
          zca_reg=zca_reg,
          zca_reg_absolute_scale= zca_reg_abs_scale)
        x_train = preprocess_op(x_train)
        x_test = preprocess_op(x_test)
    else:
      NotImplementedError('Preprocess type %s is not implemented' %
                          preprocess_type)

  return x_train, x_test


def _get_preprocess_op(x_train,
                      layer_norm=True,
                      zca_reg=1e-5,
                      zca_reg_absolute_scale=False,
                      on_cpu=False):
  """ZCA preprocessing function."""
  whitening_transform = _get_whitening_transform(x_train, layer_norm, zca_reg,
                                                zca_reg_absolute_scale,
                                                on_cpu)

  def _preprocess_op(images):
    orig_shape = images.shape
    images = images.reshape(orig_shape[0], -1)
    if layer_norm:
      # Zero mean every feature
      images = images - jnp.mean(images, axis=1)[:, jnp.newaxis]
      # Normalize
      image_norms = jnp.linalg.norm(images, axis=1)
      # Make features unit norm
      images = images / image_norms[:, jnp.newaxis]

    images = (images).dot(whitening_transform)
    images = images.reshape(orig_shape)
    return images

  return _preprocess_op


def _get_whitening_transform(x_train,
                             layer_norm=True,
                             zca_reg=1e-5,
                             zca_reg_absolute_scale=False,
                             on_cpu=False):
  """Returns 2D matrix that performs whitening transform.

  Whitening transform is a (d,d) matrix (d = number of features) which acts on
  the right of a (n, d) batch of flattened data.
  """
  orig_train_shape = x_train.shape
  x_train = x_train.reshape(orig_train_shape[0], -1).astype('float64')
  if on_cpu:
    x_train = jax.device_put(x_train, jax.devices('cpu')[0])

  n_train = x_train.shape[0]
  if layer_norm:
    logging.info('Performing layer norm preprocessing.')
    # Zero mean every feature
    x_train = x_train - jnp.mean(x_train, axis=1)[:, jnp.newaxis]
    # Normalize
    train_norms = jnp.linalg.norm(x_train, axis=1)
    # Make features unit norm
    x_train = x_train / train_norms[:, jnp.newaxis]

  logging.info('Performing zca whitening preprocessing with reg: %.2e', zca_reg)
  cov = 1.0 / n_train * x_train.T.dot(x_train)
  if zca_reg_absolute_scale:
    reg_amount = zca_reg
  else:
    reg_amount = zca_reg * jnp.trace(cov) / cov.shape[0]
  logging.info('Raw zca regularization strength: %f', reg_amount)

  u, s, _ = jnp.linalg.svd(cov + reg_amount * jnp.eye(cov.shape[0]))
  inv_sqrt_zca_eigs = s**(-1 / 2)

  # rank control
  if n_train < x_train.shape[1]:
    inv_sqrt_zca_eigs = inv_sqrt_zca_eigs.at[n_train:].set(
        jnp.ones(inv_sqrt_zca_eigs[n_train:].shape[0]))
  whitening_transform = jnp.einsum(
      'ij,j,kj->ik', u, inv_sqrt_zca_eigs, u, optimize=True)
  return whitening_transform