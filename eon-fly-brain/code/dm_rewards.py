"""Soft indicator function evaluating whether a number is within bounds.

Vendored verbatim (`tolerance` and its `_sigmoids` helper only) from
dm_control/utils/rewards.py, google-deepmind/dm_control @ main, retrieved
2026-09-03. Apache-2.0, copyright 2017 The dm_control Authors; the license
header below is theirs and is reproduced as required.

Vendored rather than imported because dm_control is a large dependency and
this project needs exactly one function from it. `flybody` builds every task
reward out of this function (`dm_control.utils.rewards.tolerance`), so using
the identical implementation is what makes our reward a faithful port of
theirs rather than a lookalike.
"""

# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import warnings

import numpy as np

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
  """Returns 1 when `x` == 0, between 0 and 1 otherwise."""
  if sigmoid in ('cosine', 'linear', 'quadratic'):
    if not 0 <= value_at_1 < 1:
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid == 'gaussian':
    scale = np.sqrt(-2 * np.log(value_at_1))
    return np.exp(-0.5 * (x*scale)**2)

  elif sigmoid == 'hyperbolic':
    scale = np.arccosh(1/value_at_1)
    return 1 / np.cosh(x*scale)

  elif sigmoid == 'long_tail':
    scale = np.sqrt(1/value_at_1 - 1)
    return 1 / ((x*scale)**2 + 1)

  elif sigmoid == 'reciprocal':
    scale = 1/value_at_1 - 1
    return 1 / (abs(x)*scale + 1)

  elif sigmoid == 'cosine':
    scale = np.arccos(2*value_at_1 - 1) / np.pi
    scaled_x = x*scale
    with warnings.catch_warnings():
      warnings.filterwarnings(
          action='ignore', message='invalid value encountered in cos')
      cos_pi_scaled_x = np.cos(np.pi*scaled_x)
    return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)

  elif sigmoid == 'linear':
    scale = 1-value_at_1
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

  elif sigmoid == 'quadratic':
    scale = np.sqrt(1-value_at_1)
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

  elif sigmoid == 'tanh_squared':
    scale = np.arctanh(np.sqrt(1-value_at_1))
    return 1 - np.tanh(x*scale)**2

  else:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise."""
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = np.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = np.where(in_bounds, 1.0, 0.0)
  else:
    d = np.where(x < lower, lower - x, x - upper) / margin
    value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return float(value) if np.isscalar(x) else value
