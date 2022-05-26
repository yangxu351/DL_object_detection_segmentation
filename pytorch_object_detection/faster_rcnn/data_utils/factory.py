# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}

from data_utils.syn_nwpu import syn_nwpu
from data_utils.real_nwpu import real_nwpu

import numpy as np

for split in ['train','val']:
  dataset = 'syn_nwpu_c1_{}'.format(split).upper()
  __sets[dataset] = (lambda name=dataset: syn_nwpu(name))
  # print('__sets',[x for x in __sets.values()])

for split in ['test']:
  dataset = 'real_nwpu_c1_{}'.format(split).upper()
  __sets[dataset] = (lambda name=dataset: real_nwpu(name))  

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
