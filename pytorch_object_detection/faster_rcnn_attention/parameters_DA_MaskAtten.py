# BASE_DIR= '/media/lab/Yang'  # Thor
BASE_DIR= '/data/users/yang' # Groot

CMT = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
REAL_CMT = 'xilin_wdt'
DATA_SEED = 0

train_syn = True
EPOCHS = 50 # 20
# BATCH_SIZE = 8
BATCH_SIZE = 4
# LEARNING_RATE = 0.05

LEARNING_RATE = 1e-4
# LEARNING_RATE = 1e-3 # too large

MODEL_SEED = 0
### with pixel attention branch, 
WITH_PA = False # True False,  Pixel attention
### with FPN Mask multiply
WITH_FPN_MASK = False # True False,  
### with RPN Mask multiply
WITH_RPN_MASK = False # True False
# WITH_RPN_MASK = True
### with RPN Mask multiply
WITH_MASK_FEATURE = True # True False
LAYER_LEVELS = ['0', '1', '2', '3']

# WITH_LC = True # True
WITH_LC = False # False

WITH_GL = False # True
# WITH_CTX = True
WITH_CTX = False

WITH_EF = False
GAMMA = 0.1
# ETA = 0.5 # 0.1
ETA = 1 # 0.1

# DEVICE= 'cuda:1'
DEVICE= 'cuda:2'
# DEVICE= 'cuda:3'

# LEARNING_RATE=0.001 # for  for RPN mask without lc or gc

if WITH_RPN_MASK:
    # -1 represent random value in [0, 1)
    SOFT_VAL = -1
    # -0.5 represent random value in [0, 0.5)
    # SOFT_VAL = -0.5 
    # fixed value for background pixels
    # SOFT_VAL = 0.5 # 0.5 0.1 1 
    # SOFT_VAL = 0
elif WITH_MASK_FEATURE:
    SOFT_VAL = -1    
elif WITH_FPN_MASK:
    SOFT_VAL = -1 # 0.5 0.1 1 # -1 represent random value in [0, 1)
else:
    SOFT_VAL = 1 # do not directly multiply masks

ANNO_FORMAT = '.xml'
REAL_IMG_FORMAT = '.jpg'
SYN_IMG_FORMAT = '.png'

from easydict import EasyDict as edict
__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = LEARNING_RATE # 0.001
# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False