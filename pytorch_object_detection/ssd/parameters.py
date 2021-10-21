# BASE_DIR= '/media/lab/Yang'  # Thor
BASE_DIR= '/data/users/yang' # Groot

CMT = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
DATA_SEED = 0

train_syn = True
EPOCHS = 20 # 20
BATCH_SIZE = 8
LEARNING_RATE = 0.0005

DEVICE= 'cuda:3'

ANNO_FORMAT = '.xml'
REAL_IMG_FORMAT = '.jpg'
SYN_IMG_FORMAT = '.png'