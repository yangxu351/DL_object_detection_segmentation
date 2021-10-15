from pickle import FALSE


CMT = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'

train_syn = True
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.05 
### with pixel attention branch, 
WITH_PA = False # True False,  Pixel attention
### with FPN Mask multiply
WITH_FPN_MASK = False # True False,  
### with RPN Mask multiply
WITH_RPN_MASK = True # True False
 
# SOFT_VAL = 1 # 0.5 0.1 1 
SOFT_VAL = -1

DEVICE= 'cuda:2'

    
