############### Global Parameters ###############
video_path = '/home/eric/Videos/MSR-VTT/train-video'
video_data_path_train = '/home/eric/Videos/MSR-VTT/videodatainfo_2017_ustc.json'
video_feat_path_train = '/home/eric/Videos/MSR-VTT/features'
meta_data_path_train = '/home/eric/Videos/MSR-VTT/metadata'

video_data_path_test = '/home/eric/Videos/MSR-VTT/videodatainfo_2017_ustc.json'
video_feat_path_test = '/home/eric/Videos/MSR-VTT/features'

model_path = './models/'

############## Train Parameters #################
dim_image = 4096
dim_embed = 512
dim_hidden= 1024
dim_obj_feats = 2048
n_obj_feats = 32 # max
encoder_step = n_obj_feats
decoder_step = n_obj_feats
n_epochs = 3000
batch_size = 128
chunk_len = 8
learning_rate = 0.0001
##################################################