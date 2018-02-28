############### Global Parameters ###############
video_path = '/home/eric/Videos/youtube_videos'
video_data_path='./data/video_corpus.csv'
video_feat_path = '/home/eric/Videos/youtube_feats'
video_obj_feat_path = '/home/eric/Videos/youtube_feats_resnet'
meta_data_path = '/home/eric/Videos/youtube_feats_meta'

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
learning_rate = 0.00005
##################################################