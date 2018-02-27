from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from config.msrvtt_config import *
from nets.obj_feat_att_net import VideoCaptionGenerator
from utils.msrvtt_utils import get_video_data, preProBuildWordVocab


def train(prev_model_path=None):
    captions = get_video_data(video_data_path_train, video_feat_path_train)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=10)

    np.save('./data/ixtoword', ixtoword)

    model = VideoCaptionGenerator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_embed=dim_embed,
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            dim_obj_feats = dim_obj_feats,
            n_obj_feats = n_obj_feats,
            encoder_max_sequence_length=encoder_step,
            decoder_max_sentence_length=decoder_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video_mask, tf_obj_feats, tf_caption, tf_caption_mask, tf_probs, _, _ = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=5)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    if prev_model_path is not None:
        saver.restore(sess, prev_model_path)

    for epoch in range(n_epochs):
        # Select one sentence randomly.
        current_videos = list()
        current_sents = list()
        for video_id, sents in captions.items():
            current_videos.append(video_id)
            current_sents.append(sents[np.random.choice(len(sents))])
        len_current_videos = len(current_videos)

        for start,end in zip(range(0, len_current_videos, batch_size),
                range(batch_size, len_current_videos, batch_size)):
            batch_videos = current_videos[start:end]
            batch_sents = current_sents[start:end]

            print batch_videos

            # Frame feature.
            #current_feats = np.zeros((batch_size, encoder_step, dim_image))
            #current_feats_vals = map(lambda vid: np.load(os.path.join(video_feat_path, vid)), current_videos)

            # Object features.
            current_obj_feats = np.zeros((batch_size, n_obj_feats, dim_obj_feats))
            current_obj_feats_vals = map(
                lambda vid: np.load(os.path.join(video_feat_path_train, vid+'.mp4.npy')), batch_videos)
            current_video_masks = np.zeros((batch_size, n_obj_feats))

            #for ind, feat in enumerate(current_feats_vals):
                #interval_frame = max(feat.shape[0]/n_frame_step, 1)
                #current_feats[ind][:len(current_feats_vals[ind])] = feat[
                #    range(0, min(n_frame_step*interval_frame, max(feat.shape[0]), interval_frame), :]
            #    current_feats[ind][:len(current_feats_vals[ind])] = feat
            #    current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            for ind, feat in enumerate(current_obj_feats_vals):
                if feat is not None and len(feat.shape) == 2:
                    n_obj = min(n_obj_feats, feat.shape[0])
                    current_obj_feats[ind][:n_obj] = feat[:n_obj]
                    current_video_masks[ind][:n_obj] = 1

            #for idx, cc in enumerate(batch_sents):
            #    current_captions[idx] = cc.replace('.', '').replace(',', '')

            current_captions_ind  = map(
                lambda cap : [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix],
                batch_sents)

            current_caption_matrix = sequence.pad_sequences(
                current_captions_ind, padding='post', maxlen=decoder_step-1, value=0)
            current_caption_matrix = np.hstack(
                [current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)

            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(map(lambda x: (x != 0).sum()+1, current_caption_matrix))
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            #probs_val = sess.run(tf_probs, feed_dict={
            #    tf_video:current_feats,
            #    tf_caption: current_caption_matrix
            #    })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        #tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_obj_feats: current_obj_feats,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                    })

            print loss_val
        if np.mod(epoch+1, 100) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


if __name__=="__main__":
    train(prev_model_path=None)
