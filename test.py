from collections import defaultdict
import cv2
import ipdb
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor

from config import *
from nets.obj_feat_att_net import VideoCaptionGenerator
from utils import get_video_data, preProBuildWordVocab


def test(model_path='models/model-61', video_feat_path=video_feat_path):
    train_data, test_data = get_video_data(video_data_path, video_feat_path, train_ratio=0.7)
    test_videos = test_data['video_path'].values
    test_captions = test_data['Description'].values
    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    test_videos_unique = list()
    test_captions_list = list()
    for (video, caption) in zip(test_videos, test_captions):
        if len(test_videos_unique) == 0 or test_videos_unique[-1] != video:
            test_videos_unique.append(video)
            test_captions_list.append([caption])
        else:
            test_captions_list[-1].append(caption)

    model = VideoCaptionGenerator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_embed=dim_embed,
            dim_hidden=dim_hidden,
            batch_size=1,
            dim_obj_feats = dim_obj_feats,
            n_obj_feats = n_obj_feats,
            encoder_max_sequence_length=encoder_step,
            decoder_max_sentence_length=decoder_step,
            bias_init_vector=None)

    tf_loss, tf_video_mask, tf_obj_feats, tf_caption, tf_caption_mask, tf_probs, tf_generated_words, tf_generated_att = model.build_model(is_test=True)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    scorer = Meteor()
    scorer_bleu = Bleu(4)
    GTS = defaultdict(list)
    RES = defaultdict(list)
    counter = 0

    for (vid, caption) in zip(test_videos_unique, test_captions_list):
        # Collect frames.
        cap = cv2.VideoCapture(os.path.join(video_path, vid[:-4]))
        frames = list()
        while True:
            ret, im = cap.read()
            if ret is False:
                break
            frames.append(im)

        # Load meta data.
        with open(os.path.join(meta_data_path, vid[:-4]+'.txt'), 'r') as f:
            meta_data = json.load(f)
            all_feats = meta_data['features']

        generated_sentence, generated_att = gen_sentence(
            sess, tf_video_mask, tf_obj_feats, tf_generated_words, tf_generated_att, vid, ixtoword)
        #generated_sentence_test, weights = gen_sentence(
        #    sess, video_tf, video_mask_tf, caption_tf, vid, ixtoword, weights_tf, 0.3)
        generated_att = [att[:, 0, 0] for att in generated_att]
        #print generated_att

        print vid, generated_sentence
        #plt.plot(generated_att)
        #plt.show()
        #print generated_sentence_test
        #print caption

        words = generated_sentence.split(' ')
        feats = list()
        for i, w in enumerate(words):
            i_best_feat = np.argmax(generated_att[i])
            print w, i_best_feat
            if i_best_feat < len(all_feats):
                feat = all_feats[i_best_feat]   
                i_frame = feat[0]
                bbox = feat[2]
                im = frames[i_frame][bbox[2]:bbox[3], bbox[0]:bbox[1]]
                cv2.imshow('test', im)
                cv2.waitKey(10000)

        GTS[str(counter)] = [{'image_id':str(counter),'cap_id':i,'caption':s} for i, s in enumerate(caption)]
        RES[str(counter)] = [{'image_id':str(counter),'caption':generated_sentence[:-2]+'.'}]

        #GTS[vid] = caption
        #RES[vid] = [generated_sentence[:-2] + '.']
        counter += 1
        
        #words = generated_sentence.split(' ')
        #fig = plt.figure()
        #for i in range(len(words)):
        #    w = weights[i]
        #    ax = fig.add_subplot(len(words), 1, i+1)
        #    ax.set_title(words[i])
        #    ax.plot(range(len(w)), [ww[0] for ww in w], 'b')
        #plt.show()

        #ipdb.set_trace()

    tokenizer = PTBTokenizer()
    GTS = tokenizer.tokenize(GTS)
    RES = tokenizer.tokenize(RES)

    score, scores = scorer.compute_score(GTS, RES)
    print "METEOR", score
    score, scores = scorer_bleu.compute_score(GTS, RES)
    print "BLEU", score

    #ipdb.set_trace()


def gen_sentence(sess, tf_video_mask, tf_obj_feats, tf_generated_words, tf_generated_att, vid, ixtoword):
    #video_feat = np.zeros((1, encoder_step, dim_image))
    #video_mask = np.zeros((video_feat.shape[0], video_feat.shape[1]))

    #feat = np.load(os.path.join(video_feat_path, vid))[None, ...]
    #video_feat[0, :feat.shape[1], :] = feat
    #video_mask[:feat.shape[1], :] = 1

    #current_feats_vals = map(lambda vid: np.load(os.path.join(video_feat_path, vid)), current_videos)
    # Object features.
    obj_feats = np.zeros((1, n_obj_feats, dim_obj_feats))
    feat = np.load(os.path.join(video_obj_feat_path, vid))
    n_obj = min(n_obj_feats, feat.shape[0])
    obj_feats[0, :n_obj] = feat[:n_obj]

    video_mask = np.zeros((1, n_obj_feats))
    video_mask[0, :n_obj] = 1

    #interval_frame = video_feat.shape[1]/encoder_step
    #video_feat = video_feat[:, range(0, encoder_step*interval_frame, interval_frame), :]
    #video_feat = sampling(video_feat, sampling_rate)

    generated_word_index, generated_att = sess.run(
        [tf_generated_words, tf_generated_att],
        feed_dict={
            #video_tf: video_feat,
            tf_video_mask: video_mask,
            tf_obj_feats: obj_feats,
        })
    #probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
    #embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
    generated_words = ixtoword[generated_word_index]

    punctuation = np.argmax(np.array(generated_words) == '.')+1
    generated_words = generated_words[:punctuation]

    generated_sentence = ' '.join(generated_words)
    return generated_sentence, generated_att


if __name__=="__main__":
    test(model_path='models/model-799')
