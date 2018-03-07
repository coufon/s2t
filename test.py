from collections import defaultdict
import cv2
import ipdb
import json
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor

from config.msrvtt_config import *
from nets.obj_feat_att_net import VideoCaptionGenerator
from utils.msrvtt_utils import get_video_data, preProBuildWordVocab


def test(model_path='models/model-61'):
    captions = get_video_data(video_data_path_test, video_feat_path_test, is_test=True)
    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    model = VideoCaptionGenerator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_embed=dim_embed,
            dim_hidden=dim_hidden,
            batch_size=1,
            dim_obj_feats = dim_obj_feats,
            n_obj_feats = n_obj_feats,
            #encoder_max_sequence_length=encoder_step,
            decoder_max_sentence_length=decoder_step,
            bias_init_vector=None)

    _, tf_obj_feats, tf_video_mask, _, _, tf_generated_words, tf_generated_att = \
        model.build_model(is_test=True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    scorer = Meteor()
    scorer_bleu = Bleu(4)
    GTS = defaultdict(list)
    RES = defaultdict(list)
    counter = 0

    for vid, caption in captions.items():
        print counter
        if False:
            # Collect frames.
            cap = cv2.VideoCapture(os.path.join(video_path_test, vid+'.mp4'))
            frames = list()
            while True:
                ret, im = cap.read()
                if ret is False:
                    break
                frames.append(im)

        # Load meta data.
        #with open(os.path.join(meta_data_path_test, vid+'.mp4.txt'), 'r') as f:
        #    meta_data = json.load(f)
        #    all_feats = meta_data['features']

        generated_sentence, generated_att, _ = gen_sentence(
            sess, tf_video_mask, tf_obj_feats, tf_generated_words, tf_generated_att, vid, ixtoword)
        #generated_sentence_test, weights = gen_sentence(
        #    sess, video_tf, video_mask_tf, caption_tf, vid, ixtoword, weights_tf, 0.3)
        generated_att = [att[:, 0, 0] for att in generated_att]
        #print generated_att

        print vid, generated_sentence[:-2]
        #plt.plot(generated_att)
        #plt.show()
        #print generated_sentence_test
        #print caption

        if False:
            words = generated_sentence.split(' ')
            feats = list()
            for i, w in enumerate(words):
                i_best_feat_list = np.argsort(generated_att[i])[::-1]
                imgs = list()
                for i_best_feat in i_best_feat_list:
                    weight = generated_att[i][i_best_feat]
                    if weight < 0.1:
                        break
                    print w, i_best_feat
                    if all_feats is None or len(all_feats) == 0:
                        im = cv2.resize(frames[:len(frames):len(frames)/4][i_best_feat], (300, 300))
                    else:
                        feat = all_feats[i_best_feat]   
                        i_frame = feat[0]
                        bbox = feat[2]
                        im = np.copy(frames[i_frame][bbox[2]:bbox[3], bbox[0]:bbox[1]])
                        im = cv2.resize(im, (300, 300))
                    constant=cv2.copyMakeBorder(im,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
                    violet= np.zeros((30, constant.shape[1], 3), np.uint8)
                    violet[:] = (255, 255, 255)
                    vcat = cv2.vconcat((violet, constant))
                    cv2.putText(vcat, str(weight),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, 0)
                    imgs.append(vcat)
                if imgs:
                    final_img = cv2.hconcat(imgs)
                    cv2.imshow('test', final_img)
                    cv2.waitKey(10000)

        GTS[str(counter)] = [{'image_id':str(counter),'cap_id':i,'caption':' '.join(s)} for i, s in enumerate(caption)]
        RES[str(counter)] = [{'image_id':str(counter),'caption':generated_sentence[:-2]}]

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
    feat = np.load(os.path.join(video_feat_path_test, vid+'.mp4.npy'))
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
    return generated_sentence, generated_att, video_mask


if __name__=="__main__":
    test(model_path=init_model_path)
