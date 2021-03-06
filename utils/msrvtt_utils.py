import json
import nltk
import numpy as np
import os
import pandas as pd

from config import *


def get_video_data(video_data_path, video_feat_path, is_test=False):
    captions = dict()
    with open(video_data_path, 'r') as f:
        video_data = json.load(f)
        if not is_test:
            all_videos = set([v['video_id'] for v in video_data['videos'][:6513]])
        else:
            all_videos = set([v['video_id'] for v in video_data['videos'][7010:]])
       	all_sentences = video_data['sentences']
        for sentence in all_sentences:
            video_id = sentence['video_id']
            if not video_id in all_videos:
                continue
            if not video_id in captions:
                captions[video_id] = list()

            sent_processed = sentence['caption'].replace('.', '').replace(',', '').replace('\n', '').replace('\r', '').encode('ascii', 'ignore').decode('ascii').lower()
            captions[video_id].append(nltk.tokenize.word_tokenize(sent_processed))
    return captions


# borrowed this function from NeuralTalk
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % \
        (word_count_threshold, )
    word_counts = dict()
    nsents = 0
    for _, sents in sentence_iterator.items():
        for sent in sents:
            nsents += 1
            #for w in sent.lower().split(' '):
            for w in sent:
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {0: '.'} # period at the end of the sentence. make first dimension be end token.
    wordtoix = {'#START#': 0} # make first vector be the start token.
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector
