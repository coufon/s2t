#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
from collections import defaultdict

from tensorflow.contrib import rnn
from keras.preprocessing import sequence

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor

############### Global Parameters ###############
video_path = './dataset/youtube_videos'
video_data_path='./data/video_corpus.csv'
video_feat_path = './dataset/youtube_feats'
video_obj_feat_path = './dataset/youtube_feats_resnet'

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


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_embed, dim_hidden, batch_size,
            dim_obj_feats, n_obj_feats, encoder_max_sequence_length, decoder_max_sentence_length,
            bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.dim_obj_feats = dim_obj_feats
        self.n_obj_feats = n_obj_feats
        self.encoder_max_sequence_length = encoder_max_sequence_length
        self.decoder_max_sentence_length = decoder_max_sentence_length

        #with tf.device("/cpu:0"):
        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')

        # Image encoder LSTM input.
        #self.encoder_lstm_W = tf.Variable(tf.random_uniform([dim_embed, dim_hidden], -0.1, 0.1), name='encoder_lstm_W')
        #self.encoder_lstm_b = tf.Variable(tf.zeros([dim_hidden]), name='encoder_lstm_b')

        # Obj feature encoder LSTM input.
        #self.encoder_obj_lstm_W = tf.Variable(tf.random_uniform([dim_embed, dim_hidden], -0.1, 0.1), name='encoder_obj_lstm_W')
        #self.encoder_obj_lstm_b = tf.Variable(tf.zeros([dim_hidden]), name='encoder_obj_lstm_b')

        # Embed image feature.
        #self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, dim_embed], -0.1, 0.1), name='embed_image_W')
        #self.embed_image_b = tf.Variable(tf.zeros([dim_embed]), name='embed_image_b')

        # Embed obj features.
        self.embed_obj_W = tf.Variable(tf.random_uniform([dim_obj_feats, dim_embed], -0.1, 0.1), name='embed_obj_W')
        self.embed_obj_b = tf.Variable(tf.zeros([dim_embed]), name='embed_obj_b')        

        # Embed decoder inputs.
        # TODO(fangzhou): it maybe unnecessary.
        #self.embed_decoder_W = tf.Variable(tf.random_uniform([dim_embed+dim_hidden*2, dim_embed], -0.1, 0.1), name='embed_decoder_W')
        #self.embed_decoder_b = tf.Variable(tf.zeros([dim_embed]), name='embed_decoder_b')    

        # Embed words.
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        # Attention of image features.
        #self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1, 0.1), name='embed_att_w')
        #self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='embed_att_Wa')
        #self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1, 0.1), name='embed_att_Ua')
        #self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')       

        # Attention of object features.
        self.obj_embed_att_w = tf.Variable(tf.random_uniform([dim_embed, 1], -0.1, 0.1), name='obj_embed_att_w')
        self.obj_embed_att_Wa = tf.Variable(tf.random_uniform([dim_embed, dim_embed], -0.1, 0.1), name='obj_embed_att_Wa')
        self.obj_embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_embed],-0.1, 0.1), name='obj_embed_att_Ua')
        self.obj_embed_att_ba = tf.Variable( tf.zeros([dim_embed]), name='obj_embed_att_ba')


    def build_model(self, is_test=False):
        # Average of all obj features.
        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            return tf.cast(tf.reduce_sum(used, 1), tf.int32) 
 
        batch_size = 1 if is_test else self.batch_size

        #video = tf.placeholder(tf.float32, [batch_size, self.encoder_max_sequence_length, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [batch_size, self.n_obj_feats])
        obj_feats = tf.placeholder(tf.float32, [batch_size, self.n_obj_feats, self.dim_obj_feats])

        caption = tf.placeholder(tf.int32, [batch_size, self.decoder_max_sentence_length])
        caption_mask = tf.placeholder(tf.float32, [batch_size, self.decoder_max_sentence_length])

        # Build model.
        generated_words = list()
        probs = list()
        loss = 0.0

        # Input feature of attention LSTM.
        # Embed obj features.
        obj_feats_flat = tf.reshape(obj_feats, [-1, self.dim_obj_feats])
        obj_embs = tf.nn.xw_plus_b(obj_feats_flat, self.embed_obj_W, self.embed_obj_b)
        obj_embs = tf.reshape(obj_embs, [batch_size, self.n_obj_feats, self.dim_embed])

        # Average features.
        avg_obj_feats = tf.reduce_sum(obj_embs, 1)/tf.reshape(
            tf.cast(length(obj_embs), tf.float32), (-1, 1))

        # Projected features for attention.
        obj_emb_projs = [
            tf.nn.xw_plus_b(obj_embs[:, i, :], self.obj_embed_att_Wa, self.obj_embed_att_ba) \
                for i in range(obj_embs.shape[1])]

        # LSTMs.
        att_lstm = rnn.BasicLSTMCell(num_units=dim_hidden, state_is_tuple=True)
        decoder = rnn.BasicLSTMCell(num_units=dim_hidden, state_is_tuple=True)
        state_att_lstm = att_lstm.zero_state(self.batch_size, dtype=tf.float32)
        state_decoder = decoder.zero_state(self.batch_size, dtype=tf.float32)
        output_decoder = state_decoder.h

        for i in range(self.encoder_max_sequence_length):
            with tf.variable_scope("Decoder"):
                # Input Att LSTM: Previously word.
                if i == 0:
                    prev_word_embed = tf.zeros([batch_size, self.dim_embed])
                else:
                    tf.get_variable_scope().reuse_variables()
                    if is_test:
                        prev_word_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                        prev_word_embed = tf.expand_dims(prev_word_embed, 0)
                    else:
                        prev_word_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i-1])

            with tf.variable_scope("Attention"):
                # Attention LSTM.
                (output_att_lstm, state_att_lstm) = att_lstm(
                    tf.concat([output_decoder, avg_obj_feats, prev_word_embed], 1), state_att_lstm)

                # Attention.
                state_proj = tf.matmul(output_att_lstm, self.obj_embed_att_Ua)
                e_list = tf.stack([
                    tf.matmul(tf.tanh(tf.add(obj_emb_proj, state_proj)), self.obj_embed_att_w) \
                        for obj_emb_proj in obj_emb_projs], axis=0)
                weights = tf.nn.softmax(e_list, dim=0)

                emb_weighted = [
                    tf.multiply(emb, tf.tile(weight, [1, dim_embed])) \
                        for weight, emb in zip(tf.unstack(weights, axis=0), tf.unstack(obj_embs, axis=1))]
                emb_weighted_sum = tf.reduce_sum(emb_weighted, 0)

            with tf.variable_scope("Decoder"):
                # Word decoder LSTM.
                (output_decoder, state_decoder) = decoder(
                    tf.concat([emb_weighted_sum, output_att_lstm], 1), state_decoder)

                logit_words = tf.nn.xw_plus_b(output_decoder, self.embed_word_W, self.embed_word_b)
                probs.append(logit_words)

                if is_test:
                    max_prob_index = tf.argmax(logit_words, 1)[0]
                    generated_words.append(max_prob_index)
                else:
                    labels = tf.expand_dims(caption[:, i], 1)
                    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, self.n_words]), 1.0, 0.0)
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                    cross_entropy = cross_entropy * caption_mask[:, i]
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss += current_loss

        if not is_test:
            loss = loss / tf.reduce_sum(caption_mask)
        return loss, video_mask, obj_feats, caption, caption_mask, probs, generated_words


def get_video_data(video_data_path, video_feat_path, train_ratio=0.7):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    #video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    #video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(os.path.join(video_feat_path, x)))]
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(os.path.join(video_obj_feat_path, x)))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
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


def train(prev_model_path=None):
    train_data, _ = get_video_data(video_data_path, video_feat_path, train_ratio=0.7)
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=10)

    np.save('./data/ixtoword', ixtoword)

    model = Video_Caption_Generator(
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

    tf_loss, tf_video_mask, tf_obj_feats, tf_caption, tf_caption_mask, tf_probs, _ = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=5)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    if prev_model_path is not None:
        saver.restore(sess, prev_model_path)

    for epoch in range(n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        current_train_data = train_data.ix[index]

        current_train_data = train_data.groupby('video_path').apply(
            lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)

        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values
            # Frame feature.
            #current_feats = np.zeros((batch_size, encoder_step, dim_image))
            #current_feats_vals = map(lambda vid: np.load(os.path.join(video_feat_path, vid)), current_videos)
            # Object features.
            current_obj_feats = np.zeros((batch_size, n_obj_feats, dim_obj_feats))
            current_obj_feats_vals = map(lambda vid: np.load(os.path.join(video_obj_feat_path, vid)), current_videos)

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

            current_captions = current_batch['Description'].values
            for idx, cc in enumerate( current_captions ):
                current_captions[idx] = cc.replace('.', '').replace(',', '')

            current_captions_ind  = map(
                lambda cap : [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix],
                current_captions)

            current_caption_matrix = sequence.pad_sequences(current_captions_ind, padding='post', maxlen=decoder_step-1, value=0)
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

    model = Video_Caption_Generator(
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

    tf_loss, tf_video_mask, tf_obj_feats, tf_caption, tf_caption_mask, tf_probs, tf_generated_words = model.build_model(is_test=True)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    scorer = Meteor()
    scorer_bleu = Bleu(4)
    GTS = defaultdict(list)
    RES = defaultdict(list)
    counter = 0

    for (vid, caption) in zip(test_videos_unique, test_captions_list):
        generated_sentence = gen_sentence(
            sess, tf_video_mask, tf_obj_feats, tf_generated_words, vid, ixtoword)
        #generated_sentence_test, weights = gen_sentence(
        #    sess, video_tf, video_mask_tf, caption_tf, vid, ixtoword, weights_tf, 0.3)

        print vid, generated_sentence
        #print generated_sentence_test
        #print caption

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


def gen_sentence(sess, tf_video_mask, tf_obj_feats, tf_generated_words, vid, ixtoword):
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

    generated_word_index = sess.run(
        tf_generated_words,
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
    return generated_sentence


def sampling(video_feat, sampling_rate):
    interval = int(1/sampling_rate)
    new_feat = video_feat[:, range(0, video_feat.shape[1], interval), :]
    padding = np.zeros(
        (video_feat.shape[0], video_feat.shape[1] - new_feat.shape[1], video_feat.shape[2]))
    return np.concatenate((new_feat, padding), axis=1)


if __name__=="__main__":
    #test(model_path='models/model-2999')
    train(prev_model_path='models/model-2999')
