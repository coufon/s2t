import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class VideoCaptionGenerator():
    def __init__(self, dim_image, n_words, dim_embed, dim_hidden, batch_size,
            dim_obj_feats, n_obj_feats, decoder_max_sentence_length,
            bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.dim_obj_feats = dim_obj_feats
        self.n_obj_feats = n_obj_feats
        #self.encoder_max_sequence_length = encoder_max_sequence_length
        self.decoder_max_sentence_length = decoder_max_sentence_length

        #with tf.device("/cpu:0"):
        # Embedding of words.
        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')

        # Decoder output to word.
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

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
        batch_size = 1 if is_test else self.batch_size

        def length(seq):
            used = tf.sign(tf.reduce_max(tf.abs(seq), 2))
            return tf.cast(tf.reduce_sum(used, 1), tf.int32) 

        # Inputs.
        #video = tf.placeholder(tf.float32, [batch_size, self.encoder_max_sequence_length, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [batch_size, self.n_obj_feats])
        obj_feats = tf.placeholder(tf.float32, [batch_size, self.n_obj_feats, self.dim_obj_feats])

        caption = tf.placeholder(tf.int32, [batch_size, self.decoder_max_sentence_length])
        caption_mask = tf.placeholder(tf.float32, [batch_size, self.decoder_max_sentence_length])

        # Build model.
        generated_words = list()
        generated_attention = list()
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
        att_lstm = rnn.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True)
        decoder = rnn.BasicLSTMCell(num_units=self.dim_hidden, state_is_tuple=True)
        state_att_lstm = att_lstm.zero_state(self.batch_size, dtype=tf.float32)
        state_decoder = decoder.zero_state(self.batch_size, dtype=tf.float32)
        output_decoder = state_decoder.h

        for i in range(self.decoder_max_sentence_length):
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
                e_list = tf.exp(e_list)
                e_list_mask = tf.multiply(tf.expand_dims(tf.transpose(video_mask), -1), e_list)
                weights = e_list_mask/tf.reduce_sum(e_list_mask, axis=0)
                generated_attention.append(weights)

                emb_weighted = [tf.multiply(emb, tf.tile(weight, [1, self.dim_embed])) \
                        for weight, emb in zip(tf.unstack(weights, axis=0), tf.unstack(obj_embs, axis=1))]
                emb_weighted_sum = tf.reduce_sum(emb_weighted, 0)

            with tf.variable_scope("Decoder"):
                # Word decoder LSTM.
                (output_decoder, state_decoder) = decoder(
                    tf.concat([emb_weighted_sum, output_att_lstm], 1), state_decoder)

                logit_words = tf.nn.xw_plus_b(output_decoder, self.embed_word_W, self.embed_word_b)

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
        return loss, obj_feats, video_mask, caption, caption_mask, generated_words, generated_attention
