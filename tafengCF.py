# coding=UTF-8

"""
The version of package.
Python: 3.6.9
Keras: 2.0.8
Tensorflow-base:1.10.0
"""
import gc
import time
from time import time

import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten,multiply, Lambda, Reshape, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input, merge, Conv2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras.utils import plot_model

from LoadTafengData import load_itemGenres_as_matrix
from LoadTafengData import load_negative_file
from LoadTafengData import load_rating_file_as_list
from LoadTafengData import load_rating_train_as_matrix
from LoadTafengData import load_user_attributes
from evaluatetafeng import evaluate_model
import tangfengMF as tafengMF
import tafengMLP as tafengMLP
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_train_instances(users_attr_mat, ratings, items_genres_mat):
    user_attr_input, item_attr_input, user_id_input, item_id_input, labels = [], [], [], [], []
    num_users, num_items = ratings.shape
    num_negatives = 4

    for (u, i) in ratings.keys():
        # positive instance
        user_attr_input.append(users_attr_mat[u])
        user_id_input.append([u])
        item_id_input.append([i])
        item_attr_input.append(items_genres_mat[i])
        labels.append([1])

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)

            user_attr_input.append(users_attr_mat[u])
            user_id_input.append([u])
            item_id_input.append([j])
            item_attr_input.append(items_genres_mat[j])
            labels.append([0])

    array_user_attr_input = np.array(user_attr_input)
    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_item_attr_input = np.array(item_attr_input)
    array_labels = np.array(labels)

    del user_attr_input, user_id_input, item_id_input, item_attr_input, labels
    gc.collect()

    return array_user_attr_input, array_user_id_input, array_item_attr_input, array_item_id_input, array_labels


def get_lCoupledCF_model(num_users, num_items):
    """
    lCoupledCF

    """
    num_users = num_users + 1
    num_items = num_items + 1

    ########################   attr side   ##################################

    # MF Input
    user_attr_input = Input(shape=(21,), dtype='float32', name='user_attr_input')
    user_attr_embedding_mf = Dense(15, activation="relu", name="user_att_embedding_mf")(user_attr_input)  # 1st hidden layer

    attention_probs_u_mf = Dense(15, activation='softmax', name='attention_probs_u_mf')(user_attr_embedding_mf)
    attention_u_mf = multiply([user_attr_embedding_mf, attention_probs_u_mf], name='attention_u_mf')
    z_u_mf = Dense(8, activation="relu", name='z_u_embedding_mf')(attention_u_mf)  # z

    item_sub_class_input = Input(shape=(1,), dtype='float32')
    item_sub_class_mf = Embedding(input_dim=2012, output_dim=8, name='item_sub_class_mf',
                               embeddings_initializer=RandomNormal(
                                   mean=0.0, stddev=0.01, seed=None),
                               W_regularizer=l2(0), input_length=1)(item_sub_class_input)
    item_sub_class_mf = Flatten()(item_sub_class_mf)
    attention_probs_i_1_mf = Dense(8, activation='softmax', name='attention_probs_i_1_mf')(item_sub_class_mf)
    attention_i_1_mf = multiply([item_sub_class_mf, attention_probs_i_1_mf], name='attention_i_1_mf')
    z_i_1_mf = Dense(4, activation="relu", name='z_i_1_embedding_mf')(attention_i_1_mf)  # z

    item_asset_price_input = Input(shape=(2,), dtype='float32')
    item_asset_price_mf = Dense(8, activation="relu", name="item_asset_price_embedding_mf")(item_asset_price_input)
    attention_probs_i_2_mf = Dense(8, activation='softmax', name='attention_probs_i_2_mf')(item_asset_price_mf)
    attention_i_2_mf = multiply([item_asset_price_mf, attention_probs_i_2_mf], name='attention_i_2_mf')
    z_i_2_mf = Dense(4, activation="relu", name='z_i_2_embedding_mf')(attention_i_2_mf)

    # MLP Input
    user_attr_embedding_mlp = Dense(15, activation="relu", name="user_att_embedding_mlp")(
        user_attr_input)  # 1st hidden layer
    attention_probs_u_mlp = Dense(15, activation='softmax', name='attention_probs_u_mlp')(user_attr_embedding_mlp)
    attention_u_mlp = multiply([user_attr_embedding_mlp, attention_probs_u_mlp], name='attention_u_mlp')
    z_u_mlp = Dense(8, activation="relu", name='z_u_embedding_mlp')(attention_u_mlp)  # z

    item_sub_class_mlp = Embedding(input_dim=2012, output_dim=8, name='item_sub_class_mlp',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)(item_sub_class_input)
    item_sub_class_mlp = Flatten()(item_sub_class_mlp)
    attention_probs_i_1_mlp = Dense(8, activation='softmax', name='attention_probs_i_1_mlp')(item_sub_class_mlp)
    attention_i_1_mlp = multiply([item_sub_class_mlp, attention_probs_i_1_mlp], name='attention_i_1_mlp')
    z_i_1_mlp = Dense(4, activation="relu", name='z_i_1_embedding_mlp')(attention_i_1_mlp)  # z


    item_asset_price_mlp = Dense(8, activation="relu", name="item_asset_price_embedding_mlp")(item_asset_price_input)
    attention_probs_i_2_mlp = Dense(8, activation='softmax', name='attention_probs_i_2_mlp')(item_asset_price_mlp)
    attention_i_2_mlp = multiply([item_asset_price_mlp, attention_probs_i_2_mlp], name='attention_i_2_mlp')
    z_i_2_mlp = Dense(4, activation="relu", name='z_i_2_embedding_mlp')(attention_i_2_mlp)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding_mf = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding_mf',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding_mf = Flatten()(user_id_Embedding_mf(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding_mf = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding_mf',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)

    item_id_Embedding_mf = Flatten()(item_id_Embedding_mf(item_id_input))
    #mlp

    user_id_Embedding_mlp = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding_mlp',
                                     embeddings_initializer=RandomNormal(
                                         mean=0.0, stddev=0.01, seed=None),
                                     W_regularizer=l2(0), input_length=1)
    user_id_Embedding_mlp = Flatten()(user_id_Embedding_mlp(user_id_input))


    item_id_Embedding_mlp = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding_mlp',
                                     embeddings_initializer=RandomNormal(
                                         mean=0.0, stddev=0.01, seed=None),
                                     W_regularizer=l2(0), input_length=1)

    item_id_Embedding_mlp = Flatten()(item_id_Embedding_mlp(item_id_input))
    # mf merge embedding
    user_id_attr_mf = concatenate([user_id_Embedding_mf, z_u_mf], axis=1)
    item_id_attr_mf = concatenate([item_id_Embedding_mf, z_i_1_mf, z_i_2_mf], axis=1)
    predict_vector_mf = multiply([user_id_attr_mf, item_id_attr_mf])
    # mlp merge embedding
    user_id_attr_mlp = concatenate([user_id_Embedding_mlp, z_u_mlp],axis=1)
    item_id_attr_mlp = concatenate([item_id_Embedding_mlp, z_i_1_mlp, z_i_2_mlp], axis=1)
    predict_vector_mlp = concatenate([user_id_attr_mlp, item_id_attr_mlp], axis=1)
    #merge_attr_id_embedding = Dense(64, activation="relu")(merge_attr_id_embedding)
    predict_vector_mlp = Dense(40, activation="relu",name="layer1")(predict_vector_mlp)
    predict_vector_mlp = Dense(20, activation="relu",name="layer2")(predict_vector_mlp)
    predict_vector_mlp = Dense(10, activation="relu",name="layer3")(predict_vector_mlp)

    vector=concatenate([predict_vector_mf,predict_vector_mlp],axis=1)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(vector)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def load_pretrain_model(model, gmf_model, mlp_model):
    # MF embeddings
    gmf_user_att_embeddings = gmf_model.get_layer('user_att_embedding').get_weights()
    gmf_user_attention_probs_u = gmf_model.get_layer('attention_probs_u').get_weights()
    gmf_attention_u = gmf_model.get_layer('attention_u').get_weights()
    gmf_z_u_embedding = gmf_model.get_layer('z_u_embedding').get_weights()

    gmf_item_att_embeddings = gmf_model.get_layer('item_sub_class').get_weights()
    gmf_item_attention_probs_i = gmf_model.get_layer('attention_probs_i_1').get_weights()
    gmf_attention_i = gmf_model.get_layer('attention_i_1').get_weights()
    gmf_z_i_embedding = gmf_model.get_layer('z_i_1_embedding').get_weights()

    gmf_item_att_embeddings_2 = gmf_model.get_layer('item_asset_price_embedding').get_weights()
    gmf_item_attention_probs_i_2 = gmf_model.get_layer('attention_probs_i_2').get_weights()
    gmf_attention_i_2 = gmf_model.get_layer('attention_i_2').get_weights()
    gmf_z_i_2_embedding = gmf_model.get_layer('z_i_2_embedding').get_weights()

    model.get_layer('user_att_embedding_mf').set_weights(gmf_user_att_embeddings)
    model.get_layer('attention_probs_u_mf').set_weights(gmf_user_attention_probs_u)
    model.get_layer('attention_u_mf').set_weights(gmf_attention_u)
    model.get_layer('z_u_embedding_mf').set_weights(gmf_z_u_embedding)

    model.get_layer('item_sub_class_mf').set_weights(gmf_item_att_embeddings)
    model.get_layer('attention_probs_i_1_mf').set_weights(gmf_item_attention_probs_i)
    model.get_layer('attention_i_1_mf').set_weights(gmf_attention_i)
    model.get_layer('z_i_1_embedding_mf').set_weights(gmf_z_i_embedding)

    model.get_layer('item_asset_price_embedding_mf').set_weights(gmf_item_att_embeddings_2)
    model.get_layer('attention_probs_i_2_mf').set_weights(gmf_item_attention_probs_i_2)
    model.get_layer('attention_i_2_mf').set_weights(gmf_attention_i_2)
    model.get_layer('z_i_2_embedding_mf').set_weights(gmf_z_i_2_embedding)

    gmf_user_embeddings = gmf_model.get_layer('user_id_Embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_id_Embedding').get_weights()
    model.get_layer('user_id_Embedding_mf').set_weights(gmf_user_embeddings)
    model.get_layer('item_id_Embedding_mf').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_id_Embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_id_Embedding').get_weights()
    model.get_layer('user_id_Embedding_mlp').set_weights(mlp_user_embeddings)
    model.get_layer('item_id_Embedding_mlp').set_weights(mlp_item_embeddings)

    # MLP embeddings
    mlp_user_att_embeddings = mlp_model.get_layer('user_att_embedding').get_weights()
    mlp_user_attention_probs_u = mlp_model.get_layer('attention_probs_u').get_weights()
    mlp_attention_u = mlp_model.get_layer('attention_u').get_weights()
    mlp_z_u_embedding = mlp_model.get_layer('z_u_embedding').get_weights()

    mlp_item_att_embeddings = mlp_model.get_layer('item_sub_class').get_weights()
    mlp_item_attention_probs_i = mlp_model.get_layer('attention_probs_i_1').get_weights()
    mlp_attention_i = mlp_model.get_layer('attention_i_1').get_weights()
    mlp_z_i_embedding = mlp_model.get_layer('z_i_1_embedding').get_weights()

    mlp_item_att_embeddings_2 = mlp_model.get_layer('item_asset_price_embedding').get_weights()
    mlp_item_attention_probs_i_2 = mlp_model.get_layer('attention_probs_i_2').get_weights()
    mlp_attention_i_2 = mlp_model.get_layer('attention_i_2').get_weights()
    mlp_z_i_2_embedding = mlp_model.get_layer('z_i_2_embedding').get_weights()

    model.get_layer('user_att_embedding_mlp').set_weights(mlp_user_att_embeddings)
    model.get_layer('attention_probs_u_mlp').set_weights(mlp_user_attention_probs_u)
    model.get_layer('attention_u_mlp').set_weights(mlp_attention_u)
    model.get_layer('z_u_embedding_mlp').set_weights(mlp_z_u_embedding)

    model.get_layer('item_sub_class_mlp').set_weights(mlp_item_att_embeddings)
    model.get_layer('attention_probs_i_1_mlp').set_weights(mlp_item_attention_probs_i)
    model.get_layer('attention_i_1_mlp').set_weights(mlp_attention_i)
    model.get_layer('z_i_1_embedding_mlp').set_weights(mlp_z_i_embedding)

    model.get_layer('item_asset_price_embedding_mlp').set_weights(mlp_item_att_embeddings_2)
    model.get_layer('attention_probs_i_2_mlp').set_weights(mlp_item_attention_probs_i_2)
    model.get_layer('attention_i_2_mlp').set_weights(mlp_attention_i_2)
    model.get_layer('z_i_2_embedding_mlp').set_weights(mlp_z_i_2_embedding)

    # MLP layers
    mlp_layer2_weights = mlp_model.get_layer('layer1').get_weights()
    mlp_layer3_weights = mlp_model.get_layer('layer2').get_weights()
    mlp_layer4_weights = mlp_model.get_layer('layer3').get_weights()
    #mlp_layer5_weights = mlp_model.get_layer('layer5').get_weights()

    model.get_layer('layer1').set_weights(mlp_layer2_weights)
    model.get_layer('layer2').set_weights(mlp_layer3_weights)
    model.get_layer('layer3').set_weights(mlp_layer4_weights)
    #model.get_layer('layer5').set_weights(mlp_layer5_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('topLayer').get_weights()
    mlp_prediction = mlp_model.get_layer('topLayer').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('topLayer').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model



def main():
    learning_rate = 0.005
    num_epochs = 100
    verbose = 1
    topK = 10
    evaluation_threads = 1
    dataset = "tafeng"
    num_factor = 10
    out=1
    # mf_pretrain = "/data/chenhai-fwxz/code/ANCF/Pretrain/tafeng_tafengMF_32_1624590083.h5"
    # mlp_pretrain = "/data/chenhai-fwxz/code/ANCF/Pretrain/tafeng_tafengMLPmlp0_9_1624590829.h5"
    mf_pretrain = ""
    mlp_pretrain = ""
    startTime = time()
    model_out_file = 'Pretrain/%s_tafengCF_%d_%d.h5' % (dataset, num_factor, time())
    # load data
    num_users, users_attr_mat = load_user_attributes()
    num_items, items_genres_mat = load_itemGenres_as_matrix()
    # users_vec_mat = load_user_vectors()
    ratings = load_rating_train_as_matrix()

    # load model
    # change the value of 'theModel' with the key in 'model_dict'
    # to load different models
    model = get_lCoupledCF_model(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )

    # to_file = 'model_' + "tangfengCF" + '.png'
    # plot_model(model, show_shapes=True, to_file=to_file)
    # model.summary()
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = tafengMF.get_lCoupledCF_model(num_users, num_items)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = tafengMLP.get_lCoupledCF_model(num_users, num_items)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model)
        print("Load pretrained movieMF (%s) and movieMLP (%s) models done. " % (mf_pretrain, mlp_pretrain))
    # Init performance
    testRatings = load_rating_file_as_list()
    testNegatives = load_negative_file()
    (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                       users_attr_mat, items_genres_mat, topK, evaluation_threads)
    hr, ndcg, recall = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(recalls).mean()
    print('Init: HR = %.4f, NDCG = %.4f,Recall = %.4f' % (hr, ndcg, recall))
    best_hr, best_ndcg, best_recall, best_iter = hr, ndcg,recall, -1
    if out > 0:
            model.save_weights(model_out_file, overwrite=True)
    # Training model
    for epoch in range(num_epochs):
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_attr_input, user_id_input, item_attr_input, item_id_input, labels = get_train_instances(users_attr_mat,
                                                                                                     ratings,
                                                                                                     items_genres_mat)
        item_sub_class = item_attr_input[:, 0]
        item_asset_price = item_attr_input[:, 1:]

        hist = model.fit([user_attr_input, item_sub_class, item_asset_price, user_id_input, item_id_input],
                         labels, epochs=1,
                         batch_size=256,
                         verbose=1,
                         shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
            hr, ndcg, recall, loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(recalls).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f,Recall = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg,recall, loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            if recall > best_recall:
                best_recall = recall
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f,  best Recall = %.4f,time = %.1f s" %
          (best_hr, best_ndcg, best_recall, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f, Recall = %.4f' % (hr, ndcg, recall))
    if out > 0:
        print("The best tafengCF model is saved to %s" %(model_out_file))

if __name__ == '__main__':
    main()
