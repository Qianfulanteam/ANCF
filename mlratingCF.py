# coding=UTF-8
"""
The version of package.
Python: 3.6.9
Keras: 2.0.8
Tensorflow-base:1.10.0
"""
import gc
import time
import keras
from time import time
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, multiply, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input, merge, Conv2D, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras.utils import plot_model
from Loadmovie100ldata import load_itemGenres_as_matrix
from Loadmovie100ldata import load_negative_file
from Loadmovie100ldata import load_rating_file_as_list
from Loadmovie100ldata import load_rating_train_as_matrix
from Loadmovie100ldata import load_user_attributes
from evaluate100k import evaluate_model
import mlratingMF as movieMF
import mlratingMLP as movieMLP
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

def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix

def get_lCoupledCF_model(num_users, num_items):

    num_users = num_users + 1
    num_items = num_items + 1
    ########################   attr side   ##################################

    # Input
    user_attr_input = Input(shape=(88,), dtype='float32', name='user_attr_input')  # 用户属性信息
    item_attr_input = Input(shape=(18,), dtype='float32', name='item_attr_input')  # 项目属性信息

    #mf attr part
    user_mf_attr_embedding = Dense(40, activation="relu",name="user_att_embedding_mf")(user_attr_input)  # 1st hidden layer
    user_mf_attr_embedding_1 = Dense(20, activation="relu", name="user_att_embedding_mf_1")(user_mf_attr_embedding)
    attention_probs_u_mf = Dense(20, activation='softmax', name='attention_probs_u_mf')(user_mf_attr_embedding_1)
    attention_u_mf = multiply([user_mf_attr_embedding_1, attention_probs_u_mf], name='attention_u_mf')
    z_u_mf = Dense(10, W_regularizer=l2(0), activation="relu", name='z_u_embedding_mf')(attention_u_mf)  # z
    item_mf_attr_embedding = Dense(10, activation="relu",name="item_att_embedding_mf")(item_attr_input)  # 1st hidden layer
    attention_probs_i_mf = Dense(10, activation='softmax', name='attention_probs_i_mf')(item_mf_attr_embedding)
    attention_i_mf = multiply([item_mf_attr_embedding, attention_probs_i_mf], name='attention_i_mf')
    z_i_mf = Dense(10, activation="relu", name='z_i_embedding_mf')(attention_i_mf)  # z

    # mlp attr part
    user_mlp_attr_embedding = Dense(40, activation="relu", name="user_att_embedding_mlp")(user_attr_input)  # 1st hidden layer
    user_mlp_attr_embedding_1 = Dense(20, activation="relu", name="user_att_embedding_mlp_1")(user_mlp_attr_embedding)
    attention_probs_u_mlp = Dense(20, activation='softmax', name='attention_probs_u_mlp')(user_mlp_attr_embedding_1)
    attention_u_mlp = multiply([user_mlp_attr_embedding_1, attention_probs_u_mlp], name='attention_u_mlp')
    z_u_mlp = Dense(10, activation="relu", name='z_u_embedding_mlp')(attention_u_mlp)  # z
    item_mlp_attr_embedding = Dense(10, activation="relu",name="item_att_embedding_mlp")(item_attr_input)  # 1st hidden layer
    attention_probs_i_mlp = Dense(10, activation='softmax', name='attention_probs_i_mlp')(item_mlp_attr_embedding)
    attention_i_mlp = multiply([item_mlp_attr_embedding, attention_probs_i_mlp], name='attention_i_mlp')
    z_i_mlp = Dense(10, activation="relu", name='z_i_embedding_mlp')(attention_i_mlp)  # z



    ########################   id side   ##################################
    #mf
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    MF_user_Embedding = Embedding(input_dim=num_users, output_dim=16, name='user_mf_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_latent_vector_mf = Flatten()(MF_user_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    MF_item_Embedding_mf = Embedding(input_dim=num_items, output_dim=16, name='item_mf_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_latent_vector_mf = Flatten()(MF_item_Embedding_mf(item_id_input))

    #mlp
    MLP_user_Embedding = Embedding(input_dim=num_users, output_dim=16, name='user_mlp_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_latent_vector_mlp = Flatten()(MLP_user_Embedding(user_id_input))

    MLP_item_Embedding = Embedding(input_dim=num_items, output_dim=16, name='item_mlp_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_latent_vector_mlp = Flatten()(MLP_item_Embedding(item_id_input))

    #mf part
    user_att_latent_mf=concatenate([z_u_mf, user_latent_vector_mf], axis=1)
    item_att_latent_mf = concatenate([z_i_mf, item_latent_vector_mf], axis=1)
    predict_vector_mf = multiply([user_att_latent_mf, item_att_latent_mf])

    #mlp part
    user_att_latent_mlp = concatenate([z_u_mlp, user_latent_vector_mlp], axis=1)
    item_att_latent_mlp = concatenate([z_i_mlp, item_latent_vector_mlp], axis=1)
    predict_vector_mlp = concatenate([user_att_latent_mlp, item_att_latent_mlp], axis=1)
    layer = Dense(26, activation='relu', name='layer2')
    vector = layer(predict_vector_mlp)
    layer = Dense(13, activation='relu', name='layer3')
    vector_mlp = layer(vector)


    predict_vector = concatenate([predict_vector_mf, vector_mlp], axis=1)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(predict_vector)
    # Final prediction layer
    model = Model(input=[user_attr_input, item_attr_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def load_pretrain_model(model, gmf_model, mlp_model):
    # MF embeddings
    gmf_user_att_embeddings = gmf_model.get_layer('user_att_embedding').get_weights()
    gmf_user_att_embeddings_1 = gmf_model.get_layer('user_att_embedding_1').get_weights()
    gmf_user_attention_probs_u = gmf_model.get_layer('attention_probs_u').get_weights()
    gmf_attention_u = gmf_model.get_layer('attention_u').get_weights()
    gmf_z_u_embedding = gmf_model.get_layer('z_u_embedding').get_weights()

    gmf_item_att_embeddings = gmf_model.get_layer('item_att_embedding').get_weights()
    gmf_item_attention_probs_i = gmf_model.get_layer('attention_probs_i').get_weights()
    gmf_attention_i = gmf_model.get_layer('attention_i').get_weights()
    gmf_z_i_embedding = gmf_model.get_layer('z_i_embedding').get_weights()

    model.get_layer('user_att_embedding_mf').set_weights(gmf_user_att_embeddings)
    model.get_layer('user_att_embedding_mf_1').set_weights(gmf_user_att_embeddings_1)
    model.get_layer('attention_probs_u_mf').set_weights(gmf_user_attention_probs_u)
    model.get_layer('attention_u_mf').set_weights(gmf_attention_u)
    model.get_layer('z_u_embedding_mf').set_weights(gmf_z_u_embedding)

    model.get_layer('item_att_embedding_mf').set_weights(gmf_item_att_embeddings)
    model.get_layer('attention_probs_i_mf').set_weights(gmf_item_attention_probs_i)
    model.get_layer('attention_i_mf').set_weights(gmf_attention_i)
    model.get_layer('z_i_embedding_mf').set_weights(gmf_z_i_embedding)

    gmf_user_embeddings = gmf_model.get_layer('user_id_Embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_id_Embedding').get_weights()
    model.get_layer('user_mf_Embedding').set_weights(gmf_user_embeddings)
    model.get_layer('item_mf_Embedding').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_id_Embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_id_Embedding').get_weights()
    model.get_layer('user_mlp_Embedding').set_weights(mlp_user_embeddings)
    model.get_layer('item_mlp_Embedding').set_weights(mlp_item_embeddings)

    mlp_user_att_embeddings = mlp_model.get_layer('user_att_embedding').get_weights()
    mlp_user_att_embeddings_1 = mlp_model.get_layer('user_att_embedding_1').get_weights()
    mlp_user_attention_probs_u = mlp_model.get_layer('attention_probs_u').get_weights()
    mlp_attention_u = mlp_model.get_layer('attention_u').get_weights()
    mlp_z_u_embedding = mlp_model.get_layer('z_u_embedding').get_weights()

    mlp_item_att_embeddings = mlp_model.get_layer('item_att_embedding').get_weights()
    mlp_item_attention_probs_i = mlp_model.get_layer('attention_probs_i').get_weights()
    mlp_attention_i = mlp_model.get_layer('attention_i').get_weights()
    mlp_z_i_embedding = mlp_model.get_layer('z_i_embedding').get_weights()

    model.get_layer('user_att_embedding_mlp').set_weights(mlp_user_att_embeddings)
    model.get_layer('user_att_embedding_mlp_1').set_weights(mlp_user_att_embeddings_1)
    model.get_layer('attention_probs_u_mlp').set_weights(mlp_user_attention_probs_u)
    model.get_layer('attention_u_mlp').set_weights(mlp_attention_u)
    model.get_layer('z_u_embedding_mlp').set_weights(mlp_z_u_embedding)

    model.get_layer('item_att_embedding_mlp').set_weights(mlp_item_att_embeddings)
    model.get_layer('attention_probs_i_mlp').set_weights(mlp_item_attention_probs_i)
    model.get_layer('attention_i_mlp').set_weights(mlp_attention_i)
    model.get_layer('z_i_embedding_mlp').set_weights(mlp_z_i_embedding)

    # MLP layers
    mlp_layer2_weights = mlp_model.get_layer('layer2').get_weights()
    mlp_layer3_weights = mlp_model.get_layer('layer3').get_weights()
    #mlp_layer4_weights = mlp_model.get_layer('layer4').get_weights()
    #mlp_layer5_weights = mlp_model.get_layer('layer5').get_weights()

    model.get_layer('layer2').set_weights(mlp_layer2_weights)
    model.get_layer('layer3').set_weights(mlp_layer3_weights)
    #model.get_layer('layer4').set_weights(mlp_layer4_weights)
    #model.get_layer('layer5').set_weights(mlp_layer5_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('topLayer').get_weights()
    mlp_prediction = mlp_model.get_layer('topLayer').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('topLayer').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model
def main():

    learning_rate = 0.001
    num_epochs = 50
    verbose = 1
    topK = 10
    out=1
    dataset="ml_100k"
    num_factor=16
    mf_pretrain = ""
    mlp_pretrain = ""
    # mf_pretrain = "Pretrain/ml_100k_movieMF100k_64_1623310379.h5"
    # mlp_pretrain = "Pretrain/ml_100k_movieMLP100k_8_1623303783.h5"
    evaluation_threads = 1
    startTime = time()
    model_out_file = '/data/chenhai-fwxz/code/ANCF/Pretrain/%s_movieCF100k_%d_%s_%d.h5' %(dataset, num_factor, 5, time())
    # load data
    num_users, users_attr_mat = load_user_attributes()  # 用户，用户属性
    num_items, items_genres_mat = load_itemGenres_as_matrix()  # 项目，项目属性
    ratings = load_rating_train_as_matrix()  # 评分矩阵

    # load model
    # change the value of 'theModel' with the key in 'model_dict'
    # to load different models

    theModel = "movieCF100k"
    model = get_lCoupledCF_model(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    # to_file = 'Model_' + theModel + '.png'
    # plot_model(model, show_shapes=True, to_file=to_file)
    # model.summary()

    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = movieMF.get_lCoupledCF_model(num_users, num_items)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = movieMLP.get_lCoupledCF_model(num_users, num_items)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model)
        print("Load pretrained movieMF (%s) and movieMLP (%s) models done. " % (mf_pretrain, mlp_pretrain))

    # Init performance
    testRatings = load_rating_file_as_list()
    testNegatives = load_negative_file()
    (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
    hr, ndcg, recall = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(recalls).mean()
    print('Init: HR = %.4f, NDCG = %.4f, Recall = %.4f' % (hr, ndcg, recall))
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
        hist5 = model.fit([user_attr_input, item_attr_input, user_id_input, item_id_input], labels, epochs=1,
                          batch_size=256, verbose=2, shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
            hr, ndcg,recall, loss5 = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(recalls).mean(), hist5.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, Recall = %.4f, loss5 = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, recall, loss5, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            if recall > best_recall:
                    best_recall = recall
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f,  best NDCG = %.4f, time = %.1f s" %
          (best_hr, best_ndcg,best_recall, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f, Recall = %.4f' % (hr, ndcg, recall))
    if out > 0:
        print("The best movieCF100k model is saved to %s" %(model_out_file))

if __name__ == '__main__':
    main()
