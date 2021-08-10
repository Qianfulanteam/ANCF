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


def get_train_instances(users_attr_mat, ratings, items_genres_mat):
    user_attr_input, item_attr_input, user_id_input, item_id_input, labels = [], [], [], [], []
    num_users, num_items = ratings.shape
    num_negatives=4

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

    user_attr_embedding = Dense(40, activation="relu", name="user_att_embedding")(user_attr_input)  # 1st hidden layer
    user_attr_embedding_1 = Dense(20, activation="relu", name="user_att_embedding_1")(user_attr_embedding)

    attention_probs_u = Dense(20, activation='softmax', name='attention_probs_u')(user_attr_embedding_1)
    attention_u = multiply([user_attr_embedding_1, attention_probs_u], name='attention_u')
    z_u = Dense(10, activation="relu", name='z_u_embedding')(attention_u)  # z


    item_attr_embedding = Dense(10, activation="relu", name="item_att_embedding")(item_attr_input)  # 1st hidden layer


    attention_probs_i = Dense(10, activation='softmax', name='attention_probs_i')(item_attr_embedding)
    attention_i = multiply([item_attr_embedding, attention_probs_i], name='attention_i')
    z_i = Dense(10, activation="relu", name='z_i_embedding')(attention_i)  # z



    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=16, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_latent_vector = Flatten()(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=16, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_latent_vector = Flatten()(item_id_Embedding(item_id_input))

    user_att_latent=concatenate([z_u, user_latent_vector], axis=1)
    item_att_latent = concatenate([z_i, item_latent_vector], axis=1)
    predict_vector = multiply([user_att_latent, item_att_latent])

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(predict_vector)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_attr_input, user_id_input, item_id_input],
                  output=topLayer)

    return model


def main():
    learning_rate = 0.001
    num_epochs = 100
    verbose = 1
    topK = 10
    out=1
    dataset="ml_100k"
    num_factor=16
    evaluation_threads = 1
    #num_negatives = 4
    startTime = time()
    model_out_file = 'Pretrain/%s_movieMF100k_%d_%d.h5' % (dataset,num_factor , time())
    # load data
    num_users, users_attr_mat = load_user_attributes()  # 用户，用户属性
    num_items, items_genres_mat = load_itemGenres_as_matrix()  # 项目，项目属性
    ratings = load_rating_train_as_matrix()  # 评分矩阵

    # load model
    # change the value of 'theModel' with the key in 'model_dict'
    # to load different models

    theModel = "movieMF100k"
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

    # Training model
    best_hr, best_ndcg, best_recall = 0, 0, 0
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
            testRatings = load_rating_file_as_list()
            testNegatives = load_negative_file()
            (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
            hr, ndcg, recall, loss5 = np.array(hits).mean(), np.array(ndcgs).mean(),np.array(recalls).mean(), hist5.history['loss'][0]
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
          (best_hr, best_ndcg, best_recall, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f, recall = %.4f' % (hr, ndcg, recall))
    if out > 0:
        print("The best movie100kMF model is saved to %s" %(model_out_file))

if __name__ == '__main__':
    main()
