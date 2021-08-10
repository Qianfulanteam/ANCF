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

    # Input
    user_attr_input = Input(shape=(21,), dtype='float32', name='user_attr_input')
    user_attr_embedding = Dense(15, activation="relu", name="user_att_embedding")(user_attr_input)  # 1st hidden layer

    attention_probs_u = Dense(15, activation='softmax', name='attention_probs_u')(user_attr_embedding)
    attention_u = multiply([user_attr_embedding, attention_probs_u], name='attention_u')
    z_u = Dense(8, activation="relu", name='z_u_embedding')(attention_u)  # z

    item_sub_class_input = Input(shape=(1,), dtype='float32')
    item_sub_class = Embedding(input_dim=2012, output_dim=8, name='item_sub_class',
                               embeddings_initializer=RandomNormal(
                                   mean=0.0, stddev=0.01, seed=None),
                               W_regularizer=l2(0), input_length=1)(item_sub_class_input)
    item_sub_class = Flatten()(item_sub_class)
    attention_probs_i_1 = Dense(8, activation='softmax', name='attention_probs_i_1')(item_sub_class)
    attention_i_1 = multiply([item_sub_class, attention_probs_i_1], name='attention_i_1')
    z_i_1 = Dense(4, activation="relu", name='z_i_1_embedding')(attention_i_1)  # z

    item_asset_price_input = Input(shape=(2,), dtype='float32')
    item_asset_price = Dense(8, activation="relu", name="item_asset_price_embedding")(item_asset_price_input)
    attention_probs_i_2 = Dense(8, activation='softmax', name='attention_probs_i_2')(item_asset_price)
    attention_i_2 = multiply([item_asset_price, attention_probs_i_2], name='attention_i_2')
    z_i_2 = Dense(4, activation="relu", name='z_i_2_embedding')(attention_i_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten()(user_id_Embedding(user_id_input))
    #user_id_Embedding = Dense(32, activation="relu", name="user_embedding")(user_id_Embedding)
    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)

    item_id_Embedding = Flatten()(item_id_Embedding(item_id_input))
    #item_id_Embedding = Dense(32, activation="relu", name="item_embedding")(item_id_Embedding)
    # id merge embedding
    user_id_attr = concatenate([user_id_Embedding, z_u], axis=1)
    item_id_attr = concatenate([item_id_Embedding, z_i_1, z_i_2],axis=1)

    # merge attr_id embedding
    merge_attr_id_embedding = concatenate([user_id_attr, item_id_attr], axis=1)
    merge_attr_id_embedding = Dense(40, activation="relu",name="layer1")(merge_attr_id_embedding)
    merge_attr_id_embedding = Dense(20, activation="relu",name="layer2")(merge_attr_id_embedding)
    merge_attr_id_embedding = Dense(10, activation="relu",name="layer3")(merge_attr_id_embedding)
    #merge_attr_id_embedding = Dense(5, W_regularizer=l2(0), activation="relu", name="layer4")(merge_attr_id_embedding)


    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(merge_attr_id_embedding)

    # Final prediction layer
    model = Model(input=[user_attr_input, item_sub_class_input, item_asset_price_input, user_id_input, item_id_input],
                  output=topLayer)

    return model





def main():
    learning_rate = 0.005
    num_epochs = 100
    verbose = 1
    topK = 10
    evaluation_threads = 1
    dataset = "tafeng"
    num_factor = 9
    out=1
    startTime = time()
    model_out_file = 'Pretrain/%s_tafengMLPmlp0_%d_%d.h5' % (dataset, num_factor, time())
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

    # to_file = 'model_' + "tangfengMLP" + '.png'
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
            testRatings = load_rating_file_as_list()
            testNegatives = load_negative_file()
            (hits, ndcgs, recalls) = evaluate_model(model, testRatings, testNegatives,
                                           users_attr_mat, items_genres_mat, topK, evaluation_threads)
            hr, ndcg, recall, loss = np.array(hits).mean(), np.array(ndcgs).mean(),  np.array(recalls).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, Recall = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, recall, loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
            if recall > best_recall:
                best_recall = recall
    endTime = time()
    print("End. best HR = %.4f, best NDCG = %.4f, best Recall = %.4f,time = %.1f s" %
          (best_hr, best_ndcg, best_recall, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f, Recall = %.4f' % (hr, ndcg, recall))
    if out > 0:
        print("The best tafengMLP model is saved to %s" %(model_out_file))

if __name__ == '__main__':
    main()
