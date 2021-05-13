##############################################################################
# This script is used to train recommendation engine based on CF using Keras #
##############################################################################

import argparse
import time
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



def RecommenderNet(n_users, n_articles, n_factors):
    user = keras.layers.Input(shape=(1,))
    u = EmbeddingLayer(n_users, n_factors)(user)
    
    article = keras.layers.Input(shape=(1,))
    a = EmbeddingLayer(n_articles, n_factors)(article)
    
    x = keras.layers.Concatenate()([u, a])
    x = keras.layers.Dropout(0.05)(x)

    x = keras.layers.Dense(512, kernel_initializer="he_normal")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)
        
    x = keras.layers.Dense(128, kernel_initializer="he_normal")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    
    output = keras.layers.Dense(1, kernel_initializer="he_normal")(x)
    
    model = keras.models.Model(inputs=[user, article], outputs=output)
    
    return model


class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = keras.layers.Embedding(self.n_items + 1, 
                                   self.n_factors, 
                                   embeddings_initializer="he_normal",
                                   embeddings_regularizer=keras.regularizers.l2(1e-6))(x)
        x = keras.layers.Reshape((self.n_factors,))(x)
        return x


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_df_train", type=str, required=True,
                        help="Path to the train set (df_train) csv file")
    parser.add_argument("--n_factors", type=int, default=100,
                        help="Size of embedding vectors")
    parser.add_argument("--loss", type=str, default="mean_squared_error",
                        help="Loss function used")                        
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Train/Validation batch size")                      
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Optimizer learning rate")
    parser.add_argument("--saved_name", type=str, default="best_model.h5", 
                        help="Name used to save the best fitted Keras model")                        
    parser.add_argument("--path_to_save_folder", type=str, required=True,
                        help="Folder to save output files")                        


    args = parser.parse_args()
    
    print("##########################################")
    print("# TRAINING STEP: COLLABORATIVE FILTERING #")
    print("##########################################\n")
                        
    print("----------------------------------")
    print("| Label encoding users and items |")
    print("----------------------------------\n")
    
    print("Path to df_train csv file:", args.path_to_df_train)
    df_train = pd.read_csv(args.path_to_df_train, sep=",")

    user_encoder = LabelEncoder()
    df_train["user_id_encoded"] = user_encoder.fit_transform(df_train["user_id"].values)
    n_users = df_train["user_id_encoded"].nunique()
    print("Number of unique users: {}".format(n_users))
    
    article_encoder = LabelEncoder()
    df_train["article_id_encoded"] = article_encoder.fit_transform(df_train["article_id"].values)
    n_articles = df_train["article_id_encoded"].nunique()    
    print("Number of unique articles: {}".format(n_articles))

    print("\n------------------------")
    print("| Training Keras model |")
    print("------------------------\n")  
    
    print("Learning rate: {}".format(args.learning_rate))
    print("Optimizer: Adam")
    print("Loss: {}".format(args.loss))
    print("batch size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.num_epochs))
    print("Best fitted model saved to: {}\n".format(os.path.join(args.path_to_save_folder, args.saved_name)))
    
    model = RecommenderNet(n_users=n_users, 
                           n_articles=n_articles, 
                           n_factors=args.n_factors)
                           
    optimizer = keras.optimizers.Adam(lr=args.learning_rate)

    try:
        os.makedirs(args.path_to_save_folder, exist_ok=True)
    except OSError as error:
        print("Save folder could not be created")
        
    df_train.to_csv(os.path.join(args.path_to_save_folder, "df_cf_nn.csv"))

    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(args.path_to_save_folder, args.saved_name), 
                                                       save_best_only=True, 
                                                       save_weights_only=False, 
                                                       monitor="val_loss",
                                                       verbose=1, 
                                                       mode="auto")
                                                       
    model.compile(optimizer=optimizer,
                  loss=args.loss,
                  metrics=None)
                  
    #X = df_train[["user_id_encoded", "article_id_encoded"]].values
    #y = df_train["session_size"].values  

    
                  
    history = model.fit(x=[df_train["user_id_encoded"], df_train["article_id_encoded"]], 
                        y=df_train["session_size"],
                        batch_size=args.batch_size, 
                        epochs=args.num_epochs,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=True,
                        callbacks=[model_checkpoint])

    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams.update({"font.size": 14})
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(["Train", "Validation"])                        

                        
if __name__ == "__main__":
    
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    
    print("\n------------")
    print("Total recommendation step time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)) 