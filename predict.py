####################################################
# This script is used to run recommendation engine #
####################################################

import argparse
import time
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import keras


def get_ratings_count_mean(df):

    ratings_count = {}
    ratings_mean = {}

    for article in tqdm(df["article_id"].unique()):
        count = 0
        total = 0
        for i in range(df.shape[0]):
            if df.iloc[i]["article_id"] == article:
                count += 1
                total += df.iloc[i]["session_size"]
        ratings_count[article] = count
        ratings_mean[article] = total / count    
    
    ratings_count_mean = pd.DataFrame.from_dict([ratings_mean, ratings_count]).T
    ratings_count_mean = ratings_count_mean.rename(columns={0: "session_size_mean", 1: "session_size_counts"})
    ratings_count_mean.index.name = "article_id"    
    
    return ratings_count_mean
    
def compute_wmr(m, ratings_count_mean):

    # Mean session size for each article
    R = ratings_count_mean["session_size_mean"].values
    
    # Mean session size overall 
    C = ratings_count_mean["session_size_mean"].mean()
        
    # Session size counts for each article 
    v = ratings_count_mean["session_size_counts"].values
    
    # Weighted formula to compute the weighted rating
    weighted_score = (v / (v+m) * R) + (m / (v+m) * C)
    
    # Sort ids to ranking
    weighted_ranking = np.argsort(weighted_score)[::-1]
    
    # Sort scores to ranking
    weighted_score = np.sort(weighted_score)[::-1]
    
    # Get article ids
    weighted_article_ids = ratings_count_mean.index[weighted_ranking]

    return weighted_score, weighted_article_ids


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["wmr", "iis", "cf"],
                        help="Recommendation model to used")
    parser.add_argument("--user_id", type=int, required=True,
                        help="id of the user")
    parser.add_argument("--num_reco", type=int, default=5,
                        help="Number of articles to recommend")
                        
    parser.add_argument("--path_to_df_wmr", type=str, required=True,
                        help="Path to the reference dataframe csv file for wmr model")
    parser.add_argument("--min_ratings", type=int, default=10,
                        help="Weighted mean m parameter: Minimum number of ratings to be considered")
                        
    parser.add_argument("--path_to_df_iis", type=str, required=True,
                        help="Path to the reference dataframe csv file for iis model")
    parser.add_argument("--path_to_embeddings", type=str, required=True,
                        help="Path to the embeddings pickle file")

    parser.add_argument("--path_to_df_cf", type=str, required=True,
                        help="Path to the reference dataframe csv file for cf model")
    parser.add_argument("--path_to_nn_model", type=str, required=True,
                        help="Path to the fitted Keras model file")

                        
    args = parser.parse_args()

    print("#########################")
    print("# RECOMMENDATION MODULE #")
    print("#########################\n")
    
    print("Recommendation for user ID:", args.user_id)
    if args.model == "wmr":
        reco_model = "Weighted mean rating"
    elif args.model == "iis":
        reco_model = "Item-Item similarity"
    elif args.model == "cf":
        reco_model = "Collaborative filtering with Keras"
        
    print("Recommendation model used:", reco_model)
    
    if args.model == "wmr":
        df = pd.read_csv(args.path_to_df_wmr, sep=",")
        ratings_count_mean = get_ratings_count_mean(df)
        weighted_score, weighted_article_ids = compute_wmr(args.min_ratings, ratings_count_mean)
        df_reco_wmr = pd.DataFrame(list(zip(weighted_article_ids.tolist(), weighted_score)), 
                                   columns=(["reco_article_id", "reco_rating_pred"]))
        recommendations = df_reco_wmr.sort_values(by=["reco_rating_pred"], 
                                                  ascending=False)["reco_article_id"].iloc[0:args.num_reco].values
    
    if args.model == "iis":
        embeddings = pickle.load(open(args.path_to_embeddings, "rb"))
        df_embeddings = pd.DataFrame(embeddings)
        
        print("\nComputing cosine similarity for user {} first opened article".format(args.user_id))
        
        df = pd.read_csv(args.path_to_df_iis, sep=",")
        article_id = df[df["user_id"] == args.user_id].sort_values(by=["click_timestamp"],
                                                                   ascending=False)["article_id"].iloc[0]
        
        article_embedding = df_embeddings.iloc[article_id].values.reshape(1, -1)

        similarities = []
        for i in tqdm(range(embeddings.shape[0])):
            cos_sim = cosine_similarity(article_embedding, df_embeddings.iloc[i].values.reshape(1, -1))
            similarities.append(cos_sim[0][0])
        
        similarities = np.array(similarities)
        
        ranked_similarities = np.sort(similarities)[::-1]
        ranked_ids = np.argsort(similarities)[::-1]   

        df_reco_sim = pd.DataFrame(list(zip(ranked_ids, ranked_similarities)), columns=(["reco_article_id", "similarity"]))
        
        recommendations = df_reco_sim["reco_article_id"].iloc[1:args.num_reco+1].values
        
    if args.model == "cf":
        model = keras.models.load_model(args.path_to_nn_model, compile=False)
        
        df = pd.read_csv(args.path_to_df_cf, sep=",")
        article_list = list(df["article_id_encoded"].unique())
        article_array = np.array(article_list)     
        user_id_encoded = df[df["user_id"] == args.user_id]["user_id_encoded"].values[0]
        print("User ID encoded", user_id_encoded)
        user_array = np.array([user_id_encoded for i in range(len(article_list))])
        
        predictions = model.predict([user_array, article_array])
        predictions = predictions.reshape(-1)
        articles = (-predictions).argsort()[0:args.num_reco]
        print(articles)
        recommendations = []
        for article in articles:
            recommendation = df[df["article_id_encoded"] == article]["article_id"].values[0]
            recommendations.append(recommendation)

    print("\nTop {} articles recommended: {}".format(args.num_reco, recommendations))
            
if __name__ == "__main__":
    
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    
    print("\n------------")
    print("Total time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)) 