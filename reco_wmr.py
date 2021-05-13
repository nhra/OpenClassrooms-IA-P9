####################################################################################
# This script is used to train recommendation engine based on Weighted Mean Rating #
####################################################################################

import argparse
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def get_ratings_count_mean(df_train):

    ratings_count = {}
    ratings_mean = {}

    for article in tqdm(df_train["article_id"].unique()):
        count = 0
        total = 0
        for i in range(df_train.shape[0]):
            if df_train.iloc[i]["article_id"] == article:
                count += 1
                total += df_train.iloc[i]["session_size"]
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

    
def compute_rmse(df_test, weighted_score, weighted_article_ids, how="left"):
    # Join labels and predictions
    if how == "left":
        df_prediction = df_test.set_index("article_id").join(pd.DataFrame(weighted_score, 
                                                                          index=weighted_article_ids, 
                                                                          columns=["session_size_pred"]))
        df_prediction = df_prediction[["session_size", "session_size_pred"]].fillna(value=0)
    elif how == "inner":
        df_prediction = df_test.set_index("article_id").join(pd.DataFrame(weighted_score, 
                                                                          index=weighted_article_ids, 
                                                                          columns=["session_size_pred"]), 
                                                             how="inner")    
        df_prediction = df_prediction[["session_size", "session_size_pred"]]
    y_true = df_prediction["session_size"]
    y_pred = df_prediction["session_size_pred"]

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    
    return rmse, df_prediction


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_df_train", type=str, required=True,
                        help="Path to the train set (df_train) csv file")
    parser.add_argument("--path_to_df_test", type=str, required=True,
                        help="Path to the test set (df_test) csv file")
    parser.add_argument("--min_ratings", type=int, default=1000,
                        help="Weighted mean m parameter: Minimum number of ratings to be considered")
    parser.add_argument("--path_to_save_folder", type=str, required=True,
                        help="Folder to save files")                        


    args = parser.parse_args()
    
    print("#######################################")
    print("# TRAINING STEP: WEIGHTED MEAN RATING #")
    print("#######################################\n")
                        
    print("----------------------------------")
    print("| Getting rating counts and mean |")
    print("----------------------------------\n")

    print("Path to df_train csv file:", args.path_to_df_train)    
    
    df_train = pd.read_csv(args.path_to_df_train, sep=",")
    
    ratings_count_mean = get_ratings_count_mean(df_train)
    
    try:
        os.makedirs(args.path_to_save_folder, exist_ok=True)
    except OSError as error:
        print("Save folder could not be created")
        
    ratings_count_mean.to_csv(os.path.join(args.path_to_save_folder, "df_counts_mean.csv"))
    
    print("\n-----------------------------------")
    print("| Computing weighted mean ratings |")
    print("-----------------------------------\n")    
    
    print("Weighted mean m parameter:", args.min_ratings)
    weighted_score, weighted_article_ids = compute_wmr(args.min_ratings, ratings_count_mean)
    
    print("Path to the saved df_reco_wmr csv file:", os.path.join(args.path_to_save_folder, "df_reco_wmr.csv"))
    df_reco_wmr = pd.DataFrame(list(zip(weighted_article_ids.tolist(), weighted_score)), columns=(["reco_article_id", "reco_rating_pred"]))
    df_reco_wmr.to_csv(os.path.join(args.path_to_save_folder, "df_reco_wmr.csv"))

    print("\n------------------")
    print("| Computing RMSE |")
    print("------------------\n")      

    print("Path to df_test csv file:", args.path_to_df_test)    
    
    df_test = pd.read_csv(args.path_to_df_test, sep=",")    
    rmse, df_prediction = compute_rmse(df_test, weighted_score, weighted_article_ids, how="inner")
    df_prediction.to_csv(os.path.join(args.path_to_save_folder, "df_reco_wmr_pred.csv"))
    
    print("RMSE:", round(rmse, 2))
    
if __name__ == "__main__":
    
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    
    print("\n------------")
    print("Total recommendation step time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))