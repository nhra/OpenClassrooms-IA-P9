######################################################################
# This script is used to prepare data for the recommendation engines #
######################################################################

import argparse
import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def collect_data(path_to_click_folder, path_to_articles):
    
    # Load click files into a dataframe
    df_clicks = pd.DataFrame().append([pd.read_csv(path_to_click_folder + "\\" + file, sep=",") for file in os.listdir(path_to_click_folder)], 
                                      ignore_index=True)
    
    # Rename article column
    df_clicks.rename(columns = {"click_article_id": "article_id"}, inplace = True)
    
    # Load article metadata into a dataframe
    df_articles = pd.read_csv("./articles_metadata.csv", sep=",")

    print("Number of unique users:", df_clicks["user_id"].nunique())
    print("Number of unique articles:", df_clicks["article_id"].nunique())
    
    # Merge article info to click dataframe
    df_merge = pd.merge(df_clicks, df_articles, 
                        how="inner", on="article_id")
    
    # Select useful columns
    df_ratings = df_merge[["user_id", "article_id", "session_size", "click_timestamp"]]
    
    # Convert data to int64
    df_ratings = df_ratings.astype(np.int64)
    
    return df_ratings


def apply_filtering(df, min_user_ratings=0, min_article_ratings=0):

    # Filter sparse users
    filter_users = (df["user_id"].value_counts() >= min_user_ratings)
    filter_users = filter_users[filter_users].index.tolist()
        
    # Filter sparse movies
    filter_articles = (df["article_id"].value_counts() >= min_article_ratings)
    filter_articles = filter_articles[filter_articles].index.tolist()
    
    # Actual filtering
    df_filtered = df[(df["article_id"].isin(filter_articles)) & (df["user_id"].isin(filter_users))]
    
    if ((min_user_ratings == 0) and (min_article_ratings == 0)):
        print("Shape of the unfiltered dataframe: {}".format(df.shape))
        print("Shape of the filtered dataframe: {}".format(df_filtered.shape))
    
    return df_filtered
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_click_folder", type=str, required=True,
                        help="Path to the folder containing click files")
    parser.add_argument("--path_to_articles", type=str, required=True,
                        help="Path to the file with article metadata") 
    parser.add_argument("--min_user_ratings", type=int, default=0,
                        help="Minimum number of ratings a user should have to be considered")
    parser.add_argument("--min_article_ratings", type=int, default=0,
                        help="Minimum number of ratings an article should have to be considered")
    parser.add_argument("--test_size_ratio", type=float, default=0.2,
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument("--random_state", type=int, default=140583,
                        help="Shuffling applied to the data before applying the split")                        
    parser.add_argument("--path_to_save_folder", type=str, required=True,
                        help="Folder to save the preprocessed train/test files")                        


    args = parser.parse_args()

    print("####################")
    print("# DATA PREPARATION #")
    print("####################\n")
                        
    print("-------------------")
    print("| Collecting data |")
    print("-------------------\n")     

    print("Path to click folder:", args.path_to_click_folder)
    print("Path to article file:", args.path_to_articles)

    df = collect_data(args.path_to_click_folder, args.path_to_articles)
    
    print("\n------------------")
    print("| Filtering data |")
    print("------------------\n")    

    print("Minimum number of user ratings to consider:", args.min_user_ratings)
    print("Minimum number of article ratings to consider:", args.min_article_ratings)       
    
    df = apply_filtering(df, args.min_user_ratings, args.min_article_ratings)

    print("Saving the data csv file to:", args.path_to_save_folder)

    try:
        os.makedirs(args.path_to_save_folder, exist_ok = True)
    except OSError as error:
        print("Save folder could not be created")
        
    df.to_csv(os.path.join(args.path_to_save_folder, "df_all.csv"), index=False)
        
    print("\n--------------------------------")
    print("| Splitting data in train/Test |")
    print("--------------------------------\n")     
    
    print("Random state used to shuffle the data before applying the split:", args.random_state)
    print("Proportion of the dataset in the test split:", args.test_size_ratio)
    
    df_train, df_test = train_test_split(df, test_size=args.test_size_ratio, random_state=args.random_state)
    
    print("Shape of the train set: {}".format(df_train.shape))
    print("Shape of the test set: {}".format(df_test.shape))
    print("Saving the train/test csv files to:", args.path_to_save_folder)

    try:
        os.makedirs(args.path_to_save_folder, exist_ok=True)
    except OSError as error:
        print("Save folder could not be created")
    
    df_train.to_csv(os.path.join(args.path_to_save_folder, "df_train.csv"), index=False)
    df_test.to_csv(os.path.join(args.path_to_save_folder, "df_test.csv"), index=False)
    
    
if __name__ == "__main__":
    
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    
    print("\n------------")
    print("Total data preparation time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
