####################################################################################
# This script is used to train recommendation engine based on Item-Item Similarity #
####################################################################################

import argparse
import time
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_embeddings", type=str, required=True,
                        help="Path to the embeddings pickle file")
    parser.add_argument("--path_to_df", type=str, required=True,
                        help="Path to the reference dataframe csv file")
    parser.add_argument("--n_random_users", type=int, default=10,
                        help="Number of random users considered to compute top N accuracy")                         
    parser.add_argument("--path_to_save_folder", type=str, required=True,
                        help="Folder to save output files")                        


    args = parser.parse_args()
    
    print("#######################################")
    print("# TRAINING STEP: ITEM-ITEM SIMILARITY #")
    print("#######################################\n")
                        
    print("----------------------")
    print("| Reading embeddings |")
    print("----------------------\n")

    print("Path to embeddings pickle file:", args.path_to_embeddings)   

    embeddings = pickle.load(open(args.path_to_embeddings, "rb"))
    df_embeddings = pd.DataFrame(embeddings)
    
    try:
        os.makedirs(args.path_to_save_folder, exist_ok=True)
    except OSError as error:
        print("Save folder could not be created")
    
    df_embeddings.to_csv(os.path.join(args.path_to_save_folder, "df_embeddings.csv"))

    print("\n-------------------------------------------")
    print("| Computing Similarity and Top N accuracy |")
    print("-------------------------------------------\n")     

    print("Path to dataframe csv file:", args.path_to_df)
    #print("N considered in top N accuracy:", args.top_n)
    print("Number of random users considered:", args.n_random_users)
    print("")
    
    df = pd.read_csv(args.path_to_df, sep=",") 
    
    n_users = df["user_id"].nunique()
    random_users = random.sample(range(0, n_users), args.n_random_users)
    random_users = np.array(random_users)    

    next_articles = []
    all_next_articles = []
    top_reco = []
    top_1_reco = []
    top_2_reco = []
    top_3_reco = []
    top_4_reco = []
    top_5_reco = []

    for user in random_users:
        
        print("Computing cosine similarity for user {} first read article".format(user))
        article_id = df[df["user_id"] == user].sort_values(by=["click_timestamp"])["article_id"].iloc[0]
        print("First read article ID:", article_id)
        next_article = df[df["user_id"] == user].sort_values(by=["click_timestamp"])["article_id"].iloc[1]
        all_next_article = df[df["user_id"] == user].sort_values(by=["click_timestamp"])["article_id"].iloc[1:].values
        print("Next read article ID:", next_article)
        print("All next articles:", all_next_article)
        next_articles.append(next_article)
        all_next_articles.append(all_next_article)
        article_embedding = df_embeddings.iloc[article_id].values.reshape(1, -1)
        
        similarities = []
        for i in tqdm(range(embeddings.shape[0])):
            cos_sim = cosine_similarity(article_embedding, df_embeddings.iloc[i].values.reshape(1, -1))
            similarities.append(cos_sim[0][0])
        
        similarities = np.array(similarities)
        
        ranked_similarities = np.sort(similarities)[::-1]
        ranked_ids = np.argsort(similarities)[::-1]    
        df_reco_sim = pd.DataFrame(list(zip(ranked_ids, ranked_similarities)), columns=(["reco_article_id", "similarity"]))
        
        thres = 0.5
        top_reco.append(df_reco_sim[df_reco_sim["similarity"] > thres]["reco_article_id"].iloc[1:].values)
        print("Number of articles with similarity > {}: {}".format(thres, len(top_reco[-1])))
        top_1_reco.append(df_reco_sim["reco_article_id"].iloc[1:2].values)
        print("Top 1 Reco:", top_1_reco[-1])
        top_2_reco.append(df_reco_sim["reco_article_id"].iloc[1:3].values)
        print("Top 2 Reco:", top_2_reco[-1])
        top_3_reco.append(df_reco_sim["reco_article_id"].iloc[1:4].values)
        print("Top 3 Reco:", top_3_reco[-1])
        top_4_reco.append(df_reco_sim["reco_article_id"].iloc[1:5].values)
        print("Top 4 Reco:", top_4_reco[-1])
        top_5_reco.append(df_reco_sim["reco_article_id"].iloc[1:6].values)
        print("Top 5 Reco:", top_5_reco[-1])
        print("")
    
    next_articles = np.array(next_articles)
    top_reco = np.array(top_reco)
    top_1_reco = np.array(top_1_reco)
    top_2_reco = np.array(top_2_reco)
    top_3_reco = np.array(top_3_reco)
    top_4_reco = np.array(top_4_reco)
    top_5_reco = np.array(top_5_reco)    
    
    accuracy_1 = np.mean(np.array([1 if next_articles[k] in top_1_reco[k] else 0 for k in range(len(next_articles))]))
    print("Top 1 accuracy on {} random users: {}".format(args.n_random_users, accuracy_1))

    accuracy_2 = np.mean(np.array([1 if next_articles[k] in top_2_reco[k] else 0 for k in range(len(next_articles))]))
    print("Top 2 accuracy on {} random users: {}".format(args.n_random_users, accuracy_2))    

    accuracy_3 = np.mean(np.array([1 if next_articles[k] in top_3_reco[k] else 0 for k in range(len(next_articles))]))
    print("Top 3 accuracy on {} random users: {}".format(args.n_random_users, accuracy_3))

    accuracy_4 = np.mean(np.array([1 if next_articles[k] in top_4_reco[k] else 0 for k in range(len(next_articles))]))
    print("Top 4 accuracy on {} random users: {}".format(args.n_random_users, accuracy_4))

    accuracy_5 = np.mean(np.array([1 if next_articles[k] in top_5_reco[k] else 0 for k in range(len(next_articles))]))
    print("Top 5 accuracy on {} random users: {}".format(args.n_random_users, accuracy_5))    

    accuracy = np.mean(np.array([1 if next_articles[k] in top_reco[k] else 0 for k in range(len(next_articles))]))
    print("Next in Reco accuracy on {} random users: {}".format(args.n_random_users, round(accuracy, 2)))
    
    accuracy_n = np.mean(np.array([1 if any(item in top_reco[k] for item in all_next_articles[k]) else 0 for k in range(len(all_next_articles))]))
    print("Any Reco in Next accuracy on {} random users: {}".format(args.n_random_users, round(accuracy_n, 2)))    
            
if __name__ == "__main__":
    
    start_time = time.time()
    main()
    time_elapsed = time.time() - start_time
    
    print("\n------------")
    print("Total recommendation step time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))     