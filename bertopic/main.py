# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:19:13 2025

@author: tmlab
"""

    #%% 01. data download


from huggingface_hub import HfApi, hf_hub_download
import os

# Hugging Face Hub 토큰 설정 (필요 시 설정)
# HF_TOKEN = "your_huggingface_token"

# 업로드할 파일과 리포지토리 설정
repo_id = "sanghyyyyyyyun/patent"
upload_path = "sample_HVAC.csv"  # Hugging Face Hub에 저장될 경로

# 파일 다운로드
local_file = hf_hub_download(
    repo_id=repo_id,
    filename=upload_path,
    repo_type="dataset"
)

print(f"파일 다운로드 완료: {local_file}")

    #%% 02. data loading and prepprocessing

if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    import pickle 
    
    import pandas as pd
    
    data = pd.read_csv(local_file, skiprows = 4) # 이 자리에 데이터 경로를 입력하면 됩니다 (기본값은 샘플 데이터)
    
    # 출원일 기준 2010년 이후 데이터만
    data['year_application'] = data['출원일'].apply(lambda x : int(x[0:4]))
    data_input = data.loc[data['year_application'] >= 2010, :].reset_index(drop = 1)
    
    # input 텍스트 = 제목 + 요약
    data_input['text'] = data_input['명칭(원문)'] + " " + data_input['요약(원문)']
    print(data_input['text'])
    
    docs = list(data_input['text']) 

        
    #%% 04. topic modelling_naive
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic import BERTopic
    
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    result_1 = topic_model.get_topic_info()
    
    
    #%% 04-2. topic modelling_best practice
    
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    
    ### A. pre-calculating embeddings
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa') # 필요한 임베딩 모델 사용
    embeddings = embedding_model.encode(docs) # text 임베딩
    
    umap_model = UMAP(n_neighbors=15, 
                      n_components = 5, 
                      min_dist=0.0, 
                      metric='cosine', 
                      random_state=42)
    
    embedding = umap_model.fit_transform(embeddings) # 차원 축소 수행
    
    ### B. Contolling Number of Topics
    from hdbscan import HDBSCAN

    hdbscan_model = HDBSCAN(min_cluster_size=15, 
                            metric='euclidean', 
                            cluster_selection_method='eom', 
                            prediction_data=True)
    
    ### C. Improving Default Representation
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    
    
    ### D.Additional Representations
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
    
    # KeyBERT
    keybert_model = KeyBERTInspired()
    
    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")
    # python -m spacy download en_core_web_sm
    
    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    
    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        "POS": pos_model
    }

    
    # E.Training
    from bertopic import BERTopic

    topic_model = BERTopic(
    
      # Pipeline models
      embedding_model=embedding_model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer_model,
      representation_model=representation_model,
    
      # Hyperparameters
      top_n_words=10,
      verbose=True
    )
    
    # Train model
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # Show topics
    topic_model.get_topic_info()
    
    # use one of the other topic representations, like KeyBERTInspired
    # topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
    topic_labels = {topic: "_".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["POS"].items()}
    
    topic_labels[-1] = "Outlier Topic"
    topic_model.set_topic_labels(topic_labels)
    
    result_2 = topic_model.get_topic_info()
    
    #%% 05-1. visualize topics_barchart
    import matplotlib.pyplot as plt
    
    # A. bartchart_term
    fig = topic_model.visualize_barchart()
    fig.write_html("barchart.html")
    
    import webbrowser  
    webbrowser.open("barchart.html")
    
    #%% 05-2. visualize topics_totchart
    timestamps = pd.to_datetime(data_input['출원일'])  # Convert to datetime format
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("topics_over_time.html")
    import webbrowser
    webbrowser.open("topics_over_time.html")
    