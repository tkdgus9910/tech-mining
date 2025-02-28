# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 12:00:53 2025

@author: tmlab
"""

######################
"""
사전에 수집할 특허번호 데이터셋이 있으며,
특허번호에 따른 명세서 전문 텍스트 수집이 필요할 때, GooglePatent scrape
"""
######################

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

    #%% 02. data loading and preprocessing


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
    data_input = data.loc[data['year_application'] >= 2023, :].reset_index(drop = 1)
    
    
    #%% 03. scraping 
    
    import requests
    from bs4 import BeautifulSoup, element
    from collections import OrderedDict
    import time
    
    data_input['description'] = dict
    
    for idx, row in data.iterrows() :
        
        print(idx)
        
        if row['description'] == dict : 
            pt_id = row['번호'] #특허 번호
            
            url = f'https://patents.google.com/patent/{pt_id}/en'
            
            response = requests.get(url)
            time.sleep(3)  # 요청 간격 조정
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = OrderedDict()
            current_header = None
            
            # 첫 번째 'heading' 태그 찾기
            navigate = soup.find('heading')
            
            while navigate:
                # 현재 태그가 'heading'이며 텍스트가 대문자인 경우
                if navigate.name == 'heading' and navigate.text.isupper():
                    current_header = navigate.text.strip()
                    result[current_header] = []
                # 현재 태그가 'div'이며 current_header가 존재하는 경우
                elif navigate.name == 'div' and current_header:
                    text = navigate.get_text(strip=True)
                    if text:
                        result[current_header].append(text)
                # 'section' 태그를 만나면 탐색 종료
                elif navigate.name == 'section':
                    break
                
                # 다음 태그로 이동하기 전에 유효성 검사
                next_navigate = navigate.find_next()
                if next_navigate is None or not isinstance(next_navigate, element.Tag):
                    break
                navigate = next_navigate
            
            data_input['description'][idx] = result
    
    #%% 04-1. save_local
    
    directory = "D:/OneDrive/" # local 경로
    
    # 전처리 후 경로에 저장 
    with open(directory + 'data_preprocessed.pkl', 'wb') as file:  
        pickle.dump(data_input, file)  
    
    #%% 04-2. save hf
    
    from huggingface_hub import HfApi
    from huggingface_hub import login
    
    # login(token = "")
    
    # 업로드할 파일과 리포지토리 설정
    repo_id = "sanghyyyyyyyun/patent"
    upload_path = directory + 'data_preprocessed.pkl'  # Hugging Face Hub에 저장될 경로
    # upload_path = 'D:/OneDrive/연구/04_TPA_tfsc_draft/data/sample_HVAC.csv'  # Hugging Face Hub에 저장될 경로
    repo_filename = "sample_HVAC_description.pkl"  # Hub에 저장될 경로
    
    api = HfApi()
    
    api.upload_file(
    path_or_fileobj=upload_path,
    path_in_repo=repo_filename,
    repo_id=repo_id,
    repo_type="dataset",  # 모델 업로드인 경우 "model" 사용
    )
    
    
    # 파일삭제
    
    # repo_id = "sanghyyyyyyyun/patent"  # 데이터셋 리포지토리 ID
    # file_path = "sample_HVAC_description.csv"  # Hub에 저장될 경로
    
    # api.delete_file(
    # path_in_repo=file_path,
    # repo_id=repo_id,
    # repo_type="dataset")  # 모델 삭제 시 "model"

    