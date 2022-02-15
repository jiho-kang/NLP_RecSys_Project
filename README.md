비고: 기업 협업 프로젝트
</br>

### 목차

[1. 프로젝트 제목](#1-프로젝트-제목)</br>
[2. 프로젝트 개요](#2-프로젝트-개요)</br>
[3-1. 프로젝트 방법론 - 태깅모델](#3-1-프로젝트-방법론---태깅모델)</br>
[3-2. 프로젝트 방법론 - 추천모델](#3-2-프로젝트-방법론---추천모델)</br>
[4-1. 프로젝트 결과 및 보점 - 태깅모델](#4-1-프로젝트-결과-및-보완점---태깅모델)</br>
[4-2. 프로젝트 결과 및 보완점 - 추천모델](#4-2-프로젝트-결과-및-보완점---추천모델)</br>



# 1. 프로젝트 제목
- KoBERT를 이용한 context 태깅 추출 & 개인-정책 매칭을 위한 추천시스템 구축

</br>

# 2. 프로젝트 개요

### <기간>
- 2021.12.27 - 2022.01.19 (4주)


### <배경>
- 대한민국에는 수많은 정책이 있음에도, 정책 대상자들은 해당 정책의 존재를 알지 못하고, 기관입장에서는 정책 대상자를 선별하여 홍보하는 과정이 번거롭다.
- 대한민국 정책/사업 공고문 context에서 정책의 특성을 자동으로 추출하고, 개인의 프로필을 기반으로 정책을 추천하는 알고리즘을 만든다.

### <데이터>
- [웰로](https://www.welfarehello.com/)로부터 제공받은 데이터를 이용함.
- USER (2만개)

  성별, 나이, 거주지역(시도, 시군구), 관심지역(시도, 시군구), 학력, 직장, 가구원 유형, 결혼, 자녀, 자녀 수, 자녀 정보, 특수상황,
관심상황특성, 장애 상황, 보훈대상 상황, 예정 상황, 소득 정보, 관심 정책
- POLICY (8.8만개)

  정책ID, 정책서비스ID, 서비스명, 소관기관, 소관기관유형, 생애주기, 신청절차, 선정기준, 지원유형, 서비스목적,  지원내용, 지원대상

</br>

# 3-1. 프로젝트 방법론 - 태깅모델

- user에서 추출할 태그 feature (13개)</br>
: 성별, 나이, 시도, 시군구, 학력, 직장, 가구원 유형, 결혼, 자녀, 자녀상세, 대상특성, 관심상황특성, 중위소득, 관심정책

- policy에서 추출할 태그 feature (20개)</br>
: 정책ID, 정책서비스ID, 서비스명, 소관기관, 소관기관유형, 생애주기, 신청절차, 선정기준, 지원유형, 서비스목적,  지원내용, 지원대상</br>

### A. 키워드 기반 label 생성 [(filtering_smilarity_code.ipynb)](https://github.com/jiho-kang/NLP_RecSys_Project/blob/main/filteirng_similarity_code.ipynb)

- 방법

  1. feature별로 카테고리의 class를 대표하는 키워드를 설정
  2. context에 해당 키워드가 n개 이상일 경우 class 추출
    ``` python
    # example
    if sum(text.count(x) for x in ['임신', '임산부', '출산']) >= 3:
      text_tag.append('여성')
    ```

- 방법론 선정 이유

  1. 신경망 모델 학습에 필요한 label값이 존재하지 않기 때문에 먼저 라벨을 생성해주어야 함.
  2. 정책 도메인에 맞는 키워드와 함께 개체명 인식기를 직접 만들려고 했으나, 존재하는 개체명 인식기를 활용하는 것이 아니라 직접 만드는 것은 상당히 어려운 일.</br>
  프로젝트의 목적은 context 속 태깅 추출과 추천 모델 구축이기 때문에 for문과 if문으로 태그 추출하여 label을 생성함.
  3. 베이스라인 모델로서 사용될 수 있음.

    
### B. 추출된 label 기반 KoBERT 학습 (Pytorch) [(KoBERT_tagging_model.ipynb)](https://github.com/jiho-kang/NLP_RecSys_Project/blob/main/KoBERT_tagging_model.ipynb)
- 방법
  1. 추출하고자 하는 태깅값과 관련된 키워드를 해당 값으로 바꿈.

      ``` python
      # example
      word = ['임신', '임산부', '출산']
      tag = '여성'
      df['성별'].apply(lambda x: re.sub(word, tag, x))
      
      # 결과
      # 성별 feature의context가 '임산부를 위한 치료비 지원 정책'일 경우,
      #                         => '여성를 위한 치료비 지원 정책'로 변경됨
      ```
      
  2. A에서 추출한 결과값을 모델의 label로 주고, KoBERT 사용하여 feature별로 학습.</br>
      Bert Tokenizer로 토큰화 후 CrossEntropyLoss를 최소화하며 학습

- 방법론 선정 이유
  1. 정책 텍스트의 길이가 길고, 복잡하기 때문에 양방향 학습이 가능해야하며, 데이터가 크기 때문에 병렬처리가 가능한 KoBERT 모델을 선택.
  2. KoBERT는 한국어 학습이 진행된 모델이므로 기존 BERT의 사전학습된 bert-base-multilingual-cased 보다 성능이 좋을 것으로 예상.
  3. context에서 키워드를 변환함으로써 정확도를 높일 수 있을 것이라 예상함

</br>

# 3-2. 프로젝트 방법론 - 추천모델

### A. 유사도 기반 추천 [{filteirng_similarity_code.ipynb)](https://github.com/jiho-kang/NLP_RecSys_Project/blob/main/filteirng_similarity_code.ipynb)

- 방법

  1. USER 데이터셋과 POLICY 데이터셋의 공통된 feature 추출.
  2. 모든 feature를 One-Hot으로 만들어서 np.array(USER * feature) X np.array(feature * POLICY) 를 진행하여 최고 점수의 정책을 추천.
  <img src='https://user-images.githubusercontent.com/43432539/154008065-fd578275-ddba-499e-b572-f2bf939587cb.png' width='400' height='400'/>
  


- 방법론 선정 이유

  1. 정책의 갯수가 매우 많기 때문에 1차적으로 basic한 유사도 모델을 통해 고객이 관심을 가질만한 정책을 선별하기 위함.
  2. CF나 CB를 사용하기에는 유저가 경험한 정책의 여부를 알 수 없음.
  3. 피처별로 가중치를 다르게 줄 수 있음.


### B. Wide & Depp 기반 추천 [{filteirng_similarity_code.ipynb)](https://github.com/jiho-kang/NLP_RecSys_Project/blob/main/filteirng_similarity_code.ipynb)

</br>

# 4-1. 프로젝트 결과 및 보완점 - 태깅모델
### A. 키워드 기반 label 생성

- **결과**

  새롭게 만든 label들의 정확도가 최대 99% ~ 최소 40% 정도임.</br>
  ![image](https://user-images.githubusercontent.com/43432539/154006070-9251f826-b870-4751-a4a1-727b604a2fee.png)

  
- **한계 및 보완점**
1. 태깅이 쉬운 feature는 정확도가 높게 나왔으나, 복잡한 context를 가진 feature는 정확도가 낮게 나옴. => 정확도를 높이기 알고리즘을 보완해야 함.
2. 기업에서 제공해준 태깅 답지로 정확도를 측정했으나, 답지 자체가 100%의 정확도를 갖지 않음. 신뢰도가 어느정도인지 알 수 없음.

### B. 추출된 label 기반 KoBERT 학습 (Pytorch)

- **결과**

- **한계 및 보완점**
1. 키워드 기반으로 만든 label 자체의 정확도가 낮기 때문에 성능에 한계가 존재.
2. Input의 label을 One-Hot 형태로 변환하지 않았음.
3. label값이 여러개인 feature의 경우 어떻게 해야할지 모르겠어서 태깅모델 A을 이용함.</br>
    프로젝트를 회고하며 알게된 multi label classification 방법을 공부한다면 해결할 수 있을 것으로 보임.

</br>

# 4-2. 프로젝트 결과 및 보완점 - 추천모델
### A. 유사도 기반 추천

- **결과**

  ㅁㄴㅇㄹ
  
- **한계 및 보완점**
1. ㅁㄴㅇㄹ

### B. Wide & Depp 기반 추천

- **결과**

- **한계 및 보완점**
1. ㅁㄴㅇㄹ

</br>

# 5. 프로젝트 outtro
