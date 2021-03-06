비고: 기업 협업 프로젝트
</br>

### 목차

[1. 프로젝트 제목](#1-프로젝트-제목)</br>
[2. 프로젝트 개요](#2-프로젝트-개요)</br>
[3-1. 프로젝트 방법론 - 태깅모델](#3-1-프로젝트-방법론---태깅모델)</br>
[3-2. 프로젝트 방법론 - 추천모델](#3-2-프로젝트-방법론---추천모델)</br>
[4-1. 프로젝트 결과 및 보점 - 태깅모델](#4-1-프로젝트-결과-및-보완점---태깅모델)</br>
[4-2. 프로젝트 결과 및 보완점 - 추천모델](#4-2-프로젝트-결과-및-보완점---추천모델)</br>
[5. 회고](#5-회고)</br>
[6. Next step after lesson learned](#6-next-step-after-lesson-learned)


# 1. 프로젝트 제목
### KoBERT를 이용한 context 태깅 추출 & 개인-정책 매칭을 위한 추천시스템 구축

</br>

# 2. 프로젝트 개요

### <구성원>
- 강지호
- 이남준

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
  2. 모든 feature를 One-Hot으로 만들어서 np.array(USER * feature) X np.array(feature * POLICY) 를 진행하여 최고 점수의 정책을 추천.</br>
    <img src='https://user-images.githubusercontent.com/43432539/154008065-fd578275-ddba-499e-b572-f2bf939587cb.png' width='800' height='600'/>
  
- 방법론 선정 이유

  1. 정책의 갯수가 매우 많기 때문에 1차적으로 basic한 유사도 모델을 통해 고객이 관심을 가질만한 정책을 선별하기 위함.
  2. CF나 CB를 사용하기에는 유저가 경험한 정책의 여부를 알 수 없음.
  3. 피처별로 가중치를 다르게 줄 수 있음.


### B. Wide & Depp 기반 추천 [{filteirng_similarity_code.ipynb)](https://github.com/jiho-kang/NLP_RecSys_Project/blob/main/filteirng_similarity_code.ipynb)
- 방법</br>
  1. label: 유저가 선호하는 정책(중복가능 11개)과 유저 태그가 정책데이터 특성과의 일치여부에 따라 1,0으로 부여.
      - 유저와 정책 데이터셋의 '선호정책'을 One Hot Encoding 하여 정책 데이터가 유저 선호정책(중복 선택 중) 한개라도 포함될 경우.
      - 유저와 정책 데이터셋의 중요태그(대상특성, 직장, 가구원 등)가 모두 부합할 경우.
      - 위 두가지 조건(관심정책 & 중요태그) 일치 여부에 따라 라벨 1 또는 0으로 부여.
  2. train
      - wide part(w/ cross product transformation) & deep part input 진행 및 학습.
  3. recommendation
      - 특정 유저의 특성과 전체 정책의 특성 조합에 대한 label을 예측.
      - 가장 높은 점수의 정책 5개 ~10개 추천 진행.
  
  4. 적은 샘플로 테스트 후 True에 대한 f1-score가 매우 낮게 나와서 필요 특성을 추가하고 클러스터링 후 군집 별 표본추출로 학습을 진행했다.
      - before
        ```python

        """
        _x: 유저 데이터 특성
        _y: 정책 데이터 특성
        두 가지가 모두 1이 되는 경우를 기억하기 위한 crossed_cols 설정
        """
        # wide part 특성
        wide_cols = ['성별_x', '자녀_x', 'mb_10+대상특성', 'mb_11+관심상황특성','대상특성', '대상특성상세',
               '소관기관유형', '지원유형', '지원유형상세', '신청절차', '성별_y','자녀_y']

        # wide part의 cross layer 설정
        crossed_cols = (['성별_x', '성별_y'], ['자녀_x', '자녀_y'], ['mb_10+대상특성','대상특성'])

        # deep part 특성
        embedding_cols = ['성별_x', '자녀_x', 'mb_10+대상특성', 'mb_11+관심상황특성','대상특성', '대상특성상세',
                          '소관기관유형', '지원유형', '지원유형상세', '신청절차', '성별_y','자녀_y']
        cont_cols = ['나이','대상연령시작','대상연령끝']
        ```

        ```
        # 예측 결과
                      precision    recall  f1-score   support

               False       0.93      0.76      0.84     17132
                True       0.32      0.67      0.43      2868

            accuracy                           0.75     20000
           macro avg       0.63      0.72      0.64     20000
        weighted avg       0.84      0.75      0.78     20000

        ```

      - after
        ```python

        # wide part 특성
        wide_cols = [ '성별_x', '학력_x', '직장_x', '결혼_x',
                      ' 농축수산인', '해당사항없음_x', '시도_x', '한부모가정/조손가정_x', '국가유공자_x', '북한이탈주민_x',
                      '질병/부상/질환자_x', '장애인_x', '다문화가족_x', '다자녀가정_x', '문화생활 지원_x', '주택-부동산 지원_x', '관심정책없음', '근로자 지원_x',
                      '의료 지원_x', '보육지원(만0~7세)', '개인금융지원_x', '교육지원(만8~19세)', '성인교육지원_x',
                      '기업금융지원_x', '취업 지원_x', '창업 지원_x', '시도_y',
                      '지원유형', '학력_y', '성별_y', '결혼_y', '직장_y','질병/부상/질환자_y', '국가유공자_y',
                      '한부모가정/조손가정_y', '다자녀가정_y', '해당사항없음_y', '다문화가족_y', '북한이탈주민_y', '농축수산인',
                      '장애인_y', '개인금융지원_y', '문화생활 지원_y', '취업 지원_y', '창업 지원_y', '교육지원(8~19세)',
                      '의료 지원_y', '보육지원(0~7세)', '주택-부동산 지원_y', '성인교육지원_y', '기업금융지원_y',
                      '근로자 지원_y']

        # deep part 특성
        crossed_cols = (['성별_x', '성별_y'], ['결혼_x', '결혼_y'])

        embedding_cols = ['시도_x', '학력_x', '직장_x', '결혼_x',
                      ' 농축수산인', '해당사항없음_x', '한부모가정/조손가정_x', '국가유공자_x', '북한이탈주민_x','질병/부상/질환자_x', '장애인_x', '다문화가족_x', '다자녀가정_x',
                      '문화생활 지원_x', '주택-부동산 지원_x', '관심정책없음', '근로자 지원_x','의료 지원_x', '보육지원(만0~7세)', '개인금융지원_x', '교육지원(만8~19세)', '성인교육지원_x','기업금융지원_x', '취업 지원_x', '창업 지원_x',
                      '시도_y', '소관기관유형', '지원유형', '학력_y', '성별_y', '결혼_y', '자녀', '직장_y',
                      '농축수산인', '해당사항없음_y','한부모가정/조손가정_y', '국가유공자_y', '북한이탈주민_y', '질병/부상/질환자_y', '장애인_y', '다문화가족_y', '다자녀가정_y', 
                      '개인금융지원_y', '문화생활 지원_y', '취업 지원_y', '창업 지원_y', '교육지원(8~19세)','의료 지원_y', '보육지원(0~7세)', '주택-부동산 지원_y', '성인교육지원_y', '기업금융지원_y','근로자 지원_y']
        cont_cols = ['나이','max_income_x','min_income_x','대상연령시작','대상연령끝','max_income_y', 'min_income_y'] 
        ```

        ```
        # accuracy 측정 결과

        accuracy: 0.8740
        wide and deep model accuracy: 0.873960018157959
                      precision    recall  f1-score   support

                   0       0.89      0.88      0.89    342435
                   1       0.85      0.86      0.85    257565

            accuracy                           0.87    600000
           macro avg       0.87      0.87      0.87    600000
        weighted avg       0.87      0.87      0.87    600000
        ```

- 방법론 선정 이유

  1. 정책 도메인 특성상 추천될 정책과 유저의 조건이 부합하는 것이 중요하면서도, 새로운 정책을 추천할 수 있어야 함.
  2. Wide Part는 Linear한 부분으로, 유저와 정책의 특성이 정확히 일치하는 부분에 대해 memorization. 상세화된 예측 결과를 제시할 수 있음. 과적합 발생 가능.
  3. Deep Part는 Non-Linear한 부분으로, 유저와 정책의 특성을 generalization하여 freshness를 높일 수 있음. 과적합 방지.
 
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
  1. 학습은 되는 것 같아 보이나, 한계가 존재함.</br>
    ![image](https://user-images.githubusercontent.com/43432539/154503561-911a3b6b-a3c4-4e6d-9385-dced71e20406.png)

- **한계 및 보완점**
  1. 신청절차, 성별, 자녀, 정책지원유형 등은 라벨에 중복되는 값이 없음. 그러나 그 외의 특성들은 라벨에 중복되는 값들이 있는데, 라벨값이 하나일수도, 열 개일수도 있음.
      ```
      policy_label['신청절차'] = ['무관', '무관', '여성', '남성'...] 
      policy_label['대상특성'] = ['해당사항없음', '장애인, 다문화가족', '임신부', '독거노인, 기관/시설, 장애인', '해당사항없음'...]
      ```
      </br>
      이 부분에 대해 학습 시, loss를 데이터 갯수에 맞게 그때그때 구해주려고 해보았으나, 결국 최종적으로 test에서 n개의 아웃풋 결과를 설정해줘야함.

      ```
      out = model(token_ids, valid_length, segment_ids)
      label = label[0]

      if ',' in label:          
          labels = list(map(int, label.split(',')))
          loss = 0
          for i in range(len(labels)):
            label = labels[i]
            # label = tuple(label, )
            label = torch.from_numpy(np.array([np.int32(label)])).long().to(device)
            loss += loss_fn(out, label)
            loss /= len(labels)
      else:
        label = torch.from_numpy(np.array([np.int32(label)])).long().to(device)
        loss = loss_fn(out, label)
      loss.backward()
      ```
  2. 학습에 사용된 라벨값은 기존에 for문, if문을 이용해서 키워드를 기준으로 태깅을 진행한 값.</br>
    즉, 100%의 정확도를 가지는 라벨값이 아니라, 피처마다 최소 40% 최대 98% 사이의 정확도를 가지는 라벨값이기 때문에 애초에 높은 모델 성능을 기대하기 어려움.
  3. Input의 label을 One-Hot 형태로 변환하지 않았음.
  4. label값이 여러개인 feature의 경우 어떻게 해야할지 모르겠어서 태깅모델 A을 이용함.</br>
      프로젝트를 회고하며 알게된 multi label classification 방법을 공부한다면 해결할 수 있을 것으로 보임. [참고자료](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=chrhdhkd&logNo=221469784081)

</br>

# 4-2. 프로젝트 결과 및 보완점 - 추천모델
### A. 유사도 기반 추천

- **결과**

  따로 성능을 평가할 수 있는 지표가 없기 때문에 USER의 정보를 임의로 넣었을 때 아래와 같은 POLICY가 추천됨.</br>
  ![image](https://user-images.githubusercontent.com/43432539/154010953-7ab597a9-0c78-41c8-9183-fc5cb3a5366b.png)

  
- **한계 및 보완점**
  1. USER나 POLICY context 정보의 질에 따라 다른 모델을 구축하는 방향으로 진행해보면 좋을 것 같음.</br>
    USER가 프로필을 가득 채운 경우 더 specific한 모델로, POLICY의 context의 질이 좋지 않을 경우 general한 모델로 나누는 방법.

### B. Wide & Depp 기반 추천

- **결과**
  ```
  # model predict 및 임의의 유저 추천 진행 결과


  유저 테이블 100번 유저 정보
  user1.iloc[100]

  num                       4969
  성별                          여성
  나이                          54
  시도                       서울특별시
  시군구                        강서구
  학력                      고등학교졸업
  직장                   소상공인,중소기업
  가구원             무주택 세대주,주택 세대주
  결혼                          기혼
  자녀                          없음
  자녀상세                    해당사항없음
  자녀수                         no
  mb_10                     저소득층
  mb_10+대상특성                저소득층
  mb_11                     해당없음
  mb_11+관심상황특성              None
  mb_12           중위소득 60~80% 사이
  mb_13                       no
  mb_14                       no
  mb_15                주택-부동산 지원
  Name: 100, dtype: object
  --------------------------------------------
  100번 유저 관련 추천 정책
  run.Recommendation(user1.iloc[100])


  정책ID	서비스명	prob
  0	51650	긴급복지 지원	0.979127
  1	22676	긴급복지 지원	0.942127
  2	70552	긴급복지 지원	0.942127
  3	45770	긴급복지 지원	0.942127
  4	44396	긴급복지 지원	0.942127
  5	47027	긴급 복지지원 제도	0.939212
  6	46117	저소득 가정 청소년 보호 (위문 격려금)	0.929142
  7	24786	가정위탁보호 지원	0.924070
  8	53930	소년소녀가정 및 가정위탁가정 아동 수련회비 또는 수학여행비 지원	0.909780
  9	46253	소년소녀가정 및 가정위탁보호아동지원	0.909780
  ```

- **한계 및 보완점**
  1. 학습을 위한 데이터 전처리 과정에서 메모리 부족 문제 발생.
      - 원인: 너무 많은 샘플 및 특성의 수. (USER * POLICY) = (20,000 * 88,000) = 1,760,000,000 개
        - 보완방법1) : High cardinality의 특성 중 중요도가 덜한 특성을 제외(ex: 시도, 시군구)  및 샘플 수 줄이는 방법.
        - 보완방법2) : Batch Size만큼만 불러와서 학습 수행 후, 다음 Batch를 불러오는 방법.

  2. 학습을 통한 성능 향상이 90% 이상 진행되지 않음. 
      - 원인: 특성 간 교차를 진행하는 cross product를 '결혼','성별'과 같이 쉽고 복잡하지 않은 특성들로만 사용.
        - 보완방법: 직장, 대상특성과 같은 다소 중요한 정책 조건으로 사용되는 특성을 cross product 진행.

  3. model.predict의 추천 시간이 평균 23초로 매우 오래걸림.
      - 원인: 유저데이터를 입력받고, 추천을 위한 model predict 인풋값으로의 전처리 작업에서 기존 정책 8만8천개의 데이터들과의 합을 만드는 과정에서 시간이 다소 소요됨.
        - 보완방법: 유저와 정책의 데이터 질에 따른 구분. 태그가 적절히 뽑힌 데이터그룹과 / '무관'이나 '없음'이 많은 데이터 그룹으로 나누어 진행한다면 데이터 샘플을 줄일 수 있음.

  4. 추천 진행 결과, 거의 추천 대상과 정책이 잘 맞지 않음.
      - 원인1: 알고리즘 자체에서 제가 실수해서 서비스명이나 정책ID가 잘못기재 되는 등의 휴먼 에러 가능성.
      - 원인2: 무의미한 정책 데이터셋(88000개 샘플)의 양이 너무 많음.
        - 해결방안: 1번과 같이 Query를 통해 먼저 샘플 수를 걸러내는 방법, 또는 Matrix 모델을 바탕으로 1차적으로 정책을 걸러낸 뒤, 와이드앤딥 모델로 최종 높은 score의 정책을 추천하는 방법.

  5. 기존 훈련 진행시 모든 중복태그를 one-hot 인코딩 하지 않는 경우, training은 진행되지만, 추천을 위한 전처리 과정에서 정확하게 feature 생성이 매우 번거로움.</br>
     모든 샘플로부터 경우의 수를 파악해서 feature로 생성하여 학습 데이터셋과 예측 데이터셋의 feature를 동일하게 생성 가능하지만, 시간효율이 떨어짐.
      - 해결방안: 필요시 해당 중요한 특성들에 한해서는 one-hot encoding을 통해 get_dummies를 피해준다. -> 단, 공수가 더욱 많이 들어가며, 시간 효율이 떨어질 수 있음.

</br>

# 5. 회고
- 코드리팩토링: 팀원과 함께 코드를 짜다보니 상대방이 이해하기 쉽게, 재사용이 가능하게 구현해야하며, 이왕이면 처음 코드를 짤 때부터 윤곽을 세워두고 구현해야 함을 깨달음.
- 단순히 모델을 개념적이고 이론적으로 아는 것과 실제 데이터에 적용해보는 것은 많이 달랐음. pretrained된 모델을 customize하기 위해 프레임워크 기능에 대해 더 공부해야겠다고 생각함.
- 추천시스템에는 다양한 특성이 고려되는데 이번에는 NLP를 활용했음. 추천도메인에서 CV와 NLP는 앞단에서 활용될 수 밖에 없기 때문에 CV와 NLP 모델에 대한 이해가 필요할 것 같음.

</br>

# 6. Next step after lesson learned
- [딥러닝 CNN 공부](https://github.com/jiho-kang/DL_CNN_STUDY)
  - 딥러닝 기초 이론 복습, computer vision 공부, keras 프레임워크 기능 구현을 위해.
- [원티드 프리온보딩 AI/ML 코스(NLP) 참가](https://www.wanted.co.kr/events/pre_onboarding_course_9)
  - NLP 공부, pytorch 프레임워크 공부를 위해.
