# 1-1. 프로젝트 구조
괄호가 쳐져있는 폴더/파일은 데이터 유출 방지를 위해 github에서는 제외되었습니다.
```text
-- EDA -- __init__.py               # 입력 데이터 및 결과 분석을 위한 소스 폴더
 |      └ outputEDA_final.ipynb     # output_analysis.py로 출력된 csv 파일을 정성 분석하는 ipynb.
 |      └ output_analysis.py        # 매 validation epoch마다 예측 결과 포함한 입력값을 csv로 출력.
 | 
 └ criterion -- __init__.py         # loss 함수 정의
 | 
 └ datasets -- __init__.py          # 데이터셋 관련 소스 폴더
 |           └ data_processing.py   # 문법 교정, 불용어 처리 등 입력 데이터 필터링 코드
 |           └ datasets.py          # 커스텀 pytorch Dataset class 정의. 기타 collate_fn 및 burketing 기능 제공
 | 
 └ models -- __init__.py       # 모델 관련 소스 폴더
 |        └ auto_models.py     # huggingface의 transformers 모듈 기반, 커스텀 automodel 모델 정의
 | 
 └ utils -- __init__.py        # 기타 유틸 / 학습용 소스 폴더
 |        └ Earlystopping.py   # 커스텀 earlystopping 코드 정의 
 |        └ trainer.py         # train, validation 소스 코드 정의
 |  
 └ (NLP_dataset)  # 입력 데이터를 포함하는 폴더
 └ .gitignore 
 └ README.md
 └ final_ensemble_infer.ipynb  # output_analysis.py로 얻은 csv파일(validation 용) 또는 output.csv(test 용)로 앙상블을 하기 위한 코드
 └ hyper_tune_train.py         # sweep을 이용한 하이퍼파라미터 튜닝용 코드
 └ inference.py                # test dataset의 추론을 위한 실행 코드.  output.csv를 생성.
 └ sbert_config.yaml           # wandb 실험 세팅 파일.
 └ set_seed.py                 # 실험 재현을 위한 seed 통합 설정 코드
 └ stopwords_ver2.txt          # 불용어 리스트 파일
 └ train.py                    # 학습을 위한 실행 코드
```

# 1-2. 프로젝트 개요

- **주제**
    - 문맥적 유사도 측정 (Semantic Textual Similarity, STS) task.
    - 피어슨 상관계수로 모델 성능 평가.
    
- **개요**
 - STS 모델의 성능을 높이기 위해 2가지 방면으로 나눠서 접근함. 
   - 데이터 전처리 관점
     - 필요한 문자 제거 및 언어 교정 : hanspell 사용
     - 데이터 증식 : Masked Language Modeling을 통한 Bert-Based Synonym Replacement
   -  모델의 관점
      -  다양한 학습 방법 실험 : MLM, NLI, SimCSE 등
      -  다양한 모델 실험 : roberta, BERT, Electra 등
      -  custom loss 함수 실험 : weighted MSE, focal+MSE loss
      -  마지막으로 앙상블과 하이퍼 파라미터 튜닝을 통해 모델의 성능을 최대한 끌어냄.
    
- **역할**
  - 김준휘: 모델 실험
  - 류재환: 모델 실험
  - 설유민: 모델, data augmentation
  - 이성구: output data 분석, loss 함수 커스터마이징, 앙상블
  - 최혜원: input data 분석, 하이퍼파라미터 튜닝
 
- **활용 장비 및 재료**
    - 개발환경: VSC, Jupyter Notebook
    - 협업 툴: github **(+.discussion)**, git graph, Notion, Slack, wandb
 
# 1-3. 프로젝트 수행 절차 및 방법
![image](https://user-images.githubusercontent.com/33012030/201002408-40596fdb-46d2-4bcb-84c7-4b383d05c5ad.png)

# 1-4. 프로젝트 수행 결과
- private leader board 6등/14팀
- **등수로 표현하지 못하는 활발한 [discussion](https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions)이 있었음.**
![image](https://user-images.githubusercontent.com/33012030/200209655-752e63c1-bb74-4369-b80f-7acee60eee83.png)      
(결과에 대한 자세한 내용은 : [링크](https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions/57) 참고)

# 1-5. 자체 평가 의견
- **잘한 점들**
    - 코어 타임 외에도 열심히 실험 돌려보면서 하는 모습도 멋졌다.
    - github를 통해 버전 관리와 코드 협업을 원활히 진행했다. PR 규칙을 만들고 이를 준수했다. 큰 무리없이 다들 잘 사용했다.
    - 한 명도 빠짐없이 프로젝트에 기여를 하였다.
    - 처음하는 정성 분석이나 데이터 시각화가 잘 되었다.
    - 실험 결과를 discussion에 공유하여 각자 시도한 것들에 대한 인사이트를 모두가 공유할 수 있었다.
- **시도 했으나 잘 되지 않았던 것들**
    - 여러가지 모델을 실험해 봤으나 이해도 부족으로 잘 적용하지 못한 것은 아쉬웠다.
- **아쉬웠던 점들**
    - 프로젝트 로드맵과 일정을 먼저 짜고, 진행했어야 하는데 그렇지 못해서 후반에 시간이 부족했다.
    - 시간이 부족해서 앙상블 부분을 제대로 시도해보지 못한 것이 아쉽다.
    - 각자의 역할에 너무 몰두한 나머지 다른 사람의 작업에 피드백을 주거나 작업을 도와주는 부분이 부족했다.
- **프로젝트를 통해 배운 점 또는 시사점**
    - 앙상블 모델이 단일 모델보다 private, public 성능 모두에서 더 높은 것을 보면, 앙상블이 일반화 성능을 높이는 좋은 기법이라는 결론을 얻었다.
    - hanspell 문법 교정을 통한 데이터 필터링으로 성능을 끌어올릴 수 있었다.
    - Data augmentation 후 private 성능이 오히려 좋지 않아서, data filtering이 필요할 것 같다.

# Reference.

[참조 1] [https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions/16](https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions/16)

[참조 2] [https://wikidocs.net/22530](https://wikidocs.net/22530)

[참조 3] [https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html](https://huggingface.co/transformers/v2.11.0/main_classes/tokenizer.html)

[참조 4] [https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions/37](https://github.com/boostcampaitech4nlp2/level1_semantictextsimilarity_nlp-level1-nlp-13/discussions/37)

