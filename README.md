# 아동청소년상담데이터



### 아동 행동 유형 특성 분류 모델
![활용모델1](https://github.com/donny95/ChildCounselingDataModel/assets/71050591/3bddcb4d-3de1-4e4f-ab2e-7ae82419211b)

<br>
- Wav2vec2.0 과 Roberta 를 사용하여 음성+텍스트 feature를 fusion 하여 분류
- 총 4개의 class 를 사용함 (협조적, 공격적, 수동적, 회피적)
- 아동 상담 특성상 불확실한 아동 어투에 대해 음성과 텍스트가 서로 상호보완하며 분류 성능을 향상시킴<br>

```
# 실행 코드 예시
cd ./01_interaction_classification/
python train_interaction.py
```
<br>


### 아동 학대 유형 분류 모델
![활용모델2](https://github.com/donny95/ChildCounselingDataModel/assets/71050591/420190e9-0cca-4359-b9b7-becedb119b61)

<br>
- 문장 분류 기능으로 뛰어난 성능을 보이는 BERT를 보완한 RoBERTa 라는 pretrained 모델을 사용하여 아동 상담 내용으로부터 아동의 학대유형을 분류
- 총 5개의 class 를 사용함 (해당없음, 방임, 성학대, 신체학대, 정서학대)<br>

```
# 실행 코드 예시
cd ./02_abuse_classification/
python train_abuse.py
```

<br>



### 아동 위기 단계 분류 모델
![활용모델3](https://github.com/donny95/ChildCounselingDataModel/assets/71050591/c407182d-67c5-41cc-999e-6922bbd561dc)

<br>
- 분류 영역에서 뛰어난 성능을 보이며, 병렬처리로 학습 및 분류 속도가 빠른 XGBoost 모델을 사용하며, 아동의 메타정보 및 임상가가 판단한 아동의 상태에 대한 체점 점수를 바탕으로 아동의 위기 단계를 분류
- Autogluon 을 사용하여 편리하게 여러 모델들로 학습한 결과를 볼 수 있음.
- 총 5개의 class 를 사용함 (정상군, 관찰필요, 상담필요, 학대의심, 응급)<br>
```
# 실행코드 예시
cd ./03_crisis_classification/
python crisis_stage_train.py
```
<br>

