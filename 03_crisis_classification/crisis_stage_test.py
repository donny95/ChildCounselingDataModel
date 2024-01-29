"""

위기단계분류 평가 를 위한 코드 입니다.
모델의 웨이트(가중치)는 weight 폴더에 저장되어 있습니다.
평가용데이터는 "평가용데이터셋.csv"를 사용할 수 있으며, 이 데이터는 make_csv.py 코드를 실행하여 원천데이터(json)으로부터 얻을 수 있습니다.
Jupyter notebook을 사용하여 시각화 자료를 볼 수 있습니다. (cirisis_stage.ipynb 참조)


"""


import pandas as pd
import sys
import numpy as np
import warnings
import matplotlib.pylab as plt
import koreanize_matplotlib
import seaborn as sns

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
import sklearn.metrics as metrics
from autogluon.tabular import TabularDataset, TabularPredictor
warnings.filterwarnings('ignore')

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)

    def flush(self):
        for file in self.files:
            file.flush()
 
logfile = open("./test_logfile_03.txt", "a")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, logfile)    

####################################################################################################

# 모델 웨이트가 있는 경로 설정 - ./crirsis_level_classification/weight/
predictor = TabularPredictor.load(path="/workspace/03_crisis_classification/weight/model/", require_py_version_match=False)

## make_csv.py 를 실행하여 생성한 평가용 데이터셋 load
df1 = pd.read_csv("/workspace/03_crisis_classification/datasets/유효성검증용.csv")

## 학습과 동일한 randomstate 42 사용하여 split
# train_df, valtest = train_test_split(df1, train_size=0.8, stratify=df1['crisis_level'], random_state=42)
# val_df, test_df = train_test_split(valtest, train_size=0.5, random_state=42)
# df1 = test_df.copy()


## 개별 결과값 csv 저장 경로 설정
save_path = "/workspace/03_results.csv"

####################################################################################################



# 평가(test)에 불필요한 file 컬럼 제거
df = df1.drop(columns=["file"])
print("Test dataset example")
print(df.head())


prediction = predictor.predict(df)
cm = metrics.confusion_matrix(df['crisis_level'], prediction)

print("Test confusion matrics")
print(cm)
print("XGBoost F1 socre ")
print(predictor.evaluate(data=df, detailed_report=False, model='XGBoost'))

df1['prediction'] = prediction
result_df = df1[['file', 'crisis_level', 'prediction']]
#result_df['file'] = result_df['file'].apply(lambda x: x.replace('.mp3', '.json'))

result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
print("Results saved to CSV.")


sys.stdout = original_stdout
logfile.close()