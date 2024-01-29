"""
crisis classification 을 위해 가장 먼저 진행하는 단계 입니다.

json file(원천데이터)을 기반으로 위기단계 분류를 위한 csv 데이터로 변환하는 코드 입니다.
make_csv.py 파일을 실행하여 json 데이터인 원천데이터를 학습/평가 가능한 csv 형태의 파일로 추출할 수 있습니다.

"""

import os
import json
import pandas as pd



###########################################################################################

## 원천데이터 json 경로 (학습시)
# json_dir = '/workspace/datasets/out/'
## csv 저장 경로 및 저장명
# save_path = '/workspace/03_crisis_classification/datasets/crisis_total.csv'

## 원천데이터 json 경로 (평가시)
json_dir = '/workspace/03_crisis_classification/datasets/테스트용원천데이터/'
## csv 저장 경로 및 저장명
save_path = '/workspace/03_crisis_classification/datasets/유효성검증용.csv'





###########################################################################################33


final_df = pd.DataFrame()

for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_dir, filename)
        file_size = os.path.getsize(json_file_path) / 1024 
        
        if file_size <= 3:  # 3KB 용량 미달시 넘기기
            continue
        
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        
        
        info = json_data["info"]
        gender = info["성별"]
        age = info["나이"]
        crisis_level = info["위기단계"]
        group = info["유형구분"]
        enviroment = info['가정환경']
        class_level = info['학년']
        interaction_label = info["상호작용 특성(종합)"]
        abuse_label = info['학대의심']
            
        first_list_item = json_data["list"]
        
        
        list_max_num = len(first_list_item)
          
        for i in range(list_max_num) :
            
            text_sector = first_list_item[i]
            title = text_sector.get('문항') #신체적불편감 포함
            
            
            if "list" in text_sector.keys() :
                problem_list = text_sector.get("list")
                for item in problem_list : 
                    problem_name = item.get('항목')
                         
                    if problem_name == "통증" :
                        pain = problem_name
                        pain_score = item.get('점수')
                        
                    if problem_name == "신체손상" :
                        physical_injury = problem_name
                        physical_injury_score = item.get('점수')
                        
                    if problem_name == "즐거움" :
                        joy = problem_name
                        joy_score = item.get('점수')
                        
                    if problem_name == "분노/짜증" :
                        anger = problem_name
                        anger_score = item.get('점수')
                        
                    if problem_name == "수면" :
                        sleep = problem_name
                        sleep_score = item.get('점수')
                        sleep_time = item.get('수면시간')
                        
                    if problem_name == "아버지" :
                        father_problem = problem_name
                        father_problem_score = item.get('점수')
                        
                    if problem_name == "어머니" :
                        mother_problem = problem_name
                        mother_problem_score = item.get('점수')
                        
                    if problem_name == "기타 보호자" :
                        others_problem = problem_name
                        others_problem_score = item.get('점수')
                        
                    if problem_name == "형제자매" :
                        siblings = problem_name
                        siblings_score = item.get('점수')
                        
                    if problem_name == "친구" :
                        friends = problem_name
                        friends_score = item.get('점수')
                        
                    if problem_name == "교사" :
                        teacher = problem_name
                        teacher_score = item.get('점수')
                        
                    if problem_name == "걱정" :
                        worry = problem_name
                        worry_score = item.get('점수')
                        
                    if problem_name == "행복" :
                        happiness = problem_name
                        happiness_score = item.get('점수')
                        
                    if problem_name == "미래/진로" :
                        future = problem_name
                        future_score = item.get('점수')
                        
                    if problem_name == "방임" :
                        neglect = problem_name
                        neglect_score = item.get('점수')
                        
                    if problem_name == "정서학대" :
                        emotional_abuse = problem_name
                        emotional_abuse_score = item.get('점수')
                        
                    if problem_name == "신체학대" :
                        physical_abuse = problem_name
                        physical_abuse_score = item.get('점수')
                        
                    if problem_name == "성학대" :
                        sexual_abuse = problem_name
                        sexual_abuse_score = item.get('점수')
                        
                    if problem_name == "가정폭력" :
                        domestic_violence = problem_name
                        domestic_violence_score = item.get('점수')
                        
                    if problem_name == "학교폭력" :
                        school_violence = problem_name
                        school_violence_score = item.get('점수')
                        
                    if problem_name == "자해/자살" :
                        self_harm = problem_name
                        self_harm_score = item.get('점수')
                        
                    if problem_name == "트라우마" :
                        trauma = problem_name
                        trauma_score = item.get('점수')
                        
                    if problem_name == "가출경험 및 가출중 정황" :
                        runaway = problem_name
                        runaway_score = item.get('점수')
                          
   
        data = {
                "file" : [filename],
                "crisis_level" : [crisis_level],
                "gender": [gender],
                "age": [age],
                "class_level" : [class_level],
                "interaction_label": [interaction_label],
                "abuse_label" : [abuse_label],
                "group": [group],
                "enviroment": [enviroment],
                "pain" : [pain_score],
                "physical_injury" : [physical_injury_score],
                "joy" : [joy_score],
                "anger": [anger_score],
                "sleep": [sleep_score],
                "father_problem": [father_problem_score],
                "mother_problem": [mother_problem_score],
                "ohthers": [others_problem_score],
                "siblings": [siblings_score],
                "friends": [friends_score],
                "teacher": [teacher_score],
                "worry": [worry_score],
                "happiness": [happiness_score],
                "future": [future_score],
                "neglect": [neglect_score],
                "emotional_abuse": [emotional_abuse_score],
                "physical_abuse": [physical_abuse_score],
                "sexual_abuse": [sexual_abuse_score],
                "domestic_violence": [domestic_violence_score],
                "school_violence": [school_violence_score],
                "self_harm": [self_harm_score],
                "runaway": [runaway_score],
                }

    df = pd.DataFrame(data)
    final_df = pd.concat([final_df, df], ignore_index=True)
print(final_df)

print(final_df.isnull().sum())
## csv 저장 경로 설정
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print("done")