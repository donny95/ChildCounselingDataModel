# -*- coding: utf-8 -*-
# coding: utf-8

import os
import json
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor # 속도를 높이기 위한 병렬프로세스 라이브러리

def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%M:%S.%f')
    seconds = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return seconds

def process_file(json_file_path, wav_dir, save_new_dir):
    merged_audio = None
    wav_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    first_list_item = json_data["list"]
    list_max_num = len(first_list_item)

    for i in range(list_max_num):
        text_sector = first_list_item[i]
        audio_list = text_sector.get("list")
        
        try:
            for in_audio in audio_list:
                audio_text = [item for item in in_audio['audio'] if item['type'] == 'A']

                if not audio_text:
                    print(f"No {wav_name} file!")
                    pass

                for item in audio_text:
                    # text_values = item['text']
                    start_time = item.get('start')
                    end_time = item.get('end')

                    try:
                        start_time = time_to_seconds(start_time)
                        end_time = time_to_seconds(end_time)
                    except ValueError:
                        continue

                    waveform, sample_rate = torchaudio.load(os.path.join(wav_dir, f"{wav_name}.mp3"))

                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    trimmed_waveform = waveform[:, start_sample:end_sample]

                    if merged_audio is None:
                        merged_audio = trimmed_waveform
                    else:
                        merged_audio = torch.cat((merged_audio, trimmed_waveform), dim=1)
        except KeyError :
            # print(f"{wav_name}.json : no audio!")
            pass
            

    try:
        sf.write(os.path.join(save_new_dir, f"{wav_name}.mp3"), merged_audio[0].numpy(), sample_rate)
    except TypeError:
        print(f"{wav_name} :::::::::::: No save file! ")



######################################################################################################################
## 경로설정

# 평가용 원천데이터 json 경로
json_dir = '/workspace/01_interaction_classification/datasets/테스트용라벨데이터/'

# 평가용 원천데이터 mp3 경로
wav_dir = '/workspace/01_interaction_classification/datasets/테스트용원천데이터/'

# 평가용 원천데이터로부터 추출한 아동음성 mp3 가 저장될 경로
save_new_dir = '/workspace/01_interaction_classification/datasets/테스트용원천데이터_cut/'

######################################################################################################################

json_files = [os.path.join(json_dir, filename) for filename in os.listdir(json_dir) if filename.endswith('.json')]


print("Making mp3 files......")
with ProcessPoolExecutor() as executor:
    executor.map(process_file, json_files, [wav_dir] * len(json_files), [save_new_dir] * len(json_files))

print("Done!")