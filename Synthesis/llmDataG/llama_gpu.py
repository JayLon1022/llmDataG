from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from transformers import set_seed
import time
import random
import pandas as pd
import re
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import re

torch.cuda.empty_cache()
# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 本地模型llama3 8B
model_name = "Meta-Llama-3-8B-Instruct"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# 如果 tokenizer 没有 pad_token，设置一个
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

# parameter

# 未知类型
unknown_type = 'G'

# 初始自体非自体
self = pd.read_csv(f'../dataset/test_20/check/self/self.csv')
known = pd.read_csv(f'../dataset/test_20/seed/train{unknown_type}_nonself.csv')
train_set_unknown = pd.read_csv(f'../dataset/test_20/check/nonself/train/train{unknown_type}.csv')
test_set_unknown = pd.read_csv(f'../dataset/test_20/check/nonself/test/test{unknown_type}.csv')

known.to_csv("detectors_0.csv",index=False)

with open(f"coverage_results_epoch.txt", "w") as f:
    f.write(f"Unknown type: {unknown_type}\n")
    f.write(f"Number of self samples: {len(self)}\n")
    f.write(f"Number of known samples: {len(known)}\n")
    f.write(f"Number of unknown samples in train_set: {len(train_set_unknown)}\n")
    f.write(f"Number of unknown samples in test_set: {len(test_set_unknown)}\n")

# 自体半径
self_radius = 0.005

# 条数：1->n
some = "5"

# 变异轮数
round = 5

# tool

# 读取数据
def get_data(data_df):
    data = data_df.map(str).apply(lambda row: f"[{' '.join(row)}]", axis=1).tolist()
    return data

# 添加检测半径
def add_detection_radius(detectors_df, self_df,device):
    detector_coords = torch.tensor(detectors_df.values).to(device)
    self_coords = torch.tensor(self_df.values).to(device)
    distances = torch.cdist(detector_coords, self_coords)
    min_distances = distances.min(dim=1).values 

    min_distances_cpu = min_distances.cpu()  # 将 GPU 张量移回 CPU

    detectors_df['radius'] = (min_distances_cpu - self_radius).numpy()
    detectors_df['radius'] = detectors_df['radius'].clip(lower=0)
    return detectors_df


# 评估非自体覆盖率
def evaluate_nonselfcoverage(detectors_df, nonself_df, device):
    detector_coords = torch.tensor(detectors_df.values[:, :-1]).to(device) 
    nonself_coords = torch.tensor(nonself_df.values).to(device) 
    distances = torch.cdist(nonself_coords, detector_coords) 
    radii = torch.tensor(detectors_df['radius'].values).to(device).reshape(1, -1)
    covered = (distances <= radii).any(axis=1)
    covered_count = covered.sum().cpu().item()
    return covered_count

# 评估自体覆盖率
def evaluate_selfcoverage(detectors_df, self_df, device):
    detector_coords = torch.tensor(detectors_df.values[:, :-1]).to(device)
    self_coords = torch.tensor(self_df.values).to(device)
    distances = torch.cdist(self_coords, detector_coords)
    radii = torch.tensor(detectors_df['radius'].values).to(device).reshape(1, -1)
    covered = (distances <= radii).any(axis=1)
    covered_count = covered.sum().cpu().item()

    return covered_count

# 小样本生成变异评估
def evaluate_fewshot(epoch, detectors_df, self, nonself):
    detectors_df_copy = detectors_df.copy()
    detectors_radius = add_detection_radius(detectors_df_copy, self)
    # print(f"round:{epoch}\n")
    with open(f"coverage_results_epoch.txt", "a") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Number of detectors: {len(detectors_df)}")
        f.write(f"Number of non-self samples covered by detectors: {evaluate_nonselfcoverage(detectors_radius,nonself)}\n")

# 未知攻击覆盖变异评估
def evaluate_unknown(epoch, detectors_df):
    detectors_df_copy = detectors_df.copy()
    detectors_radius = add_detection_radius(detectors_df_copy, self)
    # print(f"round:{epoch}\n")
    with open(f"coverage_results_epoch.txt", "a") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Number of detectors: {len(detectors_df)}\n")
        f.write(f"Number of unknown samples in train_set covered by detectors: {evaluate_nonselfcoverage(detectors_radius,train_set_unknown)}\n")
        f.write(f"Number of unknown samples in test_set covered by detectors: {evaluate_nonselfcoverage(detectors_radius,test_set_unknown)}\n")


# 清除对话的函数
def reset() -> list:
    return []

# 调用模型生成对话的函数
def generate_new_vector(prompt: str, user_prompt: str, temp=0.7) -> list:
    try:
        messages = []
        # 添加任务提示
        messages.append({'role': 'system', 'content': prompt})
        
        # # 构建历史对话记录
        # for input_text, response_text in chatbot[-3:]:
        #     messages.append({'role': 'user', 'content': input_text})
        #     messages.append({'role': 'assistant', 'content': response_text})

        # 添加当前用户输入
        messages.append({'role': 'user', 'content': user_prompt})

        # 生成输出
        outputs = model.generate(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device),
            max_new_tokens=1024,
            temperature=temp,
            eos_token_id=[tokenizer.eos_token_id],
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # 将结果添加到对话历史
        # chatbot.append((user_prompt, response))
        

    except Exception as e:
        print(f"发生错误：{e}")
        response = f"抱歉，发生了错误：{e}"
    
    return response

def clean_value(value):
    return float(re.sub(r'[\[\]]', '', value))


# main

for epoch in range(round):
    # epoch = 1
    dataset = pd.read_csv(f"detectors_{epoch}.csv")

    # 评估覆盖
    # evaluate(epoch,dataset,self,nonself,device)

    # 获取数据
    data = get_data(dataset)
    # detectors = get_detectors(dataset)


    chatbot_history = reset()

    results = []
    for i, vector in enumerate(data):
        other_indices = random.sample([idx for idx in range(len(data)) if idx != i], 10)
        input_dataset = [data[idx] for idx in other_indices] + [data[i]]
        
        # input_dataset = data[i -10 :i + 10]
        system_prompt = f"You are a professional data analyst with expertise in vector analysis and spatial distribution. Task: You will be given a dataset: {input_dataset}. Analyze the dataset's features and its spatial distribution. Based on the provided input vector, generate new vectors that exhibit similar characteristics but fill in gaps in the feature space of the dataset. Each element of the new vectors must be a five-decimal number within the range (0, 1) and must not have zero as the last decimal place. Ensure that each generated number is meaningful and does not contain redundant zeros or repeating digits. Important: Ensure that the generated vectors represent meaningful variations that reflect the underlying spatial structure of the dataset. Provide only the output vectors, without any explanation or process details. Output Format MUST be: 1.23456, 2.34567, 3.45678, ...Note: Focus on the spatial analysis and ensure that new vectors cover areas of the feature space that are underrepresented or not captured by the current dataset."
        response = generate_new_vector(system_prompt, f"Generate exactly {some} new vectors based on the input vector: {vector}")
        # print(f"Original Vector: {vector}")
        response = response.split('assistant\n\n')
        response = response[-1]
        response = re.sub(r'(?<!\b0\.)\b[1-9]\d*\.\s*|[,\[\]]', '', response)
        # print(response)
        vectors = re.findall(r'(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b(?:\s+\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b){41})', response)
        
        # 筛除异常输出
        if(len(vectors)==0):
            continue
        
        # 检验生成质量
        check = len(vectors[0].split())
        # print(check)
        
        # 不合格再次请求
        while(check!=42):
            response = generate_new_vector(system_prompt, f"Generate exactly {some} new vectors based on the input vector: {vector}")
            response = response.split('assistant\n\n')
            response = response[-1]
            response = re.sub(r'(?<!\b0\.)\b[1-9]\d*\.\s*|[,\[\]]', '', response)
            vectors = re.findall(r'(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b(?:\s+\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b){41})', response)
            check = len(vectors[0].split())
            # print(check)
            

        for v in vectors:
            result = list(map(float, v.split()))
            # print(f"Mutated Vector: {v}")
            results.append(result)
            
    df = pd.DataFrame(results, columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41"])
    df.to_csv(f"detectors_{epoch+1}.csv", index=False)

