import argparse
import json
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from openai import AzureOpenAI

client = AzureOpenAI(api_key="dac9cddf80e14e49b0afb1e6f8401351",azure_endpoint="https://ustc-law-gpt4-3.openai.azure.com",api_version="2024-02-15-preview")
def execute_openai_chat(client: AzureOpenAI, model, messages, temperature = 0, max_tokens: int | None = None, extra: dict = {}):
    """
    调用 OpenAI 的 chat.completions.create 接口，并记录 token 使用情况。
    打印完整的 response JSON，并返回消息内容。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra
    )

    return response.choices[0].message.content

def normalize_answer(s):
    """
    标准化答案文本：转小写、去除标点、冠词和多余空格
    对 None 或非字符串输入做安全处理。
    """
    # 将 None 与非字符串安全转换
    if s is None:
        s = ""
    else:
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower() if isinstance(text, str) else ""
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """
    计算单个预测与标准答案之间的 F1 分数
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth):
    """
    先对答案进行标准化，再计算 F1 分数
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def F1_scorer(predictions, answers):
    """
    针对整个数据集计算平均 F1 分数
    predictions: 模型预测答案列表
    answers: 标准答案列表（每个元素可能包含多个正确答案）
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def exact_match_score(prediction, ground_truth):
    """
    判断预测答案是否与标准答案完全一致（标准化后）
    """
    if normalize_answer(prediction) in normalize_answer(ground_truth):
        return 1
    return 0

def EM_scorer(predictions, answers):
    """
    针对整个数据集计算平均 Exact Match 分数
    """
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, exact_match_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

acc_prompt = """In the following task, you are given a Question, a model Prediction for the Question, and a Ground-truth Answer to the Question. You should decide whether the model Prediction implies the Ground-truth Answer.
Question
{question}
Prediction
{model_output}
Ground-truth Answer
{answer}
Does the Prediction imply the Ground-truth Answer? Output Yes or No:"""


def execute_openai_chat(client: AzureOpenAI, model, messages, temperature = 0, max_tokens: int | None = None, extra: dict = {}):
    """
    调用 OpenAI 的 chat.completions.create 接口，并记录 token 使用情况。
    打印完整的 response JSON，并返回消息内容。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra
    )

    return response.choices[0].message.content


def ACC_scorer(questions, predictions, answers):
    """
    针对整个数据集计算 Accuracy 分数
    """
    total_score = 0.
    tasks = []

    # Prepare tasks for parallel execution
    with ThreadPoolExecutor(max_workers=10) as executor:
        for question, prediction, answer in zip(questions, predictions, answers):
            prompt = acc_prompt.format(question=question, model_output=prediction, answer=' '.join(answer)) # every answer is a ground truth list
            task = executor.submit(execute_openai_chat, client, "gpt-4o", [{"role": "user", "content": prompt}], 0, 32)
            tasks.append(task)

    # Collect results and compute accuracy
    for task in tasks:
        result = task.result()
        if "yes" in result.lower():
            total_score += 1

    return round(100 * total_score / len(questions), 2)

def ACC_scorer_single(question, prediction, answer):
    """
    针对整个数据集计算 Accuracy 分数
    """
    tasks = []

    # Prepare tasks for parallel execution
    with ThreadPoolExecutor(max_workers=1) as executor:
        prompt = acc_prompt.format(question=question, model_output=prediction, answer=' '.join(answer)) # every answer is a ground truth list
        task = executor.submit(execute_openai_chat, client, "gpt-4o", [{"role": "user", "content": prompt}], 0, 32)
        tasks.append(task)

    # Collect results and compute accuracy
    for task in tasks:
        result = task.result()
        if "yes" in result.lower():
            return True
    return False


if __name__ == "__main__":
    questions = ["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Italy?"]
    predictions = ["Par", "Berlin", "ome"]
    answers = ["Paris", "Berlin", "Rome"]
    print(ACC_scorer(questions, predictions, answers))
    question = "What is the capital of France?"
    prediction = "Paris"
    answer = "Berlin"
    print(ACC_scorer_single(question, prediction, answer))