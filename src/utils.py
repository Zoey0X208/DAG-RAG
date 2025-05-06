import re
import json
from transformers import AutoTokenizer
from openai import OpenAI
import threading
set_prompt_tokenizer = AutoTokenizer.from_pretrained('/data/pretrained_models/Qwen2.5-7B-Instruct', trust_remote_code=True)
def get_word_count(text):
    regEx = re.compile('[\W]')
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    words = regEx.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([w for w in word_list if len(w.strip()) > 0])

def get_word_len(input):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)

def set_prompt(input, maxlen):
    tokenized_prompt = set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokenized_prompt) > maxlen:
         half = int(maxlen * 0.5)
         input = set_prompt_tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + set_prompt_tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return input, len(tokenized_prompt)

def parse_response_list(tasks_str):
    try:
        tasks = json.loads(tasks_str)
        if not isinstance(tasks, list):
            tasks = [tasks_str.strip()]
    except json.JSONDecodeError:
        tasks = [tasks_str.strip()]
    return tasks

def parse_list_score(score_str):

    match = re.search(r'[-+]?\d*\.\d+', score_str)
    if match:
        float_num = float(match.group())
        print("提取的浮点数为:", float_num)
    else:
        print("未找到浮点数, 返回 0.0")
        float_num = 0.0
    return float_num

class Experiment_Counter:

    def __init__(self,client: OpenAI):
        self.client = client
        # 各实验实例独立的 token 计数
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        # 用于多线程下计数更新的锁
        self._lock = threading.Lock()

    def execute_openai_chat(self, model, messages, temperature, max_tokens: int | None = None, extra: dict = {}):
        """
        调用 OpenAI 的 chat.completions.create 接口，并记录 token 使用情况。
        打印完整的 response JSON，并返回消息内容。
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra
        )
        # 打印整个 response 的 JSON 数据

        # 线程安全地更新计数
        with self._lock:
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.total_tokens += response.usage.total_tokens

        # 返回生成的消息内容
        return response.choices[0].message.content

# 示例用法：
if __name__ == "__main__":
    client = OpenAI(api_key="your-api-key", base_url="https://models.kclab.cloud")

    # 创建一个实验实例
    experiment = Experiment_Counter(client)
    test_input = ["Hello, world!", "How are you?", "What's your name?"]
    test_input = test_input * 10

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(experiment.execute_openai_chat, "qwen2.5-7b-instruct", [{"role": "user", "content": input}], 0.5, 100) for input in test_input]
        for future in as_completed(futures):
            print(future.result())
    print(f"Total completion tokens: {experiment.completion_tokens}")
    print(f"Total prompt tokens: {experiment.prompt_tokens}")
    print(f"Total tokens: {experiment.total_tokens}")