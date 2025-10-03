import re
import json
from transformers import AutoTokenizer
import logging
from openai import OpenAI
import threading
set_prompt_tokenizer = AutoTokenizer.from_pretrained('/data/pretrained_models/Qwen2.5-7B-Instruct', trust_remote_code=True)
def get_word_count(text):
    regEx = re.compile(r'[\W]')
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    words = regEx.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([w for w in word_list if len(w.strip()) > 0])

def get_word_len(prompt_text):
    tokenized_prompt = set_prompt_tokenizer(prompt_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)

def set_prompt(prompt_text, maxlen):
    tokenized_prompt = set_prompt_tokenizer(prompt_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokenized_prompt) > maxlen:
        half = int(maxlen * 0.5)
        prompt_text = set_prompt_tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + set_prompt_tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return prompt_text, len(tokenized_prompt)

def parse_response_list(tasks_str):
    # 统一健壮处理：None、空串、非字符串、无法解析的 JSON、"Unknown" 等
    if tasks_str is None:
        return []
    if isinstance(tasks_str, (list, tuple)):
        return list(tasks_str)
    if not isinstance(tasks_str, str):
        tasks_str = str(tasks_str)
    stripped = tasks_str.strip()
    if not stripped or stripped.lower() == "unknown":
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        # 若解析到的不是列表，则退化为单元素列表
        return [stripped]
    except Exception:
        # 任意解析失败时，退化为单元素或空列表
        return [stripped] if stripped else []

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

    def execute_openai_chat(self, model, messages, temperature, max_tokens: int | None = None, extra: dict | None = None):
        """
        调用 OpenAI 的 chat.completions.create 接口，并记录 token 使用情况。
        打印完整的 response JSON，并返回消息内容。
        """
        logger = logging.getLogger(__name__)
        
        def _looks_like_jailbreak(text: str) -> bool:
            # 轻量越狱/注入特征检测，命中则兜底
            if not text:
                return False
            patterns = [
                r"(?i)jailbreak",
                r"(?i)DAN",
                r"(?i)ignore (all|previous) instructions",
                r"(?i)system prompt",
                r"(?i)prompt injection",
                r"(?i)developer mode",
                r"(?i)as an ai language model",
            ]
            return any(re.search(p, text) for p in patterns)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **(extra or {})
            )
        except Exception as e:
            # 针对 Azure 内容过滤/越狱拦截等错误，直接返回兜底回答
            err_text = str(e)
            logger.warning(
                "OpenAI chat.create failed, falling back. model=%s, temp=%s, max_tokens=%s, extra_keys=%s, error=%s",
                model,
                temperature,
                max_tokens,
                list(extra.keys()) if extra else [],
                err_text,
            )
            if ("content_filter" in err_text) or ("ResponsibleAIPolicyViolation" in err_text) or ("status: 400" in err_text):
                return "Unknown"
            # 其他错误保留原行为：抛出以便外层感知
            raise
        # 打印整个 response 的 JSON 数据

        # 线程安全地更新计数
        with self._lock:
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.total_tokens += response.usage.total_tokens
        logger.info("LLM usage: prompt=%s, completion=%s, total=%s", response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)

        # 返回生成的消息内容（若检测为越狱/注入则兜底）
        content = response.choices[0].message.content
        if _looks_like_jailbreak(content):
            logger.warning("Potential jailbreak/injection detected. Returning fallback answer.")
            return "Unknown"
        return content

# 示例用法：
if __name__ == "__main__":
    demo_client = OpenAI(api_key="your-api-key", base_url="https://models.kclab.cloud")

    # 创建一个实验实例
    experiment = Experiment_Counter(demo_client)
    samples = ["Hello, world!", "How are you?", "What's your name?"]
    samples = samples * 10

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(experiment.execute_openai_chat, "qwen2.5-7b-instruct", [{"role": "user", "content": sample_text}], 0.5, 100) for sample_text in samples]
        for future in as_completed(futures):
            print(future.result())
    print(f"Total completion tokens: {experiment.completion_tokens}")
    print(f"Total prompt tokens: {experiment.prompt_tokens}")
    print(f"Total tokens: {experiment.total_tokens}")