# retriever.py

import json
import faiss
import os 
import torch
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import get_word_len, set_prompt, parse_response_list, Experiment_Counter

class Retriever:
    def __init__(self,
                answer_model: str,
                experiment_counter: Experiment_Counter,
                raw_data_path: str,
                vector_store_path: str,
                chunk_json_path: str,
                emb_model: SentenceTransformer,
                cross_model: AutoModelForSequenceClassification,
                cross_tokenizer: AutoTokenizer,
                id_to_rawid_path: str,
                maxlen,
                top_k1,
                top_k2,
                tp,
                device):
        """
        :param vector_store_path: 存储FAISS索引的文件路径 (e.g. ./output_dir/chunks.index)
        :param chunk_json_path: 文本块列表的JSON文件 (e.g. ./output_dir/chunks.json)
        :param model_path: SentenceTransformer模型路径或名称
        :param id_to_rawid_path: 可选，用于加载id_to_rawid映射 (e.g. ./output_dir/id_to_rawid.json)
        :param use_gpu: 是否使用GPU进行向量编码 (需SentenceTransformer支持)
        """
        self.raw_data = self._load_json(raw_data_path)
        self.vector = faiss.read_index(vector_store_path)
        self.chunks = self._load_json(chunk_json_path)
        self.id_to_rawid = self._load_json(id_to_rawid_path) if id_to_rawid_path else None
        self.emb_model = emb_model
        self.cross_model = cross_model
        self.cross_tokenizer = cross_tokenizer
        self.cross_model.eval()
        self.maxlen = maxlen
        self.top_k1 = top_k1
        self.top_k2 = top_k2
        self.experiment_counter = experiment_counter
        self.answer_model = answer_model
        self.tp = tp
        self.device = device
        self.retrieve_cnt = 0

    def _load_json(self, path: str):
        with open(path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def _vector_search(self, question):
        feature = self.emb_model.encode([question])
        distance, match_id = self.vector.search(feature, self.top_k1)
        content = [self.chunks[int(i)] for i in match_id[0]]
        self.retrieve_cnt += 1
        return content, list(match_id[0]), list(distance[0])

    def _process_docs(self, sub_question, retriever, match_id, is_filter, is_extractor):
        rerank, match_id = self._sort_section(sub_question, retriever, match_id)

        if is_filter:
            filter_content="\n".join(rerank)
            prompt, prompt_len = set_prompt(f"{filter_content}\n\nPlease combine the above information and give your thinking process for the following question:{sub_question}.", self.maxlen)
            thought_process = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=[{"role": "user", "content": prompt}], temperature=1, max_tokens=1000)
            filter_rerank = self._filter(sub_question, rerank, thought_process)

        if is_extractor:
            extractor_rerank = self._extractor(sub_question, rerank, match_id)

        if is_filter and is_extractor:
            return filter_rerank + extractor_rerank
        elif is_filter:
            return filter_rerank
        elif is_extractor:
            return rerank + extractor_rerank
        else:
            return rerank

    def retrieve_long_content(self, query: str) -> str:
        retriever, match_id, _ = self._vector_search(query)
        rerank, match_id = self._sort_section(query, retriever, match_id)
        long_docs, _ = self._s2l_doc(rerank, match_id, self.maxlen)
        return long_docs

    def light_retrieve_score(self, query: str) -> tuple[str, float]:
        retriever, match_id, distance = self._vector_search(query)
        return distance[0]

    def retrieve_q(self, query: str, multi_search: int = 1, is_filter: bool = False, is_extractor: bool = False) -> list[str]:
        unique_results = set()  # 用来存储去重后的结果
        if multi_search == 1:
            retriever, match_id, _ = self._vector_search(query)
            result = self._process_docs(query, retriever, match_id, is_filter, is_extractor)
            unique_results.update(result)
        else:
            sub_queries = self._create_multiple_queries(query, multi_search)

            for q in sub_queries:
                retriever, match_id, _ = self._vector_search(q)
                result = self._process_docs(query, retriever, match_id, is_filter, is_extractor)
                unique_results.update(result)
        return list(unique_results)

    def _sort_section(self,question, section, match_id):
        q = [question] * len(section)
        features = self.cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            scores = self.cross_model(**features).logits.squeeze(dim=1)

        sort_scores = torch.argsort(scores, dim=0, descending=True).cpu()
        result = [section[sort_scores[i].item()] for i in range(self.top_k2)]
        match_id = [match_id[sort_scores[i].item()] for i in range(self.top_k2)]
        return result, match_id

    def _s2l_doc(self, rerank:list, match_id:list, maxlen:int):
        unique_raw_id = []
        contents = []
        s2l_index = {}
        section_index = [self.id_to_rawid[str(i)] for i in match_id]

        for index, id in enumerate(section_index):
            data = self.raw_data[id]
            text = data["paragraph_text"]
            if id in unique_raw_id and get_word_len(text) < maxlen:
                continue
            if get_word_len(text) >= maxlen:
                content = rerank[index]
            else:
                unique_raw_id.append(id)
                content = text
            s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
            contents.append(content)
        return contents, s2l_index

    def _create_multiple_queries(self, sub_question: str, multi_search: int):
        messages = [{"role": "system", "content": "You are a query generation module."},{"role": "user", "content": f"Question: {sub_question}\nPlease output {multi_search} relevant queries for embedding retrieval as a JSON array. Element must be strings and phrased as a factual, declarative sentence. The multiple queries list: "}]

        multiple_queries_str = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=1000)
        
        return parse_response_list(multiple_queries_str)

    def _filter(self, sub_question, rank_docs, thought_process):
        def process_doc(doc):
            messages = [
                {"role": "user", "content": f"Given an article:{doc}\nQuestion: {sub_question}.\nThought process:{thought_process}.\nYour task is to use the thought process provided to decide whether you need to cite the article to answer this question. If you need to cite the article, set the status value to True. If not, set the status value to False. Only output the status value. The status value:"}
            ]
            status = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=1, max_tokens=1000,extra={"extra_body":{"guided_choice": ["True", "False"]}})

            return doc if status == "True" else None

        selected = []
        with ThreadPoolExecutor(max_workers=len(rank_docs)) as executor:
            futures = [executor.submit(process_doc, doc) for doc in rank_docs]
            for future in futures:
                result = future.result()
                if result:
                    selected.append(result)
        return selected

    def _extractor(self, sub_question, rank_docs, match_id):
        long_docs = self._s2l_doc(rank_docs, match_id, self.maxlen)[0]
        long_content = ''.join(long_docs)
        prompt, prompt_len = set_prompt(f"{long_content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{sub_question}", self.maxlen)

        cite_info = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=[{"role": "user", "content": prompt}], temperature=1, max_tokens=1000)

        return [cite_info]