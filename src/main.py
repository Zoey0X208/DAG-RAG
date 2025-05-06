# main_adaptive_planning_rag.py
import json
import os
from datetime import datetime
import argparse
import logging
import openai
import copy
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from planner import TaskPlanner
from retriever import Retriever
from scheduler import Scheduler
from answerer import Answerer
from utils import Experiment_Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from metric import F1_scorer, EM_scorer, ACC_scorer
logger = logging.getLogger()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

choices = [
    "glm-4-9b-chat", "mistral-7b-v0.3-instruct", "gpt-3.5-turbo-ca","qwen2.5-0.5b-instruct", "qwen2.5-3b-instruct","qwen2.5-7b-instruct","qwen2.5-14b-instruct","llama3-8b-instruct","llama3.1-8b-instruct"]
cross_model_choices = ["/data/pretrained_models/ms-marco-MiniLM-L-12-v2"]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"], default="hotpotqa", help="Name of the dataset")
    parser.add_argument('--top_k1', type=int, default=100, help="Number of candidates after initial retrieval")
    parser.add_argument('--top_k2', type=int, default=7, help="Number of candidates after reranking")
    parser.add_argument('--model', type=str, choices=choices, default="qwen2.5-7b-instruct", help="Model for generation")
    parser.add_argument('--dynamic_model', type=str, choices=['dynamic_llama', 'dynamic_qwen'], default="", help="Model for dynamic lora task planner")
    parser.add_argument('--dynamic_num', type=int, default=1, help="Number of dynamic tasks chains")
    parser.add_argument('--sub_force', action="store_true", default=False, help="is force to answer sub_question")
    parser.add_argument('--emb_model_path', type=str, default="/data/pretrained_models/multilingual-e5-large", help="Path to the embedding model")
    parser.add_argument('--cross_model_path', type=str, default=cross_model_choices[0], help="Path to the cross model")
    parser.add_argument('--r_path', type=str, default="data/corpus/processed/200_2_2", help="Path to the vector database")
    parser.add_argument('--tp', type=float, default=0.7, help="tp for task planner")
    parser.add_argument('--max_iter', type=int, default=5, help="Maximum number of retries")
    parser.add_argument('--is_cot', action="store_true", default=False, help="Whether to use COT")
    parser.add_argument('--policy', type=str, choices=["raw", "static", "bfs", "dynamic", "longrag", "long", "baserag", "withoutchain", "iterdrag", "drag", "iter_retgen", "selfask"], default="dynamic", help="Policy for multi-hop reasoning")
    parser.add_argument('--maxlen', type=int, default=30000, help="Maximum length of the passage")
    parser.add_argument('--bfs_num', type=str, default="", help="Number of bfs tasks")
    parser.add_argument('--bfs_topk', type=int, default=1, help="Number of bfs topk")
    parser.add_argument('--is_pre', action="store_true", default=False, help="Whether to preprocess the data")
    parser.add_argument('--is_llm_score', action="store_true", default=False, help="Whether to use LLM to score")
    parser.add_argument('--backend_name', type=str, choices=["openai", "sglang", "vllm"], default="openai", help="Backend name")
    parser.add_argument('--auto_experiment', action="store_true", default=False, help="Whether to run auto experiment")
    parser.add_argument('--num_threads', type=int, default=20, help="Number of threads for parallel processing")
    return parser.parse_args()

def evaluate(scheduler: Scheduler, args: argparse.Namespace):
    """评估多跳推理系统"""
    # 加载数据集
    with open(f"/data2/swh/APRAG/data/eval/{args.dataset}.json", "r") as f:
        eval_data = json.load(f)
    print("Lenght of dateval_dataset: ", len(eval_data))
        
    predictions = [None] * len(eval_data)
    answers = [None] * len(eval_data)
    questions = [None] * len(eval_data)
    
    # 定义每个任务的处理函数
    def process_data(idx, data):
        question_id = data.get("question_id", "")
        question = data.get("question", "")
        ground_truths = data.get("answers", [])
        
        # 根据策略调用对应的方法进行多跳推理
        if args.policy == "dynamic":
            final_answer = scheduler.dynamic_execute(question, args.dynamic_num)
        elif args.policy == "static":
            final_answer = scheduler.static_execute(question)
        elif args.policy == "bfs":
            final_answer = scheduler.dynamic_bfs_execute(question, args.bfs_num, args.bfs_topk, args.is_pre, args.is_llm_score)
        elif args.policy == "longrag":
            final_answer = scheduler.longrag_execute(question)
        elif args.policy == "long":
            final_answer = scheduler.long_execute(question)
        elif args.policy == "raw":
            final_answer = scheduler.raw_execute(question)
        elif args.policy == "baserag":
            final_answer = scheduler.baserag_execute(question)
        elif args.policy == "withoutchain":
            final_answer = scheduler.withoutchain_execute(question)
        elif args.policy == "drag":
            final_answer = scheduler.drag_execute(question)
        elif args.policy == "iterdrag":
            final_answer = scheduler.iterdrag_execute(question)
        elif args.policy == "iter_retgen":
            final_answer = scheduler.iter_retgen_execute(question, dataset=args.dataset)
        elif args.policy == "selfask":
            final_answer = scheduler.selfask_execute(question)
        else:
            raise ValueError("Unsupported policy: " + args.policy)
        
        return idx, question, final_answer, ground_truths

    # 使用线程池并发处理数据
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # 为每个数据项提交一个任务，同时传入数据项的索引
        futures = [executor.submit(process_data, idx, data) for idx, data in enumerate(eval_data)]
        
        # 收集所有任务结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx, question, final_answer, ground_truths = future.result()
            questions[idx] = question
            predictions[idx] = final_answer
            answers[idx] = ground_truths

    # 计算整体 F1 和 EM 分数
    f1 = F1_scorer(predictions, answers)
    em = EM_scorer(predictions, answers)
    acc = ACC_scorer(questions, predictions, answers)
    
    return f1, em, acc
    
def run_single_experiment(args: argparse.Namespace,emb_model, cross_model, cross_tokenizer):
    """运行单次实验"""
    if args.model == "qwen2.5-b-instruct":
        client = openai.OpenAI(
            api_key="sk-1234",
            base_url="https://models.kclab.cloud"
        )
    elif args.model == "gpt-3.5-turbo-ca":
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    else:
        client = openai.OpenAI(
            api_key="token-abc123",
            base_url="http://localhost:8100/v1"
        )

    experiment_counter = Experiment_Counter(client)

    if args.dynamic_model == "":
        planner = TaskPlanner(experiment_counter=experiment_counter, tp=args.tp, dynamic_model=args.model, backend_name=args.backend_name)
    else:
        planner = TaskPlanner(experiment_counter=experiment_counter, tp=args.tp, dynamic_model=args.dynamic_model, backend_name=args.backend_name)

    raw_data_path = f"data/corpus/raw/{args.dataset}.json"
    vector_store = f"{args.r_path}/{args.dataset}/vector.index"
    chunks_path = f"{args.r_path}/{args.dataset}/chunks.json"
    id_to_rawid_path = f"{args.r_path}/{args.dataset}/id_to_rawid.json"

    retriever = Retriever(answer_model = args.model, experiment_counter=experiment_counter, raw_data_path=raw_data_path, vector_store_path=vector_store, chunk_json_path=chunks_path, emb_model = emb_model, cross_model = cross_model, cross_tokenizer = cross_tokenizer, id_to_rawid_path=id_to_rawid_path, maxlen=args.maxlen, top_k1=args.top_k1, top_k2=args.top_k2, tp=args.tp, device=device)

    answerer = Answerer(experiment_counter=experiment_counter, tp=args.tp, answer_model=args.model)

    scheduler = Scheduler(planner, retriever, answerer, experiment_counter, args.maxlen, args.is_cot, args.max_iter, args.sub_force)

    f1, em, acc = evaluate(scheduler, args)

    # Add the results to the dictionary
    args_dict = vars(args)
    args_dict.update({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "f1": f1,
        "em": em,
        "acc": acc,
        "retrieve_cnt": retriever.retrieve_cnt,
        "completion_tokens": experiment_counter.completion_tokens,
        "prompt_tokens": experiment_counter.prompt_tokens,
        "total_tokens": experiment_counter.total_tokens
    })

    # Write the updated dictionary to a file
    if args.policy == "dynamic":
        with open(f"{args.dataset}_dynamic_result.json", "a") as f:
            f.write(json.dumps(args_dict) + "\n")
    else:
        with open(f"{args.dataset}_result.json", "a") as f:
            f.write(json.dumps(args_dict) + "\n")

def main():
    args = parse_args()
    emb_model = SentenceTransformer(args.emb_model_path).to(device)
    cross_tokenizer = AutoTokenizer.from_pretrained(args.cross_model_path)
    cross_model = AutoModelForSequenceClassification.from_pretrained(args.cross_model_path).to(device)
    # 当 auto_experiment 模式不为 "none" 时，自动遍历数据集
    if args.auto_experiment:
        args.num_threads = 25
        datasets = ["hotpotqa", "2wikimultihopqa", "musique"]
        
        # 定义实验配置，不包含 dataset 参数
        # recall_experiments = [
        #     {"policy": "raw"},
        #     {"policy": "raw", "is_cot": True},
        #     {"policy": "longrag"},
        #     {"policy": "long"},
        #     {"policy": "baserag"},
        #     {"policy": "iterdrag"},
        #     {"policy": "drag"},
        #     {"policy": "iter_retgen"},
        #     {"policy": "selfask"},
        # ]

        recall_experiments = [
            {"policy": "withoutchain", "model": "qwen2.5-7b-instruct"}
            # {"policy": "dynamic", "model": "llama3.1-8b-instruct"}
            # {"policy": "dynamic", "model": "mistral-7b-v0.3-instruct"}
            # {"policy": "dynamic", "model": "glm-4-9b-chat"}
        ]
        # recall_experiments = [
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 1, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 2, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 3, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 4, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 6, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 7, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 12, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/500_2_2", "top_k2": 3, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/500_2_2", "top_k2": 5, "max_iter": 5, "dynamic_model": ""},
        # ]

        # temp_experiments = [{"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 5, "dynamic_model": "dynamic_qwen"}]*20

        for exp_index, exp in enumerate(recall_experiments):
            for dataset in datasets:
                current_args = copy.deepcopy(args)
                current_args.dataset = dataset

                for key, value in exp.items():
                    setattr(current_args, key, value)
                
                print(
                    f"Running experiment {exp_index+1}/{len(recall_experiments)} for dataset {dataset} "
                    f"with configuration: {exp}"
                )
                run_single_experiment(current_args, emb_model, cross_model, cross_tokenizer)

    else:
        # 非自动实验模式，直接运行单次实验（数据集采用传入的参数）
        run_single_experiment(args,emb_model, cross_model, cross_tokenizer)

if __name__ == "__main__":
    main()
        # recall_experiments = [
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 2, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 3, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 4, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 6, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 7, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/200_2_2", "top_k2": 12, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/500_2_2", "top_k2": 3, "max_iter": 5, "dynamic_model": ""},
        #     {"r_path": "data/corpus/processed/500_2_2", "top_k2": 5, "max_iter": 5, "dynamic_model": ""},
        # ]
        # recall_experiments.extend([{"r_path": "data/corpus/processed/200_2_2", "top_k2": 7, "max_iter": 5, "dynamic_model": "dynamic_qwen"}]*20)
        # 外层遍历每个实验配置，内层遍历每个数据集，实现交错运行