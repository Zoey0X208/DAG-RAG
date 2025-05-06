# DTS-RAG
```bash
export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""
cd APRAG
python3 src/main.py --dataset musique --policy dynamic --model gpt-3.5-turbo-ca --max_iter 5
# python3 src/main.py --dataset hotpotqa --policy iter_retgen --model qwen2.5-7b-instruct --max_iter 4
# python3 src/main.py --dataset hotpotqa --policy raw --model qwen2.5-7b-instruct --is_cot
# python3 src/main.py --dataset hotpotqa --policy long --model qwen2.5-7b-instruct --is_filter --is_extractor
# python3 src/main.py --dataset musique --model qwen2.5-7b-instruct --policy iterdrag
# python3 src/main.py --dataset 2wikimultihopqa --policy dynamic --model llama3-8b-instruct
# python3 src/main.py --dataset hotpotqa --policy dynamic --model qwen2.5-7b-instruct
# python3 src/main.py --dataset musique --policy iterdrag --model qwen2.5-7b-instruct --tp 0.7
# python3 src/main.py --dataset hotpotqa --policy bfs --model qwen2.5-7b-instruct --is_pre --is_llm_score
# python3 src/main.py --dataset hotpotqa --policy iter_retgen --model qwen2.5-7b-instruct
```

