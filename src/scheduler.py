# scheduler.py
import json
from planner import TaskPlanner
from retriever import Retriever
from answerer import Answerer
import concurrent.futures
from utils import Experiment_Counter, set_prompt
from templates import DRAG_PROMPT,ITER_DRAG_PROMPT,ITER_RETGEN_HOTPOTQA_PROMPT,ITER_RETGEN_MUSIQUE_PROMPT,ITER_RETGEN_SYSTEM_PROMPT,ITER_RETGEN_WIKIMQA_PROMPT, Iter_retgen_format, SELFASK_FOLLOW_PROMPT, SELFASK_PROMPT

class Scheduler:
    def __init__(self,
                 planner: TaskPlanner,
                 retriever: Retriever,
                 answerer: Answerer,
                 experiment_counter: Experiment_Counter,
                 maxlen,
                 is_cot=False,
                 max_iter=5,
                 sub_force=False):
        
        self.planner = planner
        self.retriever = retriever
        self.answerer = answerer
        self.experiment_counter = experiment_counter
        self.maxlen = maxlen
        self.is_cot = is_cot
        self.max_iter = max_iter
        self.sub_force = sub_force
        self.intermediate_token = "Intermediate answer:"
        self.followup_token = "Follow up:"
        self.final_answer_token = "So the final answer is:"

    def _execute_sub_question(self, 
                        sub_question: str,
                        is_force: bool=False):
        
        accumulated_docs = self.retriever.retrieve_q(query=sub_question)
 
        sub_answer = self.answerer.generate_sub_answer(sub_question=sub_question, accumulated_docs=accumulated_docs, is_cot=self.is_cot, is_force=is_force)

        # 返回子答案与对应检索到的上下文，便于上层记录分步日志
        return sub_answer, accumulated_docs

    def dynamic_execute(self, question: str, dynamic_num: int):
        solved_information = []
        trace_steps = []

        def gen_list_and_scores():
            sub_questions = self.planner.generate_tasks(question, solved_information)
            if not sub_questions:# 如果没有生成子任务，返回空列表和-1
                return sub_questions, -1
            score = self.planner.score_tasks(question, sub_questions, solved_information)
            return sub_questions, score

        for i in range(self.max_iter):
            if dynamic_num > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=dynamic_num) as executor:
                    results = list(executor.map(lambda _: gen_list_and_scores(), range(dynamic_num)))
                scored_tasks = [(score, sub_questions[0]) for sub_questions, score in results]
                sub_question = max(scored_tasks, key=lambda x: x[0])[1]
            else:
                sub_questions = self.planner.generate_tasks(question, solved_information)
                if not sub_questions:
                    break
                sub_question = str(sub_questions[0])

            partial_answer, contexts = self._execute_sub_question(sub_question, is_force=self.sub_force)

            solved_information.append({"sub_question": sub_question, "answer": partial_answer, "contexts": contexts})
            trace_steps.append({"sub_question": sub_question, "contexts": contexts, "sub_answer": partial_answer})
            final_answer = self.answerer.generate_final_answer(question, solved_information, is_cot=self.is_cot)

            if final_answer and "unknown" not in final_answer.lower():
                return final_answer, trace_steps
            
        partial_answer, contexts = self._execute_sub_question(sub_question = question, is_force = True)
        trace_steps.append({"sub_question": question, "contexts": contexts, "sub_answer": partial_answer})
        return partial_answer, trace_steps

    def dynamic_bfs_execute(self, question: str, bfs_num: str, bfs_topk: int, is_pre: bool, is_llm_score: bool):
        solved_information = []
        trace_steps = []
        pending_tasks = self.planner.generate_bfs_tasks(question, solved_information, bfs_num)
        print("[Scheduler] BFS pending tasks:", pending_tasks)

        for i in range(self.max_iter):
            # 初始化任务的执行条件
            if is_pre and is_llm_score:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results1 = executor.map(self.retriever.light_retrieve_score, pending_tasks)
                    results2 = executor.map(self.planner.score_sub_question, [question]*len(pending_tasks), pending_tasks, [solved_information]*len(pending_tasks))
                scored_tasks = [(0.7 * r + 0.3 * l, task) for r, l, task in zip(results1, results2, pending_tasks)]
            elif is_pre:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results1 = executor.map(self.retriever.light_retrieve_score, pending_tasks)
                scored_tasks = [(0.6 * r, task) for r, task in zip(results1, pending_tasks)]
            elif is_llm_score:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results2 = executor.map(self.planner.score_sub_question, [question]*len(pending_tasks), pending_tasks, [solved_information]*len(pending_tasks))
                scored_tasks = [(0.4 * l, task) for l, task in zip(results2, pending_tasks)]
            else:
                # 没有启用预检索和 LLM 分数时，直接选择前 bfs_topk 个任务
                scored_tasks = [(0, task) for task in pending_tasks]

            # 根据 total_score 排序并取前 topk 个
            scored_tasks.sort(key=lambda x: x[0], reverse=True)
            filtered_tasks = [task for _, task in scored_tasks[:bfs_topk]]
            print("[Scheduler] BFS filtered tasks:", filtered_tasks)

            def _exec(sq):
                return self._execute_sub_question(sq, is_force=self.sub_force)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(_exec, filtered_tasks))

            for index, (partial_answer, contexts) in enumerate(results):
                subq = filtered_tasks[index]
                solved_information.append({"sub_question": subq, "answer": partial_answer, "contexts": contexts})
                trace_steps.append({"sub_question": subq, "contexts": contexts, "sub_answer": partial_answer})

            final_answer = self.answerer.generate_final_answer(question, solved_information, is_cot=self.is_cot)
            if "unknown" not in final_answer.lower():
                print(f"尝试次数: {i}后找到答案{final_answer}")
                return final_answer, trace_steps
            pending_tasks= self.planner.generate_bfs_tasks(question, solved_information, bfs_num)

        final_answer = self.answerer.generate_final_answer(question, solved_information, is_cot=self.is_cot, is_force=True)
        return final_answer, trace_steps

    def withoutchain_execute(self, question: str):
        solved_information = []
        trace_steps = []
        sub_question = self.planner.generate_next_task(question, solved_information)
        # 小于最大重试次数
        for i in range(self.max_iter):
            # 针对该子任务获取多个检索query（若需要），也可以直接用单一sub_question去检索
            partial_answer, contexts = self._execute_sub_question(sub_question, is_force=self.sub_force)

            solved_information.append({"sub_question": sub_question, "answer": partial_answer, "contexts": contexts})
            trace_steps.append({"sub_question": sub_question, "contexts": contexts, "sub_answer": partial_answer})
            final_answer = self.answerer.generate_final_answer(question, solved_information, is_cot=self.is_cot)

            if "unknown" not in final_answer.lower():
                return final_answer, trace_steps
            sub_question = self.planner.generate_next_task(question, solved_information)
        partial_answer, contexts = self._execute_sub_question(sub_question = question, is_force = True)
        trace_steps.append({"sub_question": question, "contexts": contexts, "sub_answer": partial_answer})
        return partial_answer, trace_steps

    def static_execute(self, question: str):
        solved_information = []
        trace_steps = []
        pending_tasks = self.planner.generate_tasks(question, solved_information)

        while pending_tasks:
            sub_question = pending_tasks[0]
            partial_answer = None
            # 对当前子任务最多尝试 3 次

            partial_answer, contexts = self._execute_sub_question(sub_question)
            solved_information.append({"sub_question": sub_question, "answer": partial_answer, "contexts": contexts})
            trace_steps.append({"sub_question": sub_question, "contexts": contexts, "sub_answer": partial_answer})
            # 将已执行的子任务出栈
            pending_tasks = pending_tasks[1:]
            # 若仍有剩余任务，则调用 update_tasks 逻辑根据当前已解任务更新后续任务（保证数量不变）

            if pending_tasks:
                pending_tasks = self.planner.update_tasks(question, solved_information, pending_tasks)
            # 检查是否能给出最终答案
        # 当所有子任务均出栈后，用已解信息生成最终答案
        final_answer = self.answerer.generate_final_answer(question, solved_information, is_force=True)
        return final_answer, trace_steps

    def longrag_execute(self, question: str):
        accumulated_docs = self.retriever.retrieve_q(query=question, is_filter=True, is_extractor=True)
        input = ''.join(accumulated_docs)

        prompt, prompt_len = set_prompt(f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:", self.maxlen)

        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt}], temperature=self.answerer.tp)
        trace_steps = [{"sub_question": question, "contexts": accumulated_docs, "sub_answer": ""}]
        return final_answer, trace_steps
    
    def long_execute(self, question: str):
        long_docs = self.retriever.retrieve_long_content(question)
        long_content = ''.join(long_docs)
        prompt, prompt_len = set_prompt(f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{long_content}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:", self.maxlen)

        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt}], temperature=self.answerer.tp)
        trace_steps = [{"sub_question": question, "contexts": long_docs, "sub_answer": ""}]
        return final_answer, trace_steps
    
    def raw_execute(self, question: str):
        if self.is_cot:
            thought_process = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": f"Think carefully step by step and give your thinking process for the following question:\n\nQuestion: {question}\nThought process:"}], temperature=self.answerer.tp, max_tokens=1000)
            
            final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": f"Question: {question}.\nThought process:{thought_process}.\nYour task is to use the thought process to answer the question. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"}], temperature=self.answerer.tp)
            return final_answer, []
            
        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": f"Only give me the answer below and do not output any other words.\n\nQuestion: {question}\nAnswer:"}], temperature=self.answerer.tp)
        return final_answer, []
    
    def baserag_execute(self, question: str):
        accumulated_docs = self.retriever.retrieve_q(query=question)
        input = ''.join(accumulated_docs)

        user_content = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"

        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": user_content}], temperature=self.answerer.tp)
        trace_steps = [{"sub_question": question, "contexts": accumulated_docs, "sub_answer": ""}]
        return final_answer, trace_steps
    
    def drag_execute(self, question: str):
        accumulated_docs = self.retriever.retrieve_q(query=question)[::-1]
        user_content = DRAG_PROMPT.format(documents=''.join(accumulated_docs), question=question)
        response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": user_content}], temperature=self.answerer.tp, extra = {"extra_body": {"guided_regex": r"Answer:.*", "stop": '\n'}})
        final_answer = response[len("Answer:"):].strip()
        trace_steps = [{"sub_question": question, "contexts": accumulated_docs, "sub_answer": ""}]
        return final_answer, trace_steps

    def iterdrag_execute(self, question: str):
        prompt_template = ITER_DRAG_PROMPT
        accumulated_docs = []
        trace_steps = []
        first_contexts = self.retriever.retrieve_q(query=question)[::-1]
        accumulated_docs.extend(first_contexts)
        # 初始问题检索上下文
        trace_steps.append({"sub_question": question, "contexts": first_contexts, "sub_answer": ""})
        iter = 1
        
        response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt_template.format(documents=''.join(accumulated_docs), question=question)}], temperature=self.answerer.tp, extra={"extra_body": {"guided_regex": r"(So the final answer is:|Follow up:).*", "stop": self.intermediate_token}}).strip()

        while iter < self.max_iter and response.startswith("Follow up:"):
            # 提取Follow up:到Intermediate answer:之间的文本作为下一个问题
            sub_question = response[len("Follow up:"):].strip()

            prompt_template += f"\nFollow up: {sub_question}"
            sub_contexts = self.retriever.retrieve_q(query=sub_question)[::-1]
            accumulated_docs.extend(sub_contexts)

            response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt_template.format(documents=''.join(accumulated_docs), question=question)}], temperature=self.answerer.tp, extra={"extra_body": {"guided_regex": r"Intermediate answer:.*", "stop": ["\n", self.final_answer_token, self.followup_token]}}).strip()
            # 记录该子问题的中间答案
            trace_steps.append({"sub_question": sub_question, "contexts": sub_contexts, "sub_answer": response[len("Intermediate answer:"):].strip()})
            
            prompt_template += f"\n{response}"
            response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt_template.format(documents=''.join(accumulated_docs), question=question)}], temperature=self.answerer.tp, extra={"extra_body": {"guided_regex": r"(So the final answer is:|Follow up:).*", "stop": self.intermediate_token}}).strip()
            iter += 1

        if not response.startswith("So the final answer is:"):
            response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": prompt_template.format(documents=''.join(accumulated_docs), question=question)}], temperature=self.answerer.tp, extra={"extra_body": {"guided_regex": r"So the final answer is:.*"}}).strip()

        long_answer = response[len("So the final answer is:"):].strip()
        user_content = f"{long_answer}\n\nAnswer the question based on the long answer above. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": user_content}], temperature=self.answerer.tp)
        return final_answer, trace_steps
    
    def iter_retgen_execute(self, question: str, dataset: str):
        if dataset == "hotpotqa":
            prompt = ITER_RETGEN_HOTPOTQA_PROMPT
        elif dataset == "musique":
            prompt = ITER_RETGEN_MUSIQUE_PROMPT
        elif dataset == "2wikimultihopqa":
            prompt = ITER_RETGEN_WIKIMQA_PROMPT
        
        query = question
        trace_steps = []
        for iter in range(self.max_iter):
            accumulated_docs = self.retriever.retrieve_q(query=query)
            documents = ''.join(accumulated_docs)
            user_content = prompt.format(documents=documents, question=question)
            response = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "system", "content": ITER_RETGEN_SYSTEM_PROMPT}, {"role": "user", "content": user_content}], temperature=self.answerer.tp, extra={"response_format":{"type":"json_schema","json_schema":{"name": "iter_retgen", "schema": Iter_retgen_format.model_json_schema()}}})

            output = Iter_retgen_format.model_validate_json(response)
            query = f"{question} {output.thought}"
            # 记录每次迭代的检索和思考（无明确子答案时，sub_answer 记录为 thought）
            trace_steps.append({"sub_question": query, "contexts": accumulated_docs, "sub_answer": output.thought})

        final_answer = output.answer
        return final_answer, trace_steps

    def selfask_execute(self, question: str):
        # 辅助函数：获取文本的最后一行
        def get_last_line(text: str) -> str:
            if "\n" in text:
                return text.split("\n")[-1]
            return text

        # 辅助函数：从生成的文本中提取子问题（后续问题）
        def extract_question(text: str) -> str:
            last_line = get_last_line(text)
            if self.followup_token in last_line:
                idx = last_line.rfind("Follow up:")
                sub_q = last_line[idx + len("Follow up:"):].strip()
                return sub_q
            return ""

        prompt_templates = [SELFASK_PROMPT, SELFASK_FOLLOW_PROMPT]

        # 构造初始提示，将用户的问题拼接到模板中
        cur_prompt = prompt_templates[0] + question + prompt_templates[1]

        ret_text = self.experiment_counter.execute_openai_chat(
            model=self.answerer.answer_model,
            messages=[{"role": "user", "content": cur_prompt}],
            temperature=self.answerer.tp,
            extra={"stop": self.intermediate_token}
        ).strip()

        trace_steps = []
        # 循环处理所有后续子问题，直到回答中不再出现“Follow up:”标识
        iter = 0
        while self.followup_token in get_last_line(ret_text) and iter < self.max_iter:
            iter += 1
            # 将当前 GPT 的回答追加到提示中，形成上下文
            cur_prompt += f"\n{ret_text}"
            sub_question = extract_question(ret_text)
            sub_answer, contexts = self._execute_sub_question(sub_question=sub_question)

            if "unknown" not in sub_answer.lower():
                # 如果获得子问题答案，则将答案追加到对话中，并保存该答案
                cur_prompt += f'\n{self.intermediate_token} {sub_answer}.'
                trace_steps.append({"sub_question": sub_question, "contexts": contexts, "sub_answer": sub_answer})
                ret_text = self.experiment_counter.execute_openai_chat(
                    model=self.answerer.answer_model,
                    messages=[{"role": "user", "content": cur_prompt}],
                    temperature=self.answerer.tp,
                    extra={"stop": self.intermediate_token}
                ).strip()
                print(f"[Scheduler] SelfAsk sub question: {sub_question}, answer: {sub_answer}")
            else:
                # 如果未获得子答案，仅追加中间答案标识，继续调用 GPT
                cur_prompt += f'\n{self.intermediate_token} '
                gpt_answer = self.experiment_counter.execute_openai_chat(
                    model=self.answerer.answer_model,
                    messages=[{"role": "user", "content": cur_prompt}],
                    temperature=self.answerer.tp,
                    extra={"stop": ["\n", self.followup_token, self.final_answer_token]}
                ).strip()
                cur_prompt += gpt_answer
                print(f"[Scheduler] SelfAsk no answer sub question: {cur_prompt}, answer: {gpt_answer}")

        # 确保回答中包含最终答案标识，如果没有，则追加该标识并重新调用 GPT
        if self.final_answer_token not in ret_text:
            cur_prompt += self.final_answer_token
            ret_text = self.experiment_counter.execute_openai_chat(
                model=self.answerer.answer_model,
                messages=[{"role": "user", "content": cur_prompt}],
                temperature=0.7,
                extra={"stop": '\n'}
            ).strip()

        full_response = cur_prompt + ret_text
        long_answer = full_response.split(self.final_answer_token)[-1]
        
        user_content = f"{long_answer}\n\nAnswer the question based on the long answer above. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
        final_answer = self.experiment_counter.execute_openai_chat(model=self.answerer.answer_model, messages=[{"role": "user", "content": user_content}], temperature=self.answerer.tp)
        return final_answer, trace_steps