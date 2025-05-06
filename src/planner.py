from utils import parse_list_score, parse_response_list, Experiment_Counter

class TaskPlanner:
    def __init__(self,
                experiment_counter: Experiment_Counter,
                tp: float,
                dynamic_model: str,
                backend_name: str = "openai"):
        """
        :param client: 已初始化的 OpenAI API 客户端，其方法 client.chat.completions.create 可调用。
        :param tp: 温度参数，传递给 API 调用。
        :param dynamic_model: 动态任务名称（原先在 dynamic_task 中使用，目前不直接传递给 API 调用，可用于日志或后续扩展）。
        """
        self.experiment_counter = experiment_counter
        self.tp = tp
        self.dynamic_model = dynamic_model
        self.system_prompt = "You are a specialized planner for multi-hop questions."
        self.scoring_prompt = "You a logical scoring module for sub_questions."
        if backend_name=="sglang":
            self.list_regex_extra = {"extra_body": {"regex": r'\[\s*"[^"]+"\s*(,\s*"[^"]+"\s*)*\]'}}
        elif backend_name=="vllm":
            self.list_regex_extra = {"extra_body": {"guided_regex": r'\[\s*"[^"]+"\s*(,\s*"[^"]+"\s*)*\]'}}
        else:
            self.list_regex_extra = {}
        print(f"TaskPlanner initialized with tp: {tp}")

    def generate_next_task(self, question, solved_information):
        messages = [{"role": "system", "content": self.system_prompt}]
        if solved_information:
            solved_str = "".join(
                [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
            )
            user_content = f"Question: {question}\nThe following sub_questions may be solved:\n{solved_str}Based on the information above, output a necessary remaining sub_question. Do not create new sub_question that has already been solved. If any sub_question is unknown, you can rewrite new sub_question from a different perspective that must help directly solve the original question. Only output the next sub_question string without any other information."
        else:
            user_content = f"Question: {question}\nBreak the question down and provide a crucial and simple sub_question as the next sub_question that directly helps solve the original question. Only output the next sub_question string without any other information."

        messages.append({"role": "user", "content": user_content})
        return self.experiment_counter.execute_openai_chat(model=self.dynamic_model, messages=messages, temperature=self.tp, max_tokens=512)

    def generate_bfs_tasks(self, question, solved_information, bfs_num: str):
        messages = [{"role": "system", "content": self.system_prompt}]
        if solved_information:
            solved_str = "".join(
                [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
            )
        else:
            solved_str = "No solved information yet."
        user_content = (
            f"Question: {question}\nThe following sub_questions may be solved:\n{solved_str}Based on the solved information, generate {bfs_num} sub_questions as a JSON array of strings to explore possible directions. Each sub_question should be independent and focus on a single information point. Avoid repeating sub_question already present even unknown. Only output the sub_questions array without any other information."
        )
        messages.append({"role": "user", "content": user_content})
        tasks_str = self.experiment_counter.execute_openai_chat(model=self.dynamic_model, messages=messages, temperature=self.tp, max_tokens=512, extra=self.list_regex_extra)
        return parse_response_list(tasks_str)

    def score_sub_question(self, question, sub_question, solved_information):
        messages = [{"role": "system", "content": self.scoring_prompt}]
        if solved_information:
            solved_str = "".join(
                [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
            )
        else:
            solved_str = "No solved information yet."

        user_content = f"""Please evaluate the quality of the sub_question based on the following dimensions (1-5 points):
        1. Relevance to the main question "{question}"
        2. Logical coherence
        3. Specificity of information
        Solved information:
        {solved_str}
        Question to evaluate: {sub_question}
        Provide a normalized score between 0 and 1, rounded to two decimal places. Only output the float without any other information."""

        messages.append({"role": "user", "content": user_content})
        score_str = self.experiment_counter.execute_openai_chat(model=self.dynamic_model, messages=messages, temperature=self.tp, max_tokens=32)
        return float(score_str)

    def score_tasks(self, question, tasks_list, solved_information):
        messages = [{"role": "system", "content": self.scoring_prompt}]
        user_content = f"Question: {question}\n"
        if solved_information:
            solved_str = "".join(
                [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
            )
            user_content += f"The following sub_questions may be solved:\n{solved_str}"
        
        user_content += f"""Evaluate the sub-questions below for solving the original question:
        {tasks_list}
        Check these criteriona:
        - Relevance: Are they directly related and non-redundant?
        - Completeness: Do they cover all key aspects?
        - Clarity: Are they clearly phrased?
        - Logical Coherence: Are they logically ordered and connected?
        - Feasibility: Are they answerable and well-scoped?
        - Format: Are they a valid JSON array of strings?

        Thinking carefully on each criteriona and output a normalized score between 0 and 1, two decimal places. Only output the float without any other information."""
        
        messages.append({"role": "user", "content": user_content})
        score_str = self.experiment_counter.execute_openai_chat(model=self.dynamic_model,messages=messages,temperature=0,max_tokens=32)
        return parse_list_score(score_str)

    def generate_tasks(self, question, solved_information):
        messages = [{"role": "system", "content": self.system_prompt}]
        if solved_information:
            solved_str = "".join(
                [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
            )
            user_content = f"Question: {question}\nThe following sub_questions may be solved:\n{solved_str}Based on the information above, output the necessary remaining sub_questions step by step as a JSON array of strings. Do not create new sub_question that has already been solved. If any sub_question is unknown, you can rewrite new sub_question from a different perspective that must help directly solve the original question. Elements can only be strings, not dict. Only output the sub_questions array without any other information."
        else:
            user_content = f"Question: {question}\nBreak the question down into a list of sub_questions that can help directly solve the original question step by step as a JSON array of strings. Elements can only be strings, not dict. Only output the sub_questions array without any other information."
            
        messages.append({"role": "user", "content": user_content})
        tasks_str = self.experiment_counter.execute_openai_chat(model=self.dynamic_model, messages=messages, temperature=self.tp, max_tokens=512, extra=self.list_regex_extra)
        return parse_response_list(tasks_str)

    def update_tasks(self, question, solved_information, pending_tasks):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"The question is: {question}\n"}
        ]
        solved_str = "".join(
            [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
        )
        user_content = f"Question: {question}\nThe following sub_questions may be solved:\n{solved_str}Use the information above to rewrite the remaining sub_questions: {pending_tasks}\nOnly rewrite and do not create new sub_questions. Only output the remaining sub_questions as a JSON array, the length should be the same as the original. Only output the array without any other information."

        messages.append({"role": "user", "content": user_content})
        tasks_str = self.experiment_counter.execute_openai_chat(model=self.dynamic_model, messages=messages, temperature=self.tp, max_tokens=512)
        return parse_response_list(tasks_str)