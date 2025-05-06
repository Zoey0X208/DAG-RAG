from utils import Experiment_Counter

class Answerer:
    def __init__(self,
                 experiment_counter: Experiment_Counter,
                 tp: float,
                 answer_model: str):
        
        self.experiment_counter = experiment_counter
        self.tp = tp
        self.answer_model = answer_model
        self.system_prompt = "You are a specialized answerer for multi-hop questions."
        print(f"Answerer initialized with tp: {tp}")

    # def generate_sub_answer(self, sub_question, accumulated_docs: list[str], is_cot: bool, is_force: bool):
    #     messages = [
    #         {"role": "user", "content": f"There is a question and given passages.\nQuestion: {sub_question}\nThe following are given passages.\n{accumulated_docs}\n"}
    #     ]
    #     if is_cot:
    #         messages.append({"role": "user", "content": "Give your thought process for this given question based on the information above, only output your thought process and do not output other information."})
    #         thought_process = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=1000)
    #         messages.append({"role": "assistant", "content": thought_process})
    #     if is_force:
    #         messages.append({"role": "user", "content": f"Answer the question based on the given passages. Only give me the brief answer and do not output any other words.\n\nQuestion: {sub_question}\nAnswer:"})
    #     else:
    #         messages.append({"role": "user", "content": f"Answer the question only based on the given passages. Only give me the brief answer and do not output any other words. If you do not have sufficient information to answer, please respond with 'Unknown'.\n\nQuestion: {sub_question}\nAnswer:"})

    #     return self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=256)
    
    # def generate_final_answer(self, question, solved_information: list[dict], is_cot=False, is_force=False):
    #     solved_str = "".join(
    #         [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
    #     )
    #     messages = [{"role": "user", "content": f"Original question: {question}\nThe following sub_questions may be solved:\n{solved_str}"}]

    #     if is_cot:
    #         messages.append({"role": "user", "content": "Think carefully and judge whether you can answer the original question based on the information above. Give me your thought process and do not output other information."})
    #         thought_process = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=1000)
    #         messages.append({"role": "assistant", "content": thought_process})
    #     if is_force:
    #         messages.append({"role": "user", "content": f"Answer the original question based on the given information. Only give me the brief answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"})
    #     else:
    #         messages.append({"role": "user", "content": f"Answer the original question only based on the given information. Give me the brief answer and do not output any other words, do not create any information that has not been provided. If you do not have sufficient information to answer, respond with 'Unknown'.\n\nQuestion: {question}\nAnswer:"})
    #     return self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=256)
    
    def generate_sub_answer(self, sub_question, accumulated_docs: list[str], is_cot: bool, is_force: bool):
        user_content = f"There is a question and given passages.\nQuestion: {sub_question}\nThe following are given passages.\n{accumulated_docs}\n"
        messages = []
        if is_cot:
            user_content += "Give your thought process for this given question based on the information above, only output your thought process and do not output other information."
            messages.append({"role": "user", "content": user_content})
            thought_process = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=1000)
            messages.append({"role": "assistant", "content": thought_process})
            if is_force:
                messages.append({"role": "user", "content": f"Answer the question based on the given passages. Only give me the brief answer and do not output any other words.\n\nQuestion: {sub_question}\nAnswer:"})
            else:
                messages.append({"role": "user", "content": f"Answer the question only based on the given passages. Only give me the brief answer and do not output any other words. If you do not have sufficient information to answer, please respond with 'Unknown'.\n\nQuestion: {sub_question}\nAnswer:"})
        else:
            if is_force:
                user_content += f"Answer the question based on the given passages. Only give me the brief answer and do not output any other words.\n\nQuestion: {sub_question}\nAnswer:"
                
            else:
                user_content += f"Answer the question only based on the given passages. Only give me the brief answer and do not output any other words. If you do not have sufficient information to answer, please respond with 'Unknown'.\n\nQuestion: {sub_question}\nAnswer:"
            messages.append({"role": "user", "content": user_content})

        return self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=256)
    
    def generate_final_answer(self, question, solved_information: list[dict], is_cot=False, is_force=False):
        solved_str = "".join(
            [f"sub_question: {info['sub_question']} has answer: {info['answer']}\n" for info in solved_information]
        )
        user_content = f"Original question: {question}\nThe following sub_questions may be solved:\n{solved_str}"
        messages = []

        if is_cot:
            user_content += "Think carefully and judge whether you can answer the original question based on the information above. Give me your thought process and do not output other information."
            messages.append({"role": "user", "content": user_content})
            thought_process = self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=1000)
            messages.append({"role": "assistant", "content": thought_process})
            if is_force:
                messages.append({"role": "user", "content": f"Answer the original question based on the given information. Only give me the brief answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"})
            else:
                messages.append({"role": "user", "content": f"Answer the original question only based on the given information. Give me the brief answer and do not output any other words, do not create any information that has not been provided. If you do not have sufficient information to answer, respond with 'Unknown'.\n\nQuestion: {question}\nAnswer:"})
        else:
            if is_force:
                user_content += f"Answer the original question based on the given information. Only give me the brief answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
            else:
                user_content += f"Answer the original question only based on the given information. Give me the brief answer and do not output any other words, do not create any information that has not been provided. If you do not have sufficient information to answer, respond with 'Unknown'.\n\nQuestion: {question}\nAnswer:"
            messages.append({"role": "user", "content": user_content})
        return self.experiment_counter.execute_openai_chat(model=self.answer_model, messages=messages, temperature=self.tp, max_tokens=256)