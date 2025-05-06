from pydantic import BaseModel,Field

DRAG_PROMPT = """You are an expert in question answering. I am going to give you one or more example triples of context, question and answer, in which the context may or may not be relevant to the question. The examples will be written.
Context (which may or may not be relevant):
<Retrieved documents>
Question: What is the place of birth of the director of film Servant’S Entrance?
Answer: Helsingfors
<Further demonstrations>
After the examples, I am going to provide another pair of context and question, in which the context may or may not be relevant to the question. I want you to answer the question. Give only the answer, and no extra commentary, formatting, or chattiness. Answer the question.
Context (which may or may not be relevant):
{documents}
Question: {question}
"""

ITER_DRAG_PROMPT = """You are an expert in question answering. I am going to give you one or more example sets of context, question, potential follow up questions and their respective answers, in which the context may or may not be relevant to the questions. The examples will be written.
Context:
<Retrieved documents>
Question: What nationality is the director of film Boggy Creek Ii: And The Legend Continues?
Follow up: Who is the director of the film Boggy Creek II: And The Legend Continues?
Intermediate answer: The director of the film Boggy Creek II: And The Legend Continues is Charles B. Pierce.
Follow up: What is the nationality of Charles B. Pierce?
Intermediate answer: The nationality of Charles B. Pierce is American.
So the final answer is: American
<Further demonstrations>
After the examples, I am going to provide another pair of context and question, in which the context may or may not be relevant to the question. I want you to answer the question. When needed, generate follow up question(s) using the format ’Follow up: X’, where X is the follow up question. Then, answer each follow up question using ’Intermediate answer: X’ with X being the answer. Finally, answer to the main question with the format ’So the final answer is: X’, where X is the final answer.
Context:
{documents}
Question: {question}"""

ITER_RETGEN_SYSTEM_PROMPT = """
Answer the questions based on given documents, you must give the answer in the format "So the final answer is".
Think step by step and answer the questions based on given documents. You must answer in JSON format with key "thought" and "answer".
""".strip()

ITER_RETGEN_MUSIQUE_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
Let's think step by step.
<Answer>:
{{
    "thought": "In Cold Blood was first published in book form by Random House. Random House was form in 2001.",
    "answer": "2011"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Let's think step by step.
<Answer>:
{{
    "thought": "The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley. Therefore, John Cranley is in charge of the city.",
    "answer": "John Cranley"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Let's think step by step.
<Answer>:
{{
    "thought": "Signal Hill is a hill which overlooks the city of St. John's. St. John's is located on the eastern tip of the Avalon Peninsula.",
    "answer": "eastern tip"
}}

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_WIKIMQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Let's think step by step.
<Answer>:
{{
    "thought": "Blind Shaft is a 2003 film, while The Mask Of Fu Manchu opened in New York on December 2, 1932. 2003 comes after 1932. Therefore, The Mask Of Fu Manchu came out earlier than Blind Shaft.",
    "answer": "The Mask Of Fu Manchu"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
Let's think step by step.
<Answer>:
{{
    "thought": "John V, Prince Of Anhalt-Zerbst was the son of Ernest I, Prince of Anhalt-Dessau. Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.",
    "answer": "12 June 1516"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Let's think step by step.
<Answer>:
{{
    "thought": "The director of El Extrano Viaje is Fernando Fernan Gomez, who was born on 28 August 1921. The director of Love In Pawn is Charles Saunders, who was born on 8 April 1904. 28 August 1921 comes after 8 April 1904. Therefore, Fernando Fernan Gomez was born later than Charles Saunders.",
    "answer": "El Extrano Viaje"
}}

Now based on the given doc, answer the question after <Question>
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_HOTPOTQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Let's think step by step.
<Answer>:
{{
    "thought": "Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter.",
    "answer": "Little Richard"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
{{
    "thought": "Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson.",
    "answer": "Chinua Achebe"
}}

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
{{
    "thought": "Remember Me Ballin' is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell's stage name, who was born in August 7, 1979, and is an American rapper.",
    "answer": "1979"
}}

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

class Iter_retgen_format(BaseModel):
    thought: str = Field(..., description="thought process")
    answer: str = Field(..., description="answer")

SELFASK_PROMPT = '''Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: '''

SELFASK_FOLLOW_PROMPT = "\nAre follow up questions needed here:"