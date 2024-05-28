import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts.chat import  ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage

MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)
 
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15


text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

text = "Explain what are Deep Neural Networks in 2-3 sentences"

template = "Act as an experienced high school teacher that teaches {subject}. Always give examples and analogies"
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template),
        HumanMessage(content="Hello teacher!"),
        AIMessage(content="Welcome everyone!"),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)

messages = chat_prompt.format_messages(
    subject="Artificial Intelligence", text="What is the most powerful AI model?"
)
result = llm.predict_messages(messages)
print(result.content)
