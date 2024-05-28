import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

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
 
template_1 = """
<s>[INST] <<SYS>>
Act as a Machine Learning engineer who is teaching high school students.
<</SYS>>
 
{text} [/INST]
"""
 
prompt_1 = PromptTemplate(
    input_variables=["text"],
    template=template_1,
)

chain_1 = LLMChain(llm=llm, prompt=prompt_1)

template_2 = """
<s>[INST] Use the summary {summary} and give 3 examples of practical applications with 1 sentence explaining each [/INST]
"""

prompt_2 = PromptTemplate(
    input_variables=["summary"],
    template=template_2,
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

multi_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)
result = multi_chain.run(text)
print(result.strip())