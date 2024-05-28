import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts.chat import  ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

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

loader = UnstructuredMarkdownLoader("bitcoin.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

from langchain.embeddings import HuggingFaceEmbeddings
 
embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
 
db = Chroma.from_documents(texts, embeddings, persist_directory="db")
 
template = """
<s>[INST] <<SYS>>
Act as a cryptocurrency expert. Use the following information to answer the question at the end.
<</SYS>>
 
{context}
 
{question} [/INST]
"""
 
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
 
 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
 
result = qa_chain(
    "How does proof-of-work solves the majority decision making problem? Explain like I am five."
)
print(result["result"].strip())