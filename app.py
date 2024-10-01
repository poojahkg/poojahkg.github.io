import streamlit as st
from langchain import PromptTemplate
from langchain.llms import Replicate
import os 
from langchain.document_loaders import YoutubeLoader, PyPDFLoader
import requests
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts import PromptTemplate
from lxml import etree
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

st.set_page_config(page_title="🦜🔗 Ask a LLM to know more about me")
st.title('🦜🔗 Ask a LLM to know more about me')
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

@st.cache_resource
def get_query_chain():
  model_name = "sentence-transformers/all-mpnet-base-v2"
  model_kwargs = {'device': 'cpu'}
  encode_kwargs = {'normalize_embeddings': False}
  hf = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
  )
  my_data=  []

  #TODO - Cache this and only do this if there is a new video
  
  links = []
  links += ["https://ayushtues.github.io/"]
  loader = WebBaseLoader(links)
  data = loader.load()
  my_data.extend(data)

  url = 'https://huggingface.co/spaces/ayushtues/personal-assistant/resolve/main/resume.pdf'
  r = requests.get(url, stream=True)

  with open('resume.pdf', 'wb') as fd:
      for chunk in r.iter_content(2000):
          fd.write(chunk)

  loader = PyPDFLoader("resume.pdf")
  pages = loader.load()
  my_data.extend(pages)


  # print(data)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 0)
  all_splits = text_splitter.split_documents(my_data)
  vectorstore = FAISS.from_documents(documents=all_splits, embedding=hf)
  retriever = VectorStoreRetriever(vectorstore=vectorstore)
  print("got retriever")
  template = """Use the following pieces of context to answer the question at the end. 
  If you don't know the answer, just say that you don't know, don't try to make up an answer. 
  Use three sentences maximum and keep the answer as concise as possible. 
  Always say "thanks for asking!" at the end of the answer. 
  {context}
  Question: {question}
  Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
  llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
  qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=retriever,
      chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
  )

  return qa_chain

def generate_response(topic, query_chain):
  result = query_chain({"query": topic})
  # print(result)
  return st.info(result['result'])

query_chain = get_query_chain()

with st.form('myform'):
  topic_text = st.text_input('Enter keyword:', '')
  submitted = st.form_submit_button('Submit')
  if submitted :
    generate_response(topic_text, query_chain)