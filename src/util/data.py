import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import kagglehub
import pandas as pd
import streamlit as st
import sqlite3
from config import data_path, db_path, file_name,CHUNK_OVERLAP, CHUNK_SIZE, UNDERSTANDING_FRAUD_PDF_URL, sqlite, sqlite_table, FRAUD_DATA_KAGGLE

os.makedirs(data_path, exist_ok=True)
os.makedirs(db_path, exist_ok=True)


class Pipe:
  def __init__(self, f):self.f = f
  def __ror__(self, x):return self.f(x)
  
  
@Pipe
def download_pdf(url):
  file_path = os.path.join(data_path, file_name)

  response = requests.get(url, stream=True)

  if response.status_code == 200:
      with open(file_path, "wb") as file:
          for chunk in response.iter_content(chunk_size=1024):
              if chunk:
                  file.write(chunk)

          return file_path
  else:
      raise Exception("Failed to download PDF")
  
@Pipe
def chunk_pdf(file_path):
  loader = PyPDFLoader(file_path)
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=CHUNK_SIZE,
      chunk_overlap=CHUNK_OVERLAP,
  )

  return text_splitter.split_documents(documents)

@Pipe
def save_chunks(chunks):

  vectordb = Chroma.from_documents(
      documents=chunks,
      embedding=st.session_state.embedding,
      persist_directory=db_path
  )

  return vectordb

@Pipe
def download_kaggle_data(dataset):
  return kagglehub.dataset_download(dataset)

@Pipe
def save_kaggle_data(kaggle_path):
  df = pd.read_csv(os.path.join(kaggle_path, "fraudTest.csv"))

  conn = sqlite3.connect(os.path.join(db_path, sqlite))

  df.to_sql(
      sqlite_table,
      conn,
      if_exists="replace",
      index=False)
  
  conn.close()

  return os.path.join(db_path, sqlite)

def init_data_pipeline():
  return FRAUD_DATA_KAGGLE | download_kaggle_data | save_kaggle_data,  UNDERSTANDING_FRAUD_PDF_URL |download_pdf | chunk_pdf | save_chunks