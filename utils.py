from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import os
import openai
from pinecone import ServerlessSpec
#from main import *
from pinecone import Pinecone
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
from langchain.llms import OpenAI
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'askmydoc2'

# get openai api key from platform.openai.com
OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
#-------------------------------------------------------------------------------------------------------
llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")
def retrieve_query(query,k=2):
    matching_results=vectorstore.similarity_search(query,k=k)
    return matching_results

def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response
#---------------------------------------------------------------------------------------------------------
def query_refiner(conversation, query):

    response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].text

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
