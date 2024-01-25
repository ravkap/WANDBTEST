#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random
from pathlib import Path
import tiktoken
from getpass import getpass
from rich.markdown import Markdown


# In[3]:


if os.getenv("OPENAI_API_KEY") is None:
  if any(['VSCODE' in x for x in os.environ.keys()]):
    print('Please enter password in the VS Code prompt at the top of your VS Code window!')
  os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")


# In[4]:


# we need a single line of code to start tracing langchain with W&B
#os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

# wandb documentation to configure wandb using env variables
# https://docs.wandb.ai/guides/track/advanced/environment-variables
# here we are configuring the wandb project name
#os.environ["WANDB_PROJECT"] = "llmapps"


# In[5]:


MODEL_NAME = "text-davinci-003"
# MODEL_NAME = "gpt-4"


# In[6]:


#!git clone https://github.com/wandb/edu.git


# In[7]:


from langchain.document_loaders import DirectoryLoader

def find_md_files(directory):
    "Find all markdown files in a directory and return a LangChain Document"
    dl = DirectoryLoader(directory, "**/*.md")
    return dl.load()

documents = find_md_files('C:/Users/ravik/docssample1/')
len(documents)


# In[8]:


# We will need to count tokens in the documents, and for that we need the tokenizer
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)


# In[9]:


# function to count the number of tokens in each document
def count_tokens(documents):
    token_counts = [len(tokenizer.encode(document.page_content)) for document in documents]
    return token_counts

count_tokens(documents)


# In[10]:


from langchain.text_splitter import MarkdownTextSplitter

md_text_splitter = MarkdownTextSplitter(chunk_size=1500)
document_sections = md_text_splitter.split_documents(documents)
len(document_sections), max(count_tokens(document_sections))


# In[11]:


Markdown(document_sections[0].page_content)


# In[12]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# We will use the OpenAIEmbeddings to embed the text, and Chroma to store the vectors
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(document_sections, embeddings)


# In[13]:


retriever = db.as_retriever(search_kwargs=dict(k=3))


# In[14]:


query = "Which of ammo type has the highest number in the penetration column?"
docs = retriever.get_relevant_documents(query)


# In[15]:


# Let's see the results
for doc in docs:
    print(doc.metadata["source"])


# In[16]:


from langchain.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

context = "\n\n".join([doc.page_content for doc in docs])
prompt = PROMPT.format(context=context, question=query)


# In[17]:


from langchain.llms import OpenAI

llm = OpenAI()
response = llm.predict(prompt)
Markdown(response)


# In[19]:


from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
result = qa.run(query)

Markdown(result)


# In[20]:


#import wandb
#wandb.finish()

