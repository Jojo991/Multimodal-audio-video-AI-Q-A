import tiktoken
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from saving_data import docs
import yt_dl_video

db = DocArrayInMemorySearch.from_documents(
    docs, OpenAIEmbeddings())

retriever = db.as_retriever()

llm = ChatOpenAI(temperature = 0.0)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)



# Running Example Queries

query1 = "What is this tutorial about?"
response1 = qa_stuff.run(query1)
response1

query2 = "What is the difference between training set and test set"
response2 = qa_stuff.run(query2)
response2

query3 = "Who should watch this video?"
response3 = qa_stuff.run(query3)
response3

query4 = "Which is the greatest cricket team on earth?"
response4 = qa_stuff.run(query4)
response4

query5 = "How long is the circumference of the earth?"
response5 = qa_stuff.run(query5)
response5
