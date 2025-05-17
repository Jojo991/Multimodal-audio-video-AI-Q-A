from langchain.document_loaders import TextLoader

loader = TextLoader("./files/transcripts/transcript.txt")

docs = loader.load()

docs[0]