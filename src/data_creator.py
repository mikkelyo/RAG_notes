from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import itertools
import re

import random
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

# for synthetic testset
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENAI_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
PINECONE_KEY = os.getenv("PINECONE_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_KEY
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec       # used to create pinecone index if not exists
from langchain_openai.embeddings import OpenAIEmbeddings

class DataCreator():
    def __init__(self):
        pass
        
    def search_pdf_files(self, directory: str, shuffle: bool):

        # Count files in folder
        for foldername in os.listdir(directory):
            folder_path = os.path.join(directory, foldername)
            # Check if the current item is a directory
            if os.path.isdir(folder_path):
                # Count the number of files in the directory
                num_files = len([filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))])
                print(f"Folder '{foldername}' contains {num_files} file(s).")

        # List to store found files
        found_files = []

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    # save path to list
                    found_files.append(file_path)
        
        # check if shuffle
        shuffled_list = found_files
        if shuffle:
            shuffled_list = random.sample(found_files, len(found_files))
        return shuffled_list
    

    def preprocessing(self, doc):
        page_content, metadata = doc.page_content, doc.metadata

        #some processing
        page_content_processed = re.sub(r'\s+', ' ', page_content.replace("\\n", " "))

        # update page_content (text)
        doc.page_content = page_content_processed

        return doc


    def langchain_loader_splitter(self, list_pdf_paths, chunk_size, overlap):

        list_of_docs = []
        for pdf_path in tqdm(list_pdf_paths):    
            try:
                # Use load_and_split() to split the document into sentences
                loader = PyPDFLoader(pdf_path)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                # will create the chunks in "documents"
                documents = text_splitter.split_documents(data)
                # One can have lists of shape [[..],[..], ..., [..]] for number of chunks created 
                documents = [self.preprocessing(doc) for doc in documents]  
                list_of_docs.append(documents)
                    
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        # can have shape [ [..], [[..], [..]] ]. Need to unpack
        list_of_docs_unpacked = list(itertools.chain(*list_of_docs))

        return list_of_docs_unpacked
    


    def create_synthetic_dataset(self, list_of_docs_unpacked: list, test_size: int, chunk_size: int,  distributions: dict ={simple: 0.5, reasoning: 0.25, multi_context: 0.25}):
        """Uses RAGAS ..."""
        
        documents = list_of_docs_unpacked

        # cannot change chunk_overlap!

        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings,
            chunk_size=chunk_size
        )


        # Due to upper limit of the number of embeddings to create per sec/min, bacthing is used
        batch_size = 1000
        lower_b = 0
        upper_b = batch_size
        list_of_dfs = []
        while lower_b < len(documents):

             # send batch size of embeddings to database
            testset = generator.generate_with_langchain_docs(documents[lower_b:upper_b], test_size=test_size, distributions=distributions)  
            list_of_dfs.append( testset.to_pandas() )
            # increase lower and upper
            lower_b += batch_size
            upper_b += batch_size
            time.sleep(30)

        df_synthetic_testset = pd.concat(list_of_dfs, ignore_index=True)
         
        return df_synthetic_testset
    
    def create_pinecone_index(self, index_name, pinecone):  # does not work right now
        pinecone.create_index(
            name=index_name,
            dimension=1536, 
            metric='cosine', 
            spec=ServerlessSpec(
                cloud="aws",
                region="eu-west-1"
            ) 
        )


    def move_docs_to_pinecone(self, documents, embedding_model='text-embedding-3-small', index_name='tester'):
        embeddings = OpenAIEmbeddings(model=embedding_model)

        # check if pinecone index exists
        #pc = Pinecone(api_key=os.environ.get("PINECONE_KEY"))
        #print(pc)
        #if index_name not in [index['name'] for index in pc.list_indexes()]:
        #    self.create_pinecone_index(index_name, pc)
        #    print(f'Created pinecone index: {index_name}')

        pinecone = PineconeVectorStore.from_documents(
            documents, embeddings, index_name=index_name
        )   
        print(f'Moved to index: {index_name}')





if __name__ == "__main__":
    # Set hyperparameters
    CHUNKSIZE = 500 
    CHUNKOVERLAP = 100
    EMBEDDING_MODEL = 'text-embedding-3-small'
    INDEX_NAME = 'bupl-index-chunk500'

    DataClass = DataCreator()
    # Specify the directory path
    directory_path = r"c:\BUPL/pdffiles/" 

    # Search for PDF files containing a keyword
    found_files = DataClass.search_pdf_files(directory_path, shuffle=True)
    print(f'Number of pdf\'s found: {len(found_files)}')

    # load and split documents
    documents = DataClass.langchain_loader_splitter(found_files[:], CHUNKSIZE, CHUNKOVERLAP)

    # create synthetic dataset
    #df_dataset = DataClass.create_synthetic_dataset(documents,
    #                                    test_size=10,
    #                                      chunk_size=CHUNKSIZE,
    #                                        distributions={simple: 0.20,
    #                                                        reasoning: 0.40,
    #                                                          multi_context: 0.40},
    #                                        )

    # move data to pinecone
    DataClass.move_docs_to_pinecone(documents=documents,
                                     embedding_model=EMBEDDING_MODEL,
                                     index_name=INDEX_NAME)
    
    print('done')