# https://scalexi.medium.com/implementing-a-retrieval-augmented-generation-rag-system-with-openais-api-using-langchain-ab39b60b4d9f


### Brug anden vectorstore
from langchain_pinecone import PineconeVectorStore
# # embeddings = OpenAIEmbeddings()
# vectorstore = PineconeVectorStore(index_name=index_name)
#)
from typing import List, Dict, Any, Optional

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
#from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# Uses Langchain with rerank of HuggingFace


class IndexQuerySol3:
    def __init__(self,
                 index_name: str,
                 api_key_pinecone: str, 
                 model_embedding_name: str,
                 api_key_openai: str,
                 model_chat_name: str, 
                 parameter_dict: dict, 
                 model_rerank: Optional[str] = None):
        """
        Initializes the IndexQuery object with the specified parameters.

        Args:
            index_name (str): The name of the Pinecone index to interact with.
            api_key_pinecone (str): The API key for accessing Pinecone services.
            model_embedding_name (str): The name of the OpenAI model used for text embedding.
            api_key_openai (str): The API key for accessing OpenAI services.
            model_chat_name (str): A string working as a identifier for the memory session.
            paramter_dict (dict): A dictionary of temperature=0.2, top_k=3, threshold=0. 
                - temperature: controls the freedom of the llm responses.
                - top_k: controls the number of relevant text embedding to retrieve.
                - threshold: sets the threshold for the cosine similarity of the retrieved text embeddings to retrieve. 

        """
        self.index_name = index_name
        self.api_key_openai = api_key_openai
        os.environ["OPENAI_API_KEY"] = api_key_openai
        self.model_embedding = model_embedding_name
        self.model_rerank = model_rerank
        self.model_chat = model_chat_name

        self.parameter_dict = parameter_dict
        self.embeddings = OpenAIEmbeddings(model=self.model_embedding)
        self.vectorstore = PineconeVectorStore(index_name=self.index_name,
                                             embedding=self.embeddings)
        self.compressor = CrossEncoderReranker(model=self.model_rerank, top_n=self.parameter_dict['rerank_k'])
        self.compression_retriever = ContextualCompressionRetriever(
                                                                    base_compressor=self.compressor,
                                                                     base_retriever=self.vectorstore.as_retriever(
                                                                                        k=self.parameter_dict['top_k']
                                                                            )
                                                                    )

    
    def get_relevant_documents(self, query: str) -> list[str]:
        """Retrieves relevant documents for the given query.

        Args:
            query (str): The query to retrieve relevant documents for.

        Returns:
            list[str]: A list of strings containing the retrieved documents.
        """
        # Define vectorstore

        retriever = self.compression_retriever

        # Retrieve relevant documents
        results = retriever.get_relevant_documents(query)
        documents = [doc.page_content for doc in results ]  

        return documents


    def complete(self, input_query, limit_txt_prompt=10000):
        """Generates a response to the input query using the LLM."""
        temperature = self.parameter_dict['temperature']
        llm = ChatOpenAI(temperature=temperature,
                          model_name=self.model_chat)

        # Define vectorstore

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.compression_retriever
        )

        # Run model
        result = chain(input_query)
        answer = result["result"]

        return answer


if __name__ == "__main__":
    
    # initiate class
        # Get keys and index name and model name (same as the one to do embeddings)
    load_dotenv('.env')
    PINECONE_KEY = os.environ.get("PINECONE_KEY")
    INDEX_NAME =  'bupl-index' #"index-cas-onboarding"
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    MODEL_EMBEDDING = 'text-embedding-3-small'
    MODEL_RERANK = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    MODEL_CHAT = 'gpt-3.5-turbo'

    # Hyperparameters
    TOP_K = 10
    THRESHOLD = 0 # minimum score of retrieved documents
    LIMIT_TXT_PROMPT = 10000
    TEMPERATURE = 0
    RERANK_K = 5


    parameter_dict = {'temperature': TEMPERATURE,
                      'threshold': THRESHOLD,
                      'top_k': TOP_K,
                      'rerank_k': RERANK_K}

        # Create class
    result = IndexQuerySol3(index_name=INDEX_NAME,
                            api_key_pinecone=PINECONE_KEY, 
                            model_embedding_name=MODEL_EMBEDDING,
                            model_rerank= MODEL_RERANK, 
                            model_chat_name=MODEL_CHAT,
                            api_key_openai=OPENAI_KEY,
                            parameter_dict=parameter_dict)
    
    answer = result.complete(input_query='Hello')
    print(answer)

    contexts = result.get_relevant_documents("""Hvad er reglerne for omlagt tjeneste? og hvor kan jeg finde det?""")
    print(contexts)