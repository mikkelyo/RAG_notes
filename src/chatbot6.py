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
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough




### SETUP USING THIS
# https://realpython.com/build-llm-rag-chatbot-with-langchain/
### USES FEW-SHOTS



class IndexQuerySol6:
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
        self.reviews_retriever = self.make_retriever()

        
        self.review_system_prompt = self.review_system_prompt()
        self.review_human_prompt = self.review_human_prompt()
        self.messages = [self.review_system_prompt, self.review_human_prompt]
        self.review_prompt_template = self.review_prompt_template()

        self.review_chain = self.make_chain()



    def review_system_prompt(self):

        review_template_str = """Your job is to use documents from the pedagogues' union to answer questions concerning the union.
            Instructions: 
            - Questions should always be replied in Danish.

            - Be as detailed as possible, but don't make up any information
            that's not from the context. If you don't know an answer, say
            you don't know. 

            Use the following context to answer questions:
        
            {context}

            Here are examples of well-formatted answers: 
            # example_question: 
            >> Kan jeg afspadsere optjent norm?
            # example_answer: 
            >> Afspadsering skal given i den efterfølgende normperiode. Hvis afspadsering ikke kan given i den efterføgelnde normperiode ydes i stedet overarbejdsbetaling.
                            Det betyder, at du normalt ikke kan spare afspadsering op over flere normperioder. Dog kan der være lokale aftaler eller særlige omstændigheder, der gør det muligt.
                            Det er derfor en god idé at tjekke med din arbejdsgiver eller fagforening for at få præcise oplysninger om din specifikke situation. 
                             
                            Svaret er genereret af kunstig intelligens. Husk altid at tjekke om svaret er korrekt.
            
            """

        return SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                    input_variables=["context"],
                    template=review_template_str,
                    )
                )



    def review_human_prompt(self):

        return HumanMessagePromptTemplate(
            prompt=PromptTemplate(
            input_variables=["question"],
            template="The question to answer is: {question}",
                    )
                )


    def review_prompt_template(self):
    
        return ChatPromptTemplate(input_variables=["context", "question"],
                                        messages=self.messages, )
    
    def make_retriever(self):
        
        compressor = CrossEncoderReranker(model=self.model_rerank, top_n=self.parameter_dict['rerank_k'])
        compression_retriever = ContextualCompressionRetriever(
                                                                    base_compressor=compressor,
                                                                     base_retriever=self.vectorstore.as_retriever(
                                                                                        k=self.parameter_dict['top_k']
                                                                            )
                                                                    )

        retriever = compression_retriever #self.vectorstore.as_retriever(k=self.parameter_dict['top_k'])
        return retriever
    
    
    def make_chain(self):
        
        chat_model = ChatOpenAI(model=self.model_chat, temperature=self.parameter_dict['temperature'])
        output_parser = StrOutputParser()

        review_chain = (
                        {"context": self.reviews_retriever, "question": RunnablePassthrough()}
                        | self.review_prompt_template 
                        | chat_model 
                        | output_parser)


        return review_chain
    


    def get_relevant_documents(self, query: str) -> list[str]:
        """Retrieves relevant documents for the given query.

        Args:
            query (str): The query to retrieve relevant documents for.

        Returns:
            list[str]: A list of strings containing the retrieved documents.
        """
        # Define vectorstore

        retriever = self.reviews_retriever

        # Retrieve relevant documents
        results = retriever.get_relevant_documents(query)
        documents = [doc.page_content for doc in results ]  

        return documents


    def complete(self, input_query, limit_txt_prompt=10000):
        """Generates a response to the input query using the LLM."""

        answer = self.review_chain.invoke(input_query) + '\n\n' + 'Svaret er genereret af kunstig intelligens og kan derfor være forkert.'

        return answer


if __name__ == "__main__":
    
    # initiate class
        # Get keys and index name and model name (same as the one to do embeddings)
    load_dotenv('.env')
    PINECONE_KEY = os.environ.get("PINECONE_KEY")
    INDEX_NAME =  'bupl-index' #"index-cas-onboarding"
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    MODEL_EMBEDDING = 'text-embedding-3-small'
    MODEL_RERANK = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base") # omre about model here : https://huggingface.co/BAAI/bge-reranker-v2-m3
    MODEL_CHAT = 'gpt-3.5-turbo'

    # Hyperparameters
    TOP_K = 3
    THRESHOLD = 0 # minimum score of retrieved documents
    LIMIT_TXT_PROMPT = 10000
    TEMPERATURE = 0.2
    RERANK_K = 5

    parameter_dict = {'temperature': TEMPERATURE,
                      'threshold': THRESHOLD,
                      'top_k': TOP_K,
                      'rerank_k': RERANK_K}

        # Create class
    result = IndexQuerySol6(index_name=INDEX_NAME,
                            api_key_pinecone=PINECONE_KEY, 
                            model_embedding_name=MODEL_EMBEDDING,
                            model_rerank= MODEL_RERANK, 
                            model_chat_name=MODEL_CHAT,
                            api_key_openai=OPENAI_KEY,
                            parameter_dict=parameter_dict)
    
    answer = result.complete(input_query='Hvad er reglerne for omlagt tjeneste? og hvor kan jeg finde det?')
    print(answer)

    #contexts = result.get_relevant_documents("""Hvad er reglerne for omlagt tjeneste? og hvor kan jeg finde det?""")
    #print(contexts)