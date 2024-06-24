import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from typing import List, Dict, Any, Optional

#%%

# Uses a manual prompt setup

class IndexQuerySol1:
    def __init__(self, index_name: str, api_key_pinecone: str, model_embedding_name: str, api_key_openai: str, model_chat_name: str, parameter_dict: dict,
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
        self.pc = Pinecone(api_key=api_key_pinecone)
        self.index = self.pc.Index(self.index_name)

        self.client_openai = OpenAI(api_key=api_key_openai)
        self.model_embedding = model_embedding_name
        self.model_chat = model_chat_name

        self.memory_id = 'SOME_ID'
        self.memory = {}
        self.parameter_dict = parameter_dict


    def get_index(self):
        """
        Retrieves the Pinecone index.

        Returns:
            Index: The Pinecone index object.
        """
        return self.index
    

    def remember_query(self, user_id, query):
        """
        Remember a query for a specific user.

        Args:
            user_id (str): Identifier for the user.
            query (str): The query to be remembered.
        """
        if user_id in self.memory:
            self.memory[self.memory_id].append(query)
        else:
            self.memory[self.memory_id] = [query]


    def get_user_queries(self, user_id):
        """
        Retrieve all queries remembered for a specific user.

        Args:
            user_id (str): Identifier for the user.

        Returns:
            list: A list of queries remembered for the user.
        """
        return self.memory.get(user_id, [])
    

    def embed_input(self, input_text: str):
        """
        Embeds a text input into an embedding vector using OpenAI's text embedding model.

        Args:
            input_text (str): The input text to be embedded.

        Returns:
            list: The embedding vector representing the input text.
        """
        res = self.client_openai.embeddings.create(input=[input_text], model=self.model_embedding)
        return res.data[0].embedding
    

    def query_index(self, input_vector=[0.3] * 1536, top_k=3):
        """
        Queries the Pinecone index with the specified input vector.

        Args:
            input_vector (list): The input vector to query the index with.
            top_k (int): The number of nearest neighbors to retrieve.

        Returns:
            list: A list of dictionaries containing the top-k nearest neighbors along with their distances.
        """
        result = self.index.query(
                    vector=input_vector,
                    top_k=top_k,
                    include_values=False,  # HERE
                    include_metadata=True
                    )
        return result
    
    def get_relevant_documents(self, input_query):
        """Used for evaluation of the model"""
        # set parameter values
        top_k = self.parameter_dict['top_k']
        threshold = self.parameter_dict['threshold']

        input_vector = self.embed_input(input_query)
        retrived_vecs = self.query_index(input_vector, top_k)

        contexts = []
        contexts = contexts + [
            x['metadata']['text'] + f" (Source: {x['metadata']['source']}, page {x['metadata']['page']})" for x in retrived_vecs['matches'] 
            if x['score'] > threshold
        ]

        return contexts
    

    def make_prompt(self, input_query, limit_txt_prompt=10000):
        """
        Constructs a prompt for answering a question based on retrieved context.
        - If no contexts meet the threshold, a default message indicating no contexts were retrieved is used.
        - The prompt is built iteratively, adding contexts until the character limit is reached.

        Args:
            input_query (str): The query for which an answer is sought.
            top_k (int, optional): The number of top contexts to retrieve based on relevance. Defaults to 3.
            threshold (float, optional): The minimum score for a context to be included. Defaults to 0.
            limit (int, optional): The character limit for the entire prompt. Defaults to 10000.

        Returns:
            str: A formatted prompt including relevant contexts and the input query.
        """
        # set parameter values
        contexts = self.get_relevant_documents(input_query=input_query)

        # Check if any context
        if contexts == []:
            contexts = ["No contexts retrieved. Try to answer the question yourself!"]

        # get former queries
        former_queries = self.get_user_queries(self.memory_id)
        former_queries_string = "\n\n---\n\n".join(former_queries) + "\n\n" if former_queries else "No former queries." # right now, only questions

        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the main question based on the context and former queries of the user below.\n\n"+
            "Context:\n"
        )
        prompt_memory = (
            "\n\nFormer queries made by user are:\n" + former_queries_string
        )
        prompt_end = (
            f"\n\nMain question: {input_query}\nAnswer:"
        )
        
        # Create the prompt string
        for i in range(0, len(contexts)):
            # iteratively check if len is too big (can add more)
            # or if we are done
            if len("\n\n---\n\n".join(contexts[:i])) >= limit_txt_prompt:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_memory +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start + 
                    "\n\n---\n\n".join(contexts) +
                    prompt_memory +
                    prompt_end
                )
        
        return prompt
    

    def complete(self, input_query, limit_txt_prompt=10000):
        """
        Generates a completion for the given prompt using the OpenAI API.

        Args:
            prompt (str): The input prompt for which the completion is to be generated.

        Returns:
            str: The string part of the response from the OpenAI API containing the generated completion.
        """
        #https://platform.openai.com/docs/quickstart

        # set parameter
        temperature = self.parameter_dict['temperature']

        # get RAG-prompt
        final_prompt = self.make_prompt(input_query, limit_txt_prompt)

        # fun the LLM model
        completion = self.client_openai.chat.completions.create(
            model=self.model_chat, 
            temperature = temperature, 
            messages=[
                {"role": "system", "content": "The answer provided should always summarize the question asked."},
                {"role": "system", "content": "Questions should always be replied in Danish."},
                {"role": "user", "content": final_prompt} 
            ]
        ) # TODO: ADD INSTRUCTIONS (EXAMPLES: COULD BE ABOUT FORMAT (rephrase question in answer), ALWAYS REMIND THAT SOME NUMBERS MIGHT BE CONFEDINCIAL)
            # {"role": "system", "content": 'Please rephrase the question in you answer.'},
        answer = completion.choices[0].message.content.strip()

        self.remember_query(self.memory_id, 'Query: ' + input_query + '\nReply: ' + answer)

        return answer


    ### DOES NOT USE----------------------------------------------------------
#    def use_RAG(self, temperature, max_tokens):
#        #https://smith.langchain.com/hub/rlm/rag-prompt
#        #and
#        #https://medium.com/@ReneMazuela/building-a-query-engine-with-pinecone-and-langchain-a-comprehensive-guide-b095ac226838
#        llm=ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", max_tokens=max_tokens)
#        qa_chain  = RetrievalQA.from_chain_type(
#        llm=llm,
#        chain_type="stuff",
#        retriever=self.pc.as_retriever()
#        )
#        question = "What are the approaches to Task Decomposition?"
#        result = qa_chain({"query": question})
#        result["result"]
    ### DOES NOT USE----------------------------------------------------------


if __name__ == "__main__":
    
    # initiate class
        # Get keys and index name and model name (same as the one to do embeddings)
    load_dotenv('.env')
    PINECONE_KEY = os.environ.get("PINECONE_KEY")
    INDEX_NAME =  'bupl-index' #"index-cas-onboarding"
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    MODEL_EMBEDDING = 'text-embedding-3-small'
    MODEL_CHAT = 'gpt-3.5-turbo'

    # Hyperparameters
    TOP_K = 3
    THRESHOLD = 0 # minimum score of retrieved documents
    LIMIT_TXT_PROMPT = 10000
    TEMPERATURE = 0.2

    parameter_dict = {'temperature': TEMPERATURE,
                      'threshold': THRESHOLD,
                      'top_k': TOP_K}

        # Create class
    result = IndexQuerySol1(index_name=INDEX_NAME,
                         api_key_pinecone=PINECONE_KEY,
                           model_embedding_name=MODEL_EMBEDDING,
                            model_chat_name=MODEL_CHAT,
                             api_key_openai=OPENAI_KEY,
                             parameter_dict=parameter_dict)
    
    prompt = """er det bare fedt?"""
    
#%%

print("Prompt:",prompt)
print("Answer:",result.complete(input_query = prompt))