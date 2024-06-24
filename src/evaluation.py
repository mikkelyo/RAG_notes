# This script aims to use the following framework for evaluating a RAG/model
# https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a

# Further sources
### RAGAS and dspy:  https://medium.com/@shramanpadhalni/enhancing-retriever-augmented-generation-with-programming-prompts-through-dspy-and-evaluate-model-b106c3aa4021

import pandas as pd
import sys
import os
from tqdm import tqdm 
import pickle
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from chatbot1 import IndexQuerySol1
from chatbot2 import IndexQuerySol2
from chatbot3 import IndexQuerySol3
from chatbot4 import IndexQuerySol4

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
# Import Weights and Biases
import wandb


class EvaluateRAG:
    def __init__(self, rag_model, questions_answers_path: str):
        """The module uses the following metrics from the RAGAS framework: https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html"""

        self.rag_model = rag_model
        # The above model should have the two modules
        # 1. .get_relevant_documents(query)
        # 2. .complete(query)

        self.questions_answers_path = questions_answers_path

        self.questions, self.ground_truths = self.load_questions_and_answers()
        self.dataset = self.prepare_eval_dict()

    
    def load_questions_and_answers(self):
        """Loads the questions made for testing a RAG-system. Expects a format of:
        id,answer_id,question,difficulty
        1,1,"Hvad er reglerne for omlagt tjeneste?",1
        2,1,"Hvornår får jeg omlagt tjeneste?",1
        3,1,"Hvordan regner jeg omlagt tjeneste ud?",1
        4,1,"Omlagt tjeneste?",2
        """
        df = pd.read_csv(self.questions_answers_path)

        # get questions
        questions = list(df['question'])
        ground_truths = None

        if 'ground_truth' in df.columns:
            ground_truths = list(df['ground_truth'])

        return questions, ground_truths
    
    
    def prepare_eval_dict(self):
        """Formats the dictionary to run the framework on."""
        questions = self.questions
        ground_truths = self.ground_truths
        answers = []
        contexts = []
        # get model answers
        for query in tqdm(questions):
            answers.append(self.rag_model.complete(query))
            contexts.append([contexts for contexts in self.rag_model.get_relevant_documents(query)])

        assert len(questions) == len(answers), f"The length of questions seem to be {len(questions)} and the length of answers seem to be {len(answers)}"
        assert len(answers) == len(contexts), f"The length of answers seem to be {len(answers)} and the length of contexts seem to be {len(contexts)}"

        if self.ground_truths == None:
            print('No ground truth are provided.')
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                }  
        
        else: 
            print('Ground truth are provided!')
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
                }  

        return Dataset.from_dict(data)
    

    def evaluate_model(self):
        """Uses the RAGAS framework for evaluating the model"""

        if self.ground_truths == None:
            result = evaluate(
            dataset = self.dataset, 
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            )
        else: 
            result = evaluate(
            dataset = self.dataset, 
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            )

        df = result.to_pandas()

        return df
    
    
if __name__ == "__main__":


    # Initialize Weights and Biases
    wNb_project_name = os.environ.get("wNb_project_name")
    run = wandb.init(project=wNb_project_name)#, entity="your_entity_name")

    # Load environment variables
    load_dotenv('.env')
    PINECONE_KEY = os.environ.get("PINECONE_KEY")
    INDEX_NAME = 'bupl-index'
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    MODEL_EMBEDDING = 'text-embedding-3-small'
    MODEL_RERANK = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    MODEL_CHAT = 'gpt-3.5-turbo'

    # Hyperparameters
    TOP_K = 30
    THRESHOLD = 0.2
    LIMIT_TXT_PROMPT = 10000
    TEMPERATURE = 0
    RERANK_K = 3

    parameter_dict = {'temperature': TEMPERATURE,
                      'threshold': THRESHOLD,
                      'top_k': TOP_K,
                      'rerank_k': RERANK_K}

    # Log hyperparameters to W&B
    wandb.config.update(parameter_dict)

    # Create class
    RAG_model = IndexQuerySol4(index_name=INDEX_NAME,
                               api_key_pinecone=PINECONE_KEY,
                               model_embedding_name=MODEL_EMBEDDING,
                               model_rerank=MODEL_RERANK,
                               model_chat_name=MODEL_CHAT,
                               api_key_openai=OPENAI_KEY,
                               parameter_dict=parameter_dict)

    path_questions_answers = 'data/BUPL/questions/questions.csv'
    #path_questions_answers = 'data/BUPL/synthetic_testset/synthetic_data.csv'


    evalModule = EvaluateRAG(RAG_model, path_questions_answers)
    df_output_result = evalModule.evaluate_model()

    # Log table to W&B
    wandb_table = wandb.Table(dataframe=df_output_result)
    run.log({"eval_table": wandb_table})
    
    # Log mean metrics
    mean_context_precision = 0
    mean_context_recall = 0
    if 'context_precision' in df_output_result.columns:
        mean_context_precision = df_output_result['context_precision'].mean()
        mean_context_recall = df_output_result['context_recall'].mean()
    mean_faithfulness = df_output_result['faithfulness'].mean()
    mean_answer_relevancy = df_output_result['answer_relevancy'].mean()

    wandb.log(
        {'mean_context_precision': mean_context_precision,
        'mean_context_recall': mean_context_recall, 
        'mean_faithfulness': mean_faithfulness,
        'mean_answer_relevancy': mean_answer_relevancy}
    )

    # Finish the W&B run
    wandb.finish()
