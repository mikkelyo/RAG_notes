import pytest
import os
import sys
from dotenv import load_dotenv

#sys.path.append("../")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chatbot1 import IndexQuerySol1



# initialize class for testing
load_dotenv('.env')
PINECONE_KEY = os.environ.get("PINECONE_KEY")
INDEX_NAME =  'index-cas-onboarding' #"index-cas-onboarding"
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


def test_query_index():
    # test 1
    string_test1 = 'Havde Jeudans nogle anlægslån i udgangen af 2022?'
    input_vector = result.embed_input(string_test1)
    retrieved = result.query_index(input_vector, top_k=3)
    assert '11563' in [x['id'] for x in retrieved['matches'] ]
    # test 2
    string_test2 = 'Hvor stor var andelen af euro obligationer i 2022 hos DLR Kredit?'
    input_vector = result.embed_input(string_test2)
    retrieved = result.query_index(input_vector, top_k=3)
    assert '1524' in [x['id'] for x in retrieved['matches'] ]
    # test 3
    string_test3 = 'What were the points on the GM April 2023 in BI Boligejendomme?'
    input_vector = result.embed_input(string_test3)
    retrieved = result.query_index(input_vector, top_k=3)
    assert '12103' in [x['id'] for x in retrieved['matches'] ]


if __name__ == "__main__":
    pytest.main()