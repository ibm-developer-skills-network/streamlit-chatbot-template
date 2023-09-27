

# Import necessary libraries
import torch
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, LangchainEmbedding
# Llamaindex also works with langchain framework to implement embeddings 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.prompts.prompts import SimpleInputPrompt
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
llm_hub = None
embeddings = None

Watsonx_API = "Watsonx_API"
Project_id= "Project_id"

# Function to initialize the Watsonx language model and its embeddings used to represent text data in a form (vectors) that machines can understand. 
def init_llm():
    global llm_hub, embeddings
    
    params = {
        GenParams.MAX_NEW_TOKENS: 250, # The maximum number of tokens that the model can generate in a single run.
        GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
        GenParams.TEMPERATURE: 0.8,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
        GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
    }
    
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : Watsonx_API
    }

    
    model = Model(
        model_id= 'meta-llama/llama-2-70b-chat',
        credentials=credentials,
        params=params,
        project_id=Project_id)

    llm_hub = WatsonxLLM(model=model)

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

init_llm()

# load the file
documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()

# LLMPredictor: to generate the text response (Completion)
llm_predictor = LLMPredictor(
        llm=llm_hub
)
                                 
# Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
embed_model = LangchainEmbedding(embeddings)
#embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

# ServiceContext: to encapsulate the resources used to create indexes and run queries    
service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, 
        embed_model=embed_model
)      
# build index
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# build a query engine
def ask_bot(input_text):

    global index

    PROMPT_QUESTION = """You are an AI agent helping answer questions about Buddy to recruiters. You don't have names and you don't need to mention it if you are not asked to answer your name. Introduce yourself when you are introducing who you are.
    If you do not know the answer, politely admit it and let users know how to contact Buddy to get more information. 
    Human: {input}
    """
    
    # query LlamaIndex and LLAMA_2_70B_CHAT for the AI's response
    output = index.as_query_engine().query(PROMPT_QUESTION.format(input=input_text))
    print(f"output: {output}")
    
    return output

# get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You can send your questions and hit Enter to know more about me:)", key="input")
    return input_text

#st.markdown("Chat With Me Now")
user_input = get_text()

if user_input:
  #text = st.text_area('Enter your questions')
    st.info(ask_bot(user_input))
