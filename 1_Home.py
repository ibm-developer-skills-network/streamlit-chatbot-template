

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
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

Watsonx_API = "uvnQIfnjPk2Jpszy0hAvr80xCUAudclZsltCi3gYxAVu"
Project_id= "177ab670-c7d0-4f34-894f-228297d644d9"

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

    
    LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat',
        credentials=credentials,
        params=params,
        project_id=Project_id)

    llm_hub = WatsonxLLM(model=LLAMA2_model)

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

init_llm()

# Store the conversation history in a List
conversation_history = []

# load the file
documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()

def ask_bot(input_text):

    global documents

    # LLMPredictor: to generate the text response (Completion)
    llm_predictor = LLMPredictor(
            llm=llm_hub
    )
                                     
    # Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
    embed_model = LangchainEmbedding(embeddings)
    
    # ServiceContext: to encapsulate the resources used to create indexes and run queries    
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            embed_model=embed_model
    )      
    # build index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    PROMPT_QUESTION = """
    Your name is IBM Skills Network. Briefly introduce yourself first if it's the first time answering a question.
    Your conversation with the human is recorded in the chat history below. After the self-introduction, you don't need to repeat mentioning your name or introducing yourself actively. If the recruiter asks about the skills or experiences you have with url links, answer it with the link.
    
    History:
    "{history}"
    
    Now continue the conversation with the human. If you do not know the answer, politely ask for more information.
    Human: {input}
    Assistant:"""
    
    # This will wrap the default prompts that are internal to llama-index
    #query_wrapper_prompt = SimpleInputPrompt(f"{PROMPT_QUESTION}<|USER|>{input_text}<|ASSISTANT|>")
    
    # update conversation history
    global conversation_history
    history_string = "\n".join(conversation_history)
    print(f"history_string: {history_string}")  
    
    # query LlamaIndex and llama-2-70b-chat for the AI's response
    output = index.as_query_engine().query(input_text)
    print(f"output: {output}")   
    
    # update conversation history with user input and AI's response
    conversation_history.append(input_text)
    conversation_history.append(output.response)
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
