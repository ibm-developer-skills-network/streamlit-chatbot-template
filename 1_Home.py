

# Import necessary libraries
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, LangchainEmbedding
# Llamaindex also works with langchain framework to implement embeddings to configure the Falcon-7B-Instruct model from Hugging Face 
from langchain.llms import HuggingFaceEndpoint
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFaceHub

# Store the conversation history in a List
conversation_history = []

def ask_bot(input_text):
    # load the file
    documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()
    # prepare Falcon Huggingface API
    llm = HuggingFaceEndpoint(
                endpoint_url= "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct" ,
                huggingfacehub_api_token="HuggingFace_API_KEY", # Replace with your own API key or use ours: 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
                task="text-generation",
                model_kwargs = {
                    "max_new_tokens":1024 # define the maximum number of tokens the model may produce in its answer         
                }
            )
    # LLMPredictor: to generate the text response (Completion)
    llm_predictor = LLMPredictor(llm=llm)
    # Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    # ServiceContext: to encapsulate the resources used to create indexes and run queries
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            embed_model=embed_model
        )      
    # build index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    PROMPT_QUESTION = """
        You are the website assistant representing IBM Skills Network, helping users to get answers regarding this website.
        
        History:
        "{history}"
        
        Now continue the conversation with the human. If you do not know the answer, politely ask for more information.
        Human: {input}
        Assistant:"""

    # update conversation history
    global conversation_history
    history_string = "\n".join(conversation_history)
    print(f"history_string: {history_string}")  
    # query LlamaIndex and Falcon-7B-Instruct for the AI's response
    output = index.as_query_engine().query(PROMPT_QUESTION.format(history=history_string, input=input_text))
    print(f"output: {output}")   
    # update conversation history with user input and AI's response
    conversation_history.append(input_text)
    conversation_history.append(output.response)
    return output.response

# get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You can send your questions and hit Enter to know more about me:)", key="input")
    return input_text

#st.markdown("Chat With Me Now")
user_input = get_text()

if user_input:
  #text = st.text_area('Enter your questions')
    st.info(ask_bot(user_input))
