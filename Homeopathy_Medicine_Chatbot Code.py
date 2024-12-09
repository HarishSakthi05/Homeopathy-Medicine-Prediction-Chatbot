
import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set OpenAI API key
OPENAI_API_KEY = "your_openai_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the template
template = """You are a helpful assistant that specializes in answering questions related to homeopathy medicines.
If the question is not related to homeopathy, apologize and inform the user that you can only answer questions about homeopathy medicines.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

# Define the LLM Chain
llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

# Chatbot function
def get_text_response(user_message, history):
    if "homeopathy medicine" in user_message.lower():
        response = llm_chain.predict(user_message=user_message)
    else:
        response = "I can only provide information related to homeopathy medicines. Please provide a symptom or disease, and I will suggest a homeopathy medicine."
    return response

# Launch Gradio app
demo = gr.ChatInterface(
    get_text_response, examples=[
        "Homeopathy Medicine for Cold",
        "Homeopathy Medicine for Chest Pain",
        "Homeopathy Medicine for Bloating"
    ]
)

if __name__ == "__main__":
    demo.launch()  # To create a public link, set share=True in launch(). To enable errors and logs, set debug=True in launch().
