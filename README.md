# Homeopathy Medicine Prediction Chatbot

## Project Overview
This project features a sophisticated chatbot designed to recommend homeopathy medicines based on user-reported symptoms and diseases. Leveraging advanced technologies like OpenAI, Gradio, LangChain, and Hugging Face, the chatbot provides personalized and accurate homeopathy recommendations.

## Features
- **Symptom-Based Recommendations**: Users can input symptoms and receive relevant homeopathy medicine suggestions.
- **Disease-Based Suggestions**: The chatbot can also recommend treatments based on specific diseases.
- **Decision-Making Algorithms**: Integrated algorithms enhance the decision-making process, ensuring personalized and effective recommendations.
- **User-Friendly Interface**: Utilizes Gradio for a seamless and interactive user experience.
- **AI-Powered Insights**: Implemented using OpenAI and Hugging Face models to offer intelligent responses.

## Technologies Used
- **OpenAI**: For generating natural language responses and understanding user inputs.
- **Gradio**: To create a user-friendly web interface for interactions.
- **LangChain**: For managing conversational logic and flow.
- **Hugging Face**: To enhance the chatbot's understanding and response capabilities.

## Usage
Users can interact with the chatbot by entering symptoms or diseases, and the chatbot will generate appropriate homeopathy medicine recommendations based on the input.

## Code
import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

#Set OpenAI API key
OPENAI_API_KEY = "your_openai_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Define the template
template = """You are a helpful assistant that specializes in answering questions related to homeopathy medicines.
If the question is not related to homeopathy, apologize and inform the user that you can only answer questions about homeopathy medicines.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

#Define the LLM Chain
llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

#Chatbot function
def get_text_response(user_message, history):
    if "homeopathy medicine" in user_message.lower():
        response = llm_chain.predict(user_message=user_message)
    else:
        response = "I can only provide information related to homeopathy medicines. Please provide a symptom or disease, and I will suggest a homeopathy medicine."
    return response

#Launch Gradio app
demo = gr.ChatInterface(
    get_text_response, examples=[
        "Homeopathy Medicine for Cold",
        "Homeopathy Medicine for Chest Pain",
        "Homeopathy Medicine for Bloating"
    ]
)

if __name__ == "__main__":
    demo.launch()  # To create a public link, set share=True in launch(). To enable errors and logs, set debug=True in launch().
