# pip install python-dotenv
# pip install langchain-google-genai
# pip install streamlit

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import os

class ConversationalChatbot:
    def __init__(self):
        self.api_key = 'AIzaSyCC2a8jtdNe2PQM0jyEwhB2_omTfD0IhpE'
        os.environ["GEMINI_API_KEY"] = self.api_key
        load_dotenv()
        self.llm = None
        self.initialize_streamlit()

    def initialize_streamlit(self):
        st.set_page_config(page_title="Conversational Bot!")
        st.title("Conversational Chatbot 💬")
        if "messages" not in st.session_state:
            st.session_state.messages = [
                AIMessage(content="Hello, I am a bot. How can I help you?")
            ]
        self.display_messages()
        self.handle_user_input()

    def get_llm_instance(self):
        """
        Return the instance of the LLM model globally.
        """
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model="gemini-pro",
                stream=True,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
            )
        return self.llm

    def get_response(self, user_query, conversation_history):
        """
        Return the response using the streaming chain.
        """
        prompt_template = f"""
        You are an AI assistant. Answer the following question considering the history of the conversation:
        Chat history: {conversation_history}
        User question: {user_query}
        """
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        llm = self.get_llm_instance()
        expression_language_chain = prompt | llm | StrOutputParser()

        return expression_language_chain.stream(
            {
                "conversation_history": conversation_history,
                "user_query": user_query
            }
        )

    def display_messages(self):
        """
        Display the conversation history in Streamlit chat messages.
        """
        for message in st.session_state.messages:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)

    def handle_user_input(self):
        """
        Handle user input from the chat interface.
        """
        prompt = st.chat_input("Say Something")
        if prompt:
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(f"{prompt}")

            with st.chat_message("assistant"):
                response = st.write_stream(self.get_response(prompt, st.session_state.messages))
            st.session_state.messages.append(AIMessage(content=response))

if __name__ == "__main__":
    ConversationalChatbot()
