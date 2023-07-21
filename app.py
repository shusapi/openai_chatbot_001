
# 以下を「app.py」に書き込み
import streamlit as st
from streamlit_chat import message

import openai
import os

os.environ["OPENAI_API_KEY"] = st.secrets.OpenAIAPI.openai_api_key
# openai.api_key = st.secrets.OpenAIAPI.openai_api_key

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Does it work?
from langchain.callbacks.streamlit import StreamlitCallbackHandler

system_message = "あなたは優秀なChatbotです。"

prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template(system_message),
  MessagesPlaceholder(variable_name="history"),
  HumanMessagePromptTemplate.from_template("{input}")
])

@st.cache_resource
def load_conversation():
  llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    streaming=True,
    callback_manager=CallbackManager([
      # StreamlitCallbackHandler(),
      StreamingStdOutCallbackHandler()
    ]),
    verbose=True,
    temperature=0
  )
  # memory = ConversationBufferMemory(return_messages=True)
  memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024, return_messages=True)
  conversation = ConversationChain(
    memory=memory,
    prompt=prompt,
    llm=llm
  )
  return conversation

st.title("ChatBot")

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

with st.form("ChatBotに質問する", clear_on_submit=True):
  user_message = st.text_area("文章を入力してください")

  submitted = st.form_submit_button("質問する")
  if submitted:
    conversation = load_conversation()
    answer = conversation.predict(input=user_message)

    st.session_state.past.append(user_message)
    st.session_state.generated.append(answer)

    if st.session_state["generated"]:
      for i in range(len(st.session_state.generated) - 1, -1, -1):
        message(st.session_state.generated[i], key=str(i))
        message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
