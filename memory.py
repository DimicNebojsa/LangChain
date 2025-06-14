import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"] or getpass("Enter your OpenAI API key: ")

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")



### ConversationBufferMemory

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    {"input": "Hi, my name is Josh"},  # user message
    {"output": "Hey Josh, what's up? I'm an AI model called Zeta."}  # AI response
)
memory.save_context(
    {"input": "I'm researching the different types of conversational memory."},  # user message
    {"output": "That's interesting, what are some examples?"}  # AI response
)
memory.save_context(
    {"input": "I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory."},  # user message
    {"output": "That's interesting, what's the difference?"}  # AI response
)
memory.save_context(
    {"input": "Buffer memory just stores the entire conversation, right?"},  # user message
    {"output": "That makes sense, what about ConversationBufferWindowMemory?"}  # AI response
)
memory.save_context(
    {"input": "Buffer window memory stores the last k messages, dropping the rest."},  # user message
    {"output": "Very cool!"}  # AI response
)


print(memory.load_memory_variables({}))
