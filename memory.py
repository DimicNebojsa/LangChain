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



memory = ConversationBufferMemory(return_messages=True)

memory.chat_memory.add_user_message("Hi, my name is Josh")
memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
memory.chat_memory.add_ai_message("Very cool!")

memory.load_memory_variables({})


from langchain.chains import ConversationChain

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)


#response = chain.invoke({"input": "what is my name again?"})
#print(response)


### Solution wuth Runnable Lambda

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)

system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

pipeline = prompt_template | llm

from langchain_core.chat_history import InMemoryChatMessageHistory

chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

from langchain_core.runnables.history import RunnableWithMessageHistory

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

response2  = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"session_id": "id_123"}
)

#response3  = pipeline_with_history.invoke(
#    {"query": "Hi, Can you tell me what is my name again?"},
#    config={"session_id": "id_123"}
#)

#print(response3)
#print(chat_map['id_123'])


#### ConversationBufferWindowMemory


from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True)


memory.chat_memory.add_user_message("Hi, my name is Josh")
memory.chat_memory.add_ai_message("Hey Josh, what's up? I'm an AI model called Zeta.")
memory.chat_memory.add_user_message("I'm researching the different types of conversational memory.")
memory.chat_memory.add_ai_message("That's interesting, what are some examples?")
memory.chat_memory.add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
memory.chat_memory.add_ai_message("That's interesting, what's the difference?")
memory.chat_memory.add_user_message("Buffer memory just stores the entire conversation, right?")
memory.chat_memory.add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
memory.chat_memory.add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
memory.chat_memory.add_ai_message("Very cool!")

chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

#result4 = chain.invoke({"input": "what is my name again?"})
#print(result4['response'])


### ConversationBufferWindowMemory with RunnableWithMessageHistory

from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    # remove anything beyond the last
    return chat_map[session_id]


from langchain_core.runnables import ConfigurableFieldSpec

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

result5 = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)

print(result5)

result6 = pipeline_with_history.invoke(
    {"query": "Well i am trying to learn about buffer memory in langchain"},
    config={"configurable": {"session_id": "id_k4", "k": 4}}
)

print(result6)

chat_map["id_k4"].clear()  # clear the history

# manually insert history
chat_map["id_k4"].add_user_message("Hi, my name is Josh")
chat_map["id_k4"].add_ai_message("I'm an AI model called Zeta.")
chat_map["id_k4"].add_user_message("I'm researching the different types of conversational memory.")
chat_map["id_k4"].add_ai_message("That's interesting, what are some examples?")
chat_map["id_k4"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
chat_map["id_k4"].add_ai_message("That's interesting, what's the difference?")
chat_map["id_k4"].add_user_message("Buffer memory just stores the entire conversation, right?")
chat_map["id_k4"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
chat_map["id_k4"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
chat_map["id_k4"].add_ai_message("Very cool!")

# if we now view the messages, we'll see that ONLY the last 4 messages are stored
print(chat_map["id_k4"].messages)


