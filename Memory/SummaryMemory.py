from langchain.memory import ConversationSummaryMemory
import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"] or getpass("Enter your OpenAI API key: ")

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
memory = ConversationSummaryMemory(llm=llm)

chain = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=True
)

# chain.invoke({"input": "hello there my name is Josh"})
# chain.invoke({"input": "I am researching the different types of conversational memory."})
# chain.invoke({"input": "I have been looking at ConversationBufferMemory and ConversationBufferWindowMemory."})
# chain.invoke({"input": "Buffer memory just stores the entire conversation"})
# chain.invoke({"input": "Buffer window memory stores the last k messages, dropping the rest."})
#
#
# result1 = chain.invoke({"input": "What is my name again?"})
# print(result1)


### SOLUTION WITH RUNNABLE LAMBDA

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
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

class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: ChatOpenAI = Field(default_factory=ChatOpenAI)

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm=llm)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        # construct the summary chat messages
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{messages}"
            )
        ])
        # format the messages and invoke the LLM
        new_summary = self.llm.invoke(
            summary_prompt.format_messages(existing_summary=self.messages, messages=messages)
        )
        # replace the existing history with a single system summary message
        self.messages = [SystemMessage(content=new_summary.content)]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}
def get_chat_history(session_id: str, llm: ChatOpenAI) -> ConversationSummaryMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = ConversationSummaryMessageHistory(llm=llm)
    # return the chat history
    return chat_map[session_id]


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
            id="llm",
            annotation=ChatOpenAI,
            name="LLM",
            description="The LLM to use for the conversation summary",
            default=llm,
        )
    ]
)

result2 = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"session_id": "id_123", "llm": llm}
)

print(result2)
print("------")
print(chat_map["id_123"].messages)

print("********")

result2 = pipeline_with_history.invoke(
    {"query": "I'm researching the different types of conversational memory."},
    config={"session_id": "id_123", "llm": llm}
)
print("------")
print(result2)

print(chat_map["id_123"].messages)




