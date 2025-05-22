from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY is missing in your .env file!")

langsmith_key = os.getenv("LANGSMITH_API_KEY")
if not langsmith_key:
    raise RuntimeError("LANGSMITH_API_KEY is missing in your .env file!")

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["LANGCHAIN_API_KEY"] = langsmith_key

# below should not be changed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# you can change this as preferred
os.environ["LANGCHAIN_PROJECT"] = "my-project-for-langchain-course"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

# response = llm.invoke("Hello, how are you today. Tell me most important discovery in stochastic calculus theory please")
response = llm.invoke("You are recruiter for LLM engineer position. Cna oyu tell me what skills successful candidate should have?")

print(response.content)


