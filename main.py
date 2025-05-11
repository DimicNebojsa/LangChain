
import os
from dotenv import load_dotenv
from getpass import getpass
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
      "Enter OpenAI API Key: "
)


### LLM model

openai_model = "gpt-4o-mini"

from langchain_openai import ChatOpenAI

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model=openai_model)

# For unique creative responses
creative_llm = ChatOpenAI(temperature=0.9, model=openai_model)

### Article

from article import article_text

article = article_text

### Prompts

from langchain.prompts import (
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate
)

# Defining the system prompt (how the AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."
)

# Defining the system prompt (how the AI should act)
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps generate article titles."
)

# the user prompt is provided by the user, in this case however the only dynamic
# input is the article
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for a article.
                The article is here for you to examine {article}
                
                The name should be based of the context of the article.
                Be creative, but make sure the names are clear, catchy,
                and relevant to the theme of the article.
                
                Only output the article name, no other explanation or
                text can be provided.""",
    input_variables=["article"]
)

print(user_prompt.format(article="TEST STRING").content)



# if __name__ == '__main__':
#     print('Start')
#     print(os.environ['OPENAI_API_KEY'])



