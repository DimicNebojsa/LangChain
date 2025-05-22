
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

# the user prompt is provided by the user, in this case however the only dynamic
# input is the article
user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a name for a article.
    The article is here for you to examine {article}.
                
    The name should be based of the context of the article.
    Be creative, but make sure the names are clear, catchy,
    and relevant to the theme of the article.
                
    Only output the article name, no other explanation or
    text can be provided.""",
    input_variables=["article"]
)

from langchain.prompts import ChatPromptTemplate

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

#print(first_prompt.format(article="TEST STRING"))


## First Chain

chain_one = (
    {"article": lambda x: x['article']}
    | first_prompt
    | creative_llm
    | {"article_title" : lambda x: x.content}
)


article_title_msg = chain_one.invoke({"article": article})

#print(article_title_msg)



## Second Promtp

second_user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a description for
the article. The article is here for you to examine:

---

{article}

---

Here is the article title '{article_title}'.

Output the SEO friendly article description. Do not output
anything other than the description.""",
    input_variables=["article", "article_title"]
)

second_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    second_user_prompt
])

chain_two = (
    {
        "article": lambda x : x["article"],
        "article_title": lambda x: x['article_title']
    }
    | second_prompt
    | llm
    | {"summary": lambda x: x.content}
)


#article_description_msg = chain_two.invoke({
#    "article": article,
#    "article_title": article_title_msg["article_title"]
#
#})


#print(article_description_msg)

## Third Chain

third_user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with creating a new paragraph for the
article. The article is here for you to examine:

---

{article}

---

Choose one paragraph to review and edit. During your edit,
ensure you provide constructive feedback to the user so they
can learn where to improve their own writing.""",
    input_variables=["article"]
)

# prompt template 3: creating a new paragraph for the article
third_prompt = ChatPromptTemplate.from_messages([
 system_prompt,
 third_user_prompt
])

from pydantic import BaseModel, Field

class Paragraph(BaseModel):
 original_paragraph: str = Field(description="The original paragraph")
 edited_paragraph: str = Field(description="The improved edited paragraph")
 feedback: str = Field(description=(
        "Constructive feedback on the original paragraph"
    ))


structured_llm = creative_llm.with_structured_output(Paragraph)


# chain 3: inputs: article / output: article_para
chain_three = (
    {"article": lambda x: x["article"]}
    | third_prompt
    | structured_llm
    | {
        "original_paragraph": lambda x: x.original_paragraph,
        "edited_paragraph": lambda x: x.edited_paragraph,
        "feedback": lambda x: x.feedback
    }
)

#out = chain_three.invoke({"article": article})
#print(out)


### create image from text
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate

image_prompt = PromptTemplate(
    input_variables=["article"],
    template=(
        "Generate a prompt with less then 500 characters to generate an image "
        "based on the following article: {article}"
    )
)

from skimage import io
import matplotlib.pyplot as plt
from langchain_core.runnables import RunnableLambda

# def generate_and_display_image(image_prompt):
#     image_url = DallEAPIWrapper().run(image_prompt)
#     image_data = io.imread(image_url)
#
#     # And update the display code to:
#     plt.imshow(image_data)
#     plt.axis('off')
#     plt.show()
#
# # we wrap this in a RunnableLambda for use with LCEL
# image_gen_runnable = RunnableLambda(generate_and_display_image)


# chain 4: inputs: article, article_para / outputs: new_suggestion_article
chain_four = (
    {"article": lambda x: x["article"]}
    | image_prompt
    | llm
    | (lambda x: x.content)
    #| image_gen_runnable
)


from article import article_text2

article = article_text2

prompt = chain_four.invoke({"article": article})

import openai
openai.api_key = os.environ["OPENAI_API_KEY"] ##"your_api_key"

response = openai.images.generate(
    model="dall-e-3",
    prompt= prompt,  #"A surreal sunset over futuristic city skyline",
    size="1024x1024",
    response_format="url"
)

image_url = response.data[0].url
print(image_url)





# if __name__ == '__main__':
#     print('Start')
#     print(os.environ['OPENAI_API_KEY'])



