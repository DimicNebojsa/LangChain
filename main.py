
import os
from dotenv import load_dotenv
from getpass import getpass
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass(
      "Enter OpenAI API Key: "
)




if __name__ == '__main__':
    print('Start')
    print(os.environ['OPENAI_API_KEY'])



