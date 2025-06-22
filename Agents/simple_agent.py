from langchain_core.tools import tool
import json

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y

@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x

#print(add)
#print(f"{add.name=}\n{add.description=}")
print(add.args_schema.model_json_schema())



### Calling a tool with LLM output parsed as string
llm_output_string = "{\"x\": 5, \"y\": 2}"  # this is the output from the LLM
llm_output_dict = json.loads(llm_output_string)  # load as dictionary
print("LLM output as dictionary:", llm_output_dict)

print(exponentiate.func(**llm_output_dict))