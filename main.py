from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import os

# Initialize the OpenAI LLM (or any other LLM of your choice)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Create the agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

def chatbot(query):
    try:
        response = agent_executor.run(query)
        return response
    except Exception as e:
        return str(e)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    print("Bot:", chatbot(user_input))
