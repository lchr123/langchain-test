from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
import os
from secret_key import OPENAI_KEY, DB_CONNECTION

# OpenAI key, db connection
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
db = SQLDatabase.from_uri(DB_CONNECTION)

# Initialize the OpenAI LLM (or any other LLM of your choice)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

input_samples = [
    {
        "input": "2024年1月3日の時点で預金残高がいくらあるか教えていただけますか",
        "query": "SELECT date, actual_total FROM deposits_cumulative WHERE date = '2024-01-03'",
    }
]

system_prefix = """
    Any DML statement (INSERT, UPDATE, DELETE, DROP etc.) is strictly forbidden to the database.
"""

example_selector = SemanticSimilarityExampleSelector.from_examples(
    input_samples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Create the agent
agent_executor = create_sql_agent(
    llm,
    db=db,
    prompt=full_prompt,
    agent_type="openai-tools",
    verbose=True
)

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
