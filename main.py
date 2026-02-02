import os
import pandas as pd
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import HumanMessage
from langchain.schema.runnable import RunnableLambda
from langchain.tools import tool


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- LCEL: simple explanation ----------
explain_prompt = ChatPromptTemplate.from_template(
    "Explain the following concept clearly:\n{topic}"
)

explain_chain = explain_prompt | llm | StrOutputParser()
print(explain_chain.invoke({"topic": "Linear Regression"}))


# ---------- LCEL with real data ----------
try:
    df = pd.read_csv("data.csv")

    summary = df.describe().to_string()
    print(summary)

    df.hist(figsize=(10, 8))
    plt.show()

    qa_prompt = ChatPromptTemplate.from_template(
        "Dataset summary:\n{summary}\nQuestion:\n{question}"
    )

    qa_chain = qa_prompt | llm | StrOutputParser()

    print(
        qa_chain.invoke({
            "summary": summary,
            "question": "What insights can we draw?"
        })
    )

except FileNotFoundError:
    pass


# ---------- Tool definitions ----------
@tool
def add(a: int, b: int) -> int:
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    return a * b


@tool
def divide(a: int, b: int) -> float:
    return a / b


tools = [add, subtract, multiply, divide]
tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)


# ---------- Manual tool execution ----------
def run_tools(tool_calls):
    results = []
    for call in tool_calls:
        fn = tool_map[call["name"]]
        results.append(fn(**call["args"]))
    return results


tool_chain = (
    llm_with_tools
    | RunnableLambda(lambda x: x.tool_calls)
    | RunnableLambda(run_tools)
)


queries = [
    "What is (8 * 5) + 10?",
    "Calculate (12 / 4) * 6",
    "What is 100 minus 37?",
    "Add 45 and 55, then divide by 10"
]

for q in queries:
    print(q, "=>", tool_chain.invoke(q))


# ---------- LLM reasoning over tool output ----------
query = "What is (8 * 5) + 10?"
response = llm_with_tools.invoke(query)
tool_results = run_tools(response.tool_calls)

final = llm.invoke([
    HumanMessage(content=query),
    HumanMessage(content=f"Tool results: {tool_results}")
])

print(final.content)
