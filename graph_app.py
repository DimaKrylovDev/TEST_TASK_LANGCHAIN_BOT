import datetime
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.chat_models import ChatOllama
from langgraph.graph.state import StateGraph
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START
from langchain_core.messages import SystemMessage
from semantic_router.utils.function_call import FunctionSchema

system_prompt = SystemMessage(
    content=(
        "You are a helpful assistant with access to a tool called get_current_time.\n"
        "It returns the current UTC time in ISO‑8601 format.\n"
        "When the user asks what time it is — in English or Russian — you MUST call the get_current_time tool.\n\n"
        "Examples (EN):\n"
        "Q: How much time?\n"
        "A: (tool:get_current_time)\n"
        "Q: What time is it?\n"
        "A: (tool:get_current_time)\n"
        "Q: Give me the time now\n"
        "A: (tool:get_current_time)\n"
        "Q: I need the current time\n"
        "A: (tool:get_current_time)\n"
        "Q: Time please\n"
        "A: (tool:get_current_time)\n"
        "Q: What is the time now in UTC?\n"
        "A: (tool:get_current_time)\n\n"
        "Примеры (RU):\n"
        "Q: Сколько времени?\n"
        "A: (tool:get_current_time)\n"
        "Q: Сколько сейчас времени?\n"
        "A: (tool:get_current_time)\n"
        "Q: Подскажи текущее время\n"
        "A: (tool:get_current_time)\n"
        "Q: Время?\n"
        "A: (tool:get_current_time)\n"
        "Q: Который час?\n"
        "A: (tool:get_current_time)\n"
        "Q: Сколько времени в UTC?\n"
        "A: (tool:get_current_time)\n"
        "Q: Скажи точное время\n"
        "A: (tool:get_current_time)\n"
    )
)

llm = ChatOllama(model='tinyllama')

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def get_current_time() -> dict:
    """Return the current UTC time in ISO‑8601 format.
    Example → {"utc": "2025‑05‑21T06:42:00Z"}"""
    return {"utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}

time_schema = FunctionSchema(get_current_time).to_ollama()

time_schema["function"]["description"] = "Getting the current time in UTC"
time_schema["function"]["parameters"] = {
    "type": "object",
    "properties": {},  
    "required": []
}


def get_system_tools_prompt(system_prompt: str, tools: list[dict]):
    tools_str = "\n".join([str(tool) for tool in tools])
    return (
        f"{system_prompt}\n\n"
        f"You may use the following tools:\n{tools_str}"
    )

def make_tool_graph():
    """Make a tool-calling agent"""    
    def agent_call(state: State):
        user_message = state['messages'][-1]
        prompt = [system_prompt, user_message]
        response = llm.invoke(prompt)
        if 'tool:get_current_time' in response.content:
            result = get_current_time()
            answer = f"Time: {result}"
        else:
            answer = response.content
            
        return {'messages': [HumanMessage(content=answer)]}
    
    graph_workflow = StateGraph(State)
    
    graph_workflow.add_node('agent', agent_call)
    graph_workflow.add_edge(START, 'agent')
    graph_workflow.add_edge('agent', END)
    
    agent = graph_workflow.compile()
    return agent

graph_app = make_tool_graph()
