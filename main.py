import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import google.genai as genai
from langchain.agents import create_agent

load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Change this from "gemini-3-flash"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a research assistant that helps generate a research paper. "
        "Answer the user query and use necessary tools."
    ),
    response_format=ResearchResponse,
)

raw_response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Explain the history of Yousafzai tribe"}
        ]
    }
)
print(raw_response)