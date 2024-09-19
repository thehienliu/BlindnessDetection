from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from tools.search import get_google_search_information
from tools.retrieve import get_diabetic_retinopathy_context
from private_key import OPENAI_API_KEY


def information_lookup(input: str) -> str:

    # Setup prompt template
    template = """Analyze the user's input to determine the appropriate response.
    If the input relates to Diabetic Retinopathy, provide relevant context based on user input.
    For other queries, perform a Google search and return the search results.
    User's input:
    {user_input}"""

    prompt_template = PromptTemplate(template=template, input_variables=["user_input"])

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)

    tools_for_agent = [
        get_google_search_information,
        get_diabetic_retinopathy_context,
    ]

    # Initialize agent
    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    result = agent.run(prompt_template.format_prompt(user_input=input))
    return result
