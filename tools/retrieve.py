import json
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from private_key import OPENAI_API_KEY

with open("utils\diabetic_disease_context.json", "r") as f:
    diabetic_disease_contexts = json.load(f)


@tool
def get_diabetic_retinopathy_context(input: str) -> str:
    """Fetches the most relevant context for Diabetic Retinopathy based on user input. If the retrieved context isn't suitable, consider refining your input to obtain more accurate results."""

    # Setup template
    template = """
    Given the user's input related to Diabetic Retinopathy, select the title that most accurately matches the input.
    User's input:
    {user_input}

    Titles:
    0. Introduction
    1. Key Facts about Diabetic retinopathy
    2. How can Diabetes affect our eyes?
    3. Diabetic retinopathy
    4. Diabetic macular edema (DME)
    5. Proliferative retinopathy
    6. Glaucoma
    7. Other symptoms of glaucoma include:
    8. Cataracts
    9. When do you need to see a doctor?
    10. Stages of Diabetic Retinopathy
    11. Symptoms and Detection
    12. Symptoms of diabetic retinopathy:
    13. Detection and diagnosis of diabetic retinopathy
    14. Supplementary testing may be required which include:
    15. A thorough dilated eye examination allows the doctor to observe the retina for:
    16. Prevention and treatment
    17. Treatment of Diabetic Retinopathy
    18. What can a diabetic do to prevent or slow down the progression of diabetic retinopathy?
    19. Treatment of Diabetic macular edema (DME)
    20. Treatment of proliferative diabetic retinopathy (PDR)
    21. Vitrectomy
    22. FAQ relating to Diabetes eye disease
    23. How are diabetes and eye disease related?
    24. What is diabetic eye disease?
    25. What is the most prevalent diabetic eye disease?
    26. What are the most significant symptoms?
    27. Who is more susceptible to getting diabetic retinopathy?
    28. How does one detect diabetic retinopathy?
    29. Can diabetic retinopathy be effectively treated?
    30. Can diabetic retinopathy be avoided?
    31. How commonplace are the other diabetic eye diseases?
    32. What research are done?
    33. What can a person do to protect their vision?
    34. Statistics
    
    Your answer should only contain the title number.
    """

    prompt_template = PromptTemplate(template=template, input_variables=["user_input"])
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_API_KEY)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    string_index = chain.invoke({"user_input": input})
    index = int(string_index["text"].split(".")[0])

    context = diabetic_disease_contexts[index]

    return context
