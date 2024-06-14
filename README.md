# DreamAI
This is the LLM part for the Study Project: DreamGuard <br/>

Use python>=3.10 (preferably a virtual environment with 3.12.3) 
Use pip to install the dependencies in the requirements: <br>
'pip install -r requirements.txt' <br/>

create a .env file in the root of the project folder with all the needed API keys: <br/>
### .env
LANGCHAIN_API_KEY = "lsv2_pt..." <br>
GROQ_API_KEY = "gsk..." <br>
OPENAI_API_KEY = "sk..." <br>
LANGCHAIN_PROJECT = "DreamGuardTest" <br/>

*API keys with limited resources are available on Slack pinned in general* <br/>

To run the chatbot with a streamlit GUI run the commmand: <br>
'streamlit run streamlit_chatbot.py' <br/>

Edit prompt templates in prompt_templates.py
