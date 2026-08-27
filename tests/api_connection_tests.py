import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Required Imports for LLM Clients
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    DEEPSEEK_API_KEY: str

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

settings = Settings()

print("Testing connections...")

try:
    openai_client = ChatOpenAI(
        model="gpt-5.4-nano", 
        api_key=settings.OPENAI_API_KEY,
        temperature=0  
    )
    res = openai_client.invoke("Respond with the exact word 'Connected'.")
    print(f"OpenAI: {res.content}")
except Exception as e:
    print(f"OpenAI Failed: {e}")

try:
    gemini_client = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite", 
        api_key=settings.GEMINI_API_KEY,
        thinking_level="minimal"
    )
    res = gemini_client.invoke("Respond with the exact word 'Connected'.")
    print(f"Gemini: {res.content[0].get('text', '')}")
except Exception as e:
    print(f"Gemini Failed: {e}")

try:
    deepseek_client = ChatDeepSeek(
        model="deepseek-v4-flash", 
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
)
    res = deepseek_client.invoke("Respond with the exact word 'Connected'.")
    print(f"DeepSeek: {res.content}")
except Exception as e:
    print(f"DeepSeek Failed: {e}")