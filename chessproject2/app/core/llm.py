from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from app.core.config import settings

import logging
from psycopg_pool import ConnectionPool

db_pool = ConnectionPool(conninfo=settings.DATABASE_URL)

openai_client = ChatOpenAI(
    model="gpt-5.4-nano", 
    api_key=settings.OPENAI_API_KEY,
    temperature=0  
)

gemini_client = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    api_key=settings.GEMINI_API_KEY,
    temperature=0,
    thinking_config={
        "thinking_level": "MINIMAL" 
    }
)
deepseek_client = ChatDeepSeek(
    model="deepseek-v4-flash", 
    api_key=settings.DEEPSEEK_API_KEY,
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {
                "type": "disabled"
            }
        }
    }
)

PRICING_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50}
}


def get_llm(model_name: str):
    match model_name:
        case "Gemini 3.5 Flash":
            return gemini_client
        case "GPT-5.5 Instant":
            return openai_client
        case "DeepSeek V4-Flash":
            return deepseek_client
        case _:
            raise ValueError(f"Unsupported model name: {model_name}")

def generate(
    prompt: str, 
    model_name: str, 
    
    game_id: int, 
    move_number: int
    ) -> str:
    
    llm = get_llm(model_name)
    
    try:
        response = llm.invoke(prompt)
        
        token_usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        
        cost = 0.0
        if model_name in PRICING_PER_MILLION:
            rates = PRICING_PER_MILLION[model_name]
            cost = ((prompt_tokens / 1_000_000) * rates["input"]) + ((completion_tokens / 1_000_000) * rates["output"])
        
        try:
            # The connection acts as a context manager and auto-commits the transaction on exit
            with db_pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO token_logs 
                        (game_id, move_number, model_name, prompt_tokens, completion_tokens, total_tokens, cost)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (game_id, move_number, model_name, prompt_tokens, completion_tokens, total_tokens, cost)
                    )
        except Exception as db_err:
            logging.error(f"Failed to log tokens to PostgreSQL: {db_err}")
            
        return str(response.content)
        
    except Exception as e:
        logging.error(f"API call failed for model {model_name}. Error: {str(e)}")
        raise RuntimeError(f"Failed to generate completion: {str(e)}")