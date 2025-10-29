import warnings
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uuid
from therapist import AgentState, langgraph_agent

# Code starts from here -----------------------------------
load_dotenv()
warnings.filterwarnings("ignore")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Men's Best Friend", 
    description="API for the LangGraph-powered AI Agent for providing help to men",
    version="1.0.0"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. In production, specify your frontend's domain(s)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic Model for Request/Response Validation
class MessageRequest(BaseModel):
    user_id:str
    user_message:str

class MessageResponse(BaseModel):
    agent_response: str
    user_id: str
    state: dict

# -------------------- Initialize LangGraph Agent --------------------
try:
    agent_graph = langgraph_agent()  # Compile your graph once at startup
    logger.info("LangGraph agent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangGraph agent: {e}", exc_info=True)
    agent_graph = None

# -------------------- FastAPI Endpoints --------------------
@app.get("/health")
def health_check():
    return {'message': 'ok'}

user_sessions = {}
@app.post("/message", response_model=MessageResponse)
def handle_message(request: MessageRequest):
    try:
        if not agent_graph:
            logger.error("Agent not initialized when /message endpoint was called.")
            raise HTTPException(
                status_code=500,
                detail="Server is not fully initialized. Please check logs for startup errors.",
            )
        
        user_id = request.user_id or uuid.uuid4().hex
        user_message = request.user_message.strip()

        # state: AgentState = {
        #     'user_message': user_message,
        #     'user_id': user_id,
        #     'messages': [{'role':'user', 'content':user_message}]
        # }

        state: AgentState = user_sessions.get(user_id, {
            "user_id": user_id,
            "messages": []
        })

        state["user_message"] = user_message
        state["messages"].append({"role": "user", "content": user_message})

        logger.info(f" State Update::::::::::::::::::::::::::::::: {state}")
        result = agent_graph.invoke(state)

        ai_message = {'role': 'ai', 'content': result['answer']}
        state['messages'].append(ai_message)

        user_sessions[user_id] = state

        final_agent_response = result.get("answer", "I'm sorry, I couldn't process that request.")
        logger.info(f"Agent Response: '{final_agent_response}'")

        return MessageResponse(
            agent_response=final_agent_response,
            user_id=user_id,
            state=state
        )
    
    except Exception as e:
        logger.error(
            f"FATAL ERROR during message processing for user {user_id}: {e}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}. Please try again.")
