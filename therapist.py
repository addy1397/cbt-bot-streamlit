import logging
import operator
import warnings
from dotenv import load_dotenv
from functools import lru_cache
# LangChain & Gemini
from langgraph.graph import StateGraph, END
from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Any
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field, ValidationError

# Code starts from here --------------------
warnings.filterwarnings("ignore")
load_dotenv()
logger = logging.getLogger(__name__)

# Model
@lru_cache(maxsize=1)
def get_llm():
    """Initializes and returns the ChatGoogleGenerativeAI model."""
    try:
        return ChatGoogleGenerativeAI(
            model='gemini-2.5-flash-lite',
            temperature=0.5,
            max_retries=3,
            request_timeout=30,
            max_tokens=100
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        return None
    
llm = get_llm()
if not llm:
    logger.warning("LLM could not be initialized. Agent may not function.")

# Output Parser
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# State
class AgentState(TypedDict):
    user_id: Optional[str]
    user_message: Optional[str]
    answer: Optional[str]
    intent: Optional[str]
    severity_level: Optional[str]  # low, medium or high
    user_emotion: Optional[str]  # e.g., sad, anxious, fearful, angry
    emotion_confidence: Optional[float]  # model's confidence in detected emotion
    trauma_flag: Optional[bool]
    phobia_type: Annotated[List[str], operator.add]  # e.g., water, heights
    messages: Annotated[List[Dict[str, str]], operator.add]
    current_step: Optional[str]

class RouterOutput(BaseModel):
    intent: Literal['casual', 'crisis', 'listener', 'solution']
    severity_level: Literal['low', 'medium', 'high']
    user_emotion: Literal['happy', 'excited', 'calm', 'neutral',
                          'sad', 'angry', 'frustrated', 'anxious',
                            'fearful', 'vulnerable', 'relieved', 'shame']
    
    trauma_flag: Optional[bool]
    phobia_type: Optional[List[str]]


# LangGraph Nodes ------------------

# Router Node
def router_node(state: AgentState) -> dict:
    logger.info("---Router Node---")
    user_message = state.get('user_message', '').strip()
    logger.info(f"Intent: {state.get('intent')}\n Severity Level: {state.get('severity_level')}")

    if not user_message:
        logger.error("Router Node: Input is Empty")
        return {'answer': 'Please provide some input'}

    if not llm:
        logger.error("Router Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step':"llm_unavailable", 'messages': [{'role': 'ai', 'content': response_content}]}
    
    router_prompt = PromptTemplate(
        template="""
            You are an intelligent and capable Therapist tasked with providing help for anxiety, depression, issues and motivating the patient.
            For the given user message, first classify its primary intent, and provide the severity level (low, medium, high).

            **Intent Classification Rules:**
            - **casual**: user is chatting casually, asking general questions, or giving neutral statements not related to mental health concerns.
            - **crisis**: User expresses thoughts or intentions that may be harmful to themselves or others, including suicidal ideation or extreme distress.
            - **listener**: User expresses emotions like anxiety, depression, or lack of motivation, and wants to vent or be understood.
            - **solution**: User explicitly asks for guidance, coping strategies, advice, or motivation (e.g., "help me", "what should I do?").
            - **irrelevant**" User asks questions unrelated to mental health or emotional support, e.g., random queries or off-topic questions.

            **Feature Extraction Rules:**
            - user_emotion:  Detect the primary user emotion: 'happy', 'excited', 'calm', 'neutral', 'sad', 'angry', 'frustrated', 'anxious', 'fearful', 'vulnerable', 'relieved', 'shame'.
            - Detect if the user mentions trauma or phobias. If yes, set trauma_flag=true and phobia_type accordingly.
                **Examples**
                - User: I fell into a river and drowned, I panicked and made my life difficult for a year -> {{'phobia_type': 'aquaphobia', 'trauma_flag': 'true'}}
                - User: I am feeling anxious for a meeting -> {{'trauma_flag': 'false'}}

            Respond strictly in JSON with keys: "intent", "severity_level", "user_emotion", "trauma_flag" and "phobia_type".
            User Message: "{user_message}"
        """,
        input_variables=['user_message'])
    
    router_chain = router_prompt | llm | json_parser
    response = router_chain.invoke({"user_message": user_message})
    intent = response.get('intent') or 'casual'
    severity_level = response.get('severity_level') or 'low'
    user_emotion = response.get('user_emotion') or 'neutral'
    trauma_flag = response.get('trauma_flag') or False
    phobia_type = response.get('phobia_type') or []
    if isinstance(phobia_type, str):
        phobia_type = [phobia_type]

    return {
        'user_message': user_message,
        'intent': intent,
        'severity_level': severity_level,
        'user_emotion': user_emotion,
        'trauma_flag': trauma_flag,
        'phobia_type': phobia_type,
        'current_step': 'router',
        'messages': [{'role': 'user', 'content': user_message}]
    }


# ChitChat Node
def chitchat_node(state: AgentState) -> dict:
    """Handles Casual Conversations"""
    logger.info("---ChitChat Node---")
    user_message = state.get('user_message', '').strip()

    if not llm:
        logger.error("Chichat Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step':"llm_unavailable", 'messages': [{'role': 'ai', 'content': response_content}]}
    
    formatted_messages = []
    for msg in state.get("messages", [])[:8]:
        if msg.get("role") == 'user':
            formatted_messages.append(f"Human: {msg.get('content', '')}")
        elif msg.get("role") == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
    history_string = "\n".join(formatted_messages)

    if history_string.strip():
        history_section = f"Conversation so far:\n{history_string}\nReference previous conversation if relevant."
    else:
        history_section = "This is your first conversation. Do not reference any previous chat."
    
    chitchat_prompt = PromptTemplate(
    template="""
        You are a warm, emotionally intelligent AI friend who genuinely cares about the user.
        You can show interest in their conversation and gently follow up if they mentioned something personal earlier.

        Your goals:
        1. Speak in a friendly, natural tone — show real interest and care.
        2. Reference past topics if available and relevant. For example:
        - If they talked about work, ask how work is going.
        - If they shared something they were struggling with, ask if it got better.
        - If this is your first conversation with the user, do not reference any previous chat
        3. Offer light encouragement or supportive thoughts if needed.
        4. You can give simple life advice or comforting messages, but do NOT give medical or clinical guidance.
        5. Keep your messages concise, natural, and caring — like a close friend texting back.

        Conversation so far:
        {history_section}

        User says: "{user_message}"

        The user is currently feeling {user_emotion}. Respond warmly and continue the conversation like a supportive friend.
        """,
        input_variables=["user_message", "history_section", "user_emotion"]
        )

    chitchat_chain = chitchat_prompt | llm | str_parser

    try:
        response_content = chitchat_chain.invoke({
            "user_message": user_message,
            "history_section": history_section,
            "user_emotion": state.get('user_emotion')
        })

        return {
            'user_message': user_message,
            "intent": "casual",
            "current_step" : "chitchat_response",
            "user_emotion": state.get('user_emotion'),
            "answer": response_content,
            "messages": [{'role':'ai', 'content': response_content}]
        }
    except Exception as e:
        logger.error(f"Chichat: Error in LLM Call: {e}", exc_info=True)
        error_message = "I'm having trouble with general conversation right now"
        return {
            'user_message': user_message,
            "intent": "casual",
            "current_step" : "chitchat_response",
            "answer": response_content,
            "messages": [{'role':'ai', 'content': error_message}]
        }
    

# Smart Listener Node
def smart_listener_node(state: AgentState) -> dict:
    """Listens to Patients and let them vent out their emotions"""
    logger.info("---Smart Listener Node---")
    logger.info(f"Severity Level in Smart Listener: {state.get('severity_level')}")
    logger.info(f"User Emotion in Smart Listener: {state.get('user_emotion')}")
    logger.info(f"Phobia in Smart Listener: {state.get('phobia_type')}")
    logger.info(f"Is Trauma in Smart Listener: {state.get('trauma_flag')}")
    
    if not llm:
        logger.error("Support Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step': 'llm_unavailable', 'messages':[{'role': 'ai', 'content': response_content}]}

    user_message = state.get("user_message", "").strip()
    severity_level = state.get("severity_level", "low")
    user_emotion = state.get("user_emotion", "neutral")
    trauma_flag = state.get("trauma_flag", False)
    phobia_type = state.get("phobia_type", [])

    formatted_messages = []

    for msg in state.get('messages', [])[:8]:
        if msg.get('role') == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
        else:
            formatted_messages.append(f"Human: {msg.get('content', '')}")

    history_string = "\n".join(formatted_messages)

    # Handle first conversation
    if not history_string.strip():
        reference_instruction = "This is your first conversation. Do not reference any previous chat."
    else:
        reference_instruction = "You may reference previous conversation if relevant."

    conversation_prompt = PromptTemplate(
        template= """
            You are an professional, empathetic Therapist tasked with letting patient vent out their feelings and emotions.
            Your goal is to **listen actively, understand feelings, and guide gently**, without forcing solutions.

            ### Requirements:
            1. Bot should act as a supportive, empathetic listener.
            2. Allow users to vent freely without interruption or any distraction from main issue.
            3. Provide validating, compassionate responses (“I hear you,” “That must feel heavy”).
            4. Ensure tone is non-judgmental, warm, and human-like.
            5. Bot should ask clarifying/follow-up questions.
                - Follow up 
                    - Extending situation 
                    - Should I suggest a solution 

            ### Context:
            1. Audience is the Patient
            2. You are in a therapy session
            3. Patient is venting out their feelings and you have to route them to the solution.

            ### Constraints:
            1. Response must be **100–200 characters**.
            2. Use concise, simple, everyday language.
            3. Minimal Markdown, no complex terms.
            4. Response must be readable
            
            Conversation context:
            {history_string}
            {reference_instruction}

            Current user message:
            "{user_message}"

            User Emotion: {user_emotion}
            Severity Level: {severity_level}
            Trauma Flag: {trauma_flag}, Phobia(s): {phobia_type}

            Generate a warm, concise, supportive response with a gentle follow-up question if appropriate.
        """,
        input_variables=['user_message', 'severity_level', 'history_string', 'reference_instruction', 'user_emotion', 'trauma_flag', 'phobia_type']
    )

    try:
        conversation_chain = conversation_prompt | llm | str_parser
        response_content = conversation_chain.invoke({
            'user_message': user_message,
            'severity_level': severity_level,
            'history_string': history_string,
            'reference_instruction': reference_instruction,
            'user_emotion': user_emotion,
            'trauma_flag': trauma_flag,
            'phobia_type': phobia_type
        })

        return {
            'user_message': user_message,
            'intent': 'listener',
            'current_step': 'listener_response_successful',
            'answer': response_content,
            'messages': [{'role':'ai', 'content':response_content}]
        }
    except Exception as e:
        logger.error(f"Listener Node: LLM call failed: {e}", exc_info=True)
        error_message = "I'm having trouble generating a response right now, but I care about what you're going through."
        return {
            'user_message': user_message,
            "intent": "listener",
            "current_step": "listener_response",
            "answer": error_message,
            'messages': [{'role':'ai', 'content':error_message}]
        }

    
# Solution Node
def solution_node(state: AgentState) -> dict:
    """Helps Patient deal with situation"""
    logger.info("---Solution Node---")
    logger.info(f"User Emotion in Solution: {state.get('user_emotion')}")
    logger.info(f"Severity Level in Solution: {state.get('severity_level')}")
    logger.info(f"Phobia in Solution: {state.get('phobia_type')}")
    logger.info(f"Is Trauma in Solution: {state.get('trauma_flag')}")

    if not llm:
        logger.error("Solution Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step': 'llm_unavailable', 'messages':[{'role': 'ai', 'content': response_content}]}

    user_message = state.get("user_message", "").strip()
    severity_level = state.get("severity_level", "low")
    
    
    formatted_messages = []

    for msg in state.get('messages', [])[:8]:
        if msg.get('role') == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
        else:
            formatted_messages.append(f"Human: {msg.get('content', '')}")

    history_string = "\n".join(formatted_messages)

    # Handle first conversation
    if not history_string.strip():
        reference_instruction = "This is your first conversation. Do not reference any previous chat."
    else:
        reference_instruction = "You may reference previous conversation if relevant."

    solution_prompt = PromptTemplate(
        template="""
            You are a compassionate therapist AI helping men manage anxiety, depression, or low motivation.
            Your goal is to provide **actionable, supportive guidance** based on the user's current severity level and message.

            ### Requirements:
            1. After listening, Provide niche, practical solutions (not generic advice).
            2. Break down solutions into small, actionable steps.
            3. Offer small, practical steps the user can take right away.
            4. Provide motivation, encouragement, and empathy.
            5. Tailor guidance according to severity:
                - Depression: Low → encouragement & coping tips; Medium → routines & mindfulness; High → professional help strongly recommended.
                - Anxiety: Low → relaxation & reassurance; Medium → structured calming techniques; High → immediate support & professional help.
                - Motivation: Provide actionable steps and encouragement.
            6. Reference previous conversation context if available.
            7. End with a gentle supportive note.
            8. {reference_instruction}

            ### Context:
            1. Audience is the Patient
            2. Patient asks you for the solution of their issue

            ### Constraints:
            1. Keep response 100–200 characters, concise, simple language.
            2. Minimal Markdown, no complex terms.
            3. Response should be actionable, empathetic, and clear.

            Conversation History:
            {history_string}

            Current User Message:
            "{user_message}"

            Severity Level: {severity_level}

            Generate a friendly, natural, and supportive response with actionable guidance.
        """,
        input_variables=['user_message', 'history_string', 'severity_level', 'reference_instruction']
    )



    try:
        solution_chain = solution_prompt | llm | str_parser
        response_content = solution_chain.invoke({
            'user_message': user_message,
            'history_string': history_string,
            'severity_level': severity_level,
            'reference_instruction': reference_instruction
        })

        return {
            'user_message': user_message,
            'intent': 'solution',
            'current_step': 'solution_response_successful',
            'answer': response_content,
            'messages': [{'role':'ai', 'content':response_content}]
        }
    except Exception as e:
        logger.error(f"Solution Node: LLM call failed: {e}", exc_info=True)
        error_message = "I'm having trouble generating a response right now, but I care about what you're going through."
        return {
            'user_message': user_message,
            "intent": "solution",
            "current_step": "solution_response",
            "answer": error_message,
            'messages': [{'role':'ai', 'content':error_message}]
        }

# Crisis Node
def crisis_node(state: AgentState) -> dict: 
    """Handles crisis situation"""
    logger.info("---Crisis Node---")
    user_message = state.get("user_message", "").strip()
    severity_level = state.get('severity_level', 'low')

    if severity_level == 'low':
        response_content = """
            I can sense that things have been stressful for you lately. It's important to take care of yourself. 
            Try reaching out to a friend, family member, or a counselor to talk about what you're going through. 
            Remember, even small steps towards support can make a big difference. 
            If anything feels more serious or unsafe, please contact a trained professional immediately.
        """
    elif severity_level == 'medium':
        response_content = """
            I’m genuinely concerned about your well-being. It sounds like things are really weighing on you. 
            I strongly encourage you to speak with a trained professional as soon as possible. 
            You can contact a local counselor, mental health professional, or a trusted support line. 
            Remember, reaching out is a brave step, and help is available.
        """
    else:
        response_content = """
            I’m very concerned about your safety. You’re not alone, and immediate help is available. 
            Please contact a trained professional right now. Here are some resources you can reach out to:

            - **National Suicide Prevention Helpline (India): 022-2772 6771 / 1800 233 3330**
            - Connect directly to a local mental health professional, counselor, or human therapist.
            - If you feel unsafe right now, call emergency services immediately.

            It’s critical to speak to someone trained who can help you. Please reach out without delay.
        """

    return {
        'user_message': user_message,
        'intent': 'crisis',
        'severity_level': severity_level,
        'answer': response_content,
        'current_step': 'crisis_situation',
        'messages': [{'role':'ai', 'content':response_content}]
    }

# Irrelevant Node
def irrelevant_node(state: AgentState) -> dict:
    """Handles irrelevant user queries."""
    logger.info("---Irrelevant Node---")
    user_message = state.get("user_message", "").strip()
    response_content = "Sorry, I don't think this message is related to your condition, so I can't help with that"
    return {
        'user_message': user_message,
        'intent': 'irrelevant',
        'current_step': 'irrelevant',
        'answer': response_content,
        'messages': [{'role': 'ai', 'content': response_content}]
    }


def langgraph_agent():
    builder = StateGraph(AgentState)

    # Adding Nodes
    builder.add_node("router", router_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("crisis", crisis_node)
    builder.add_node("solution", solution_node)
    builder.add_node("smart_listener", smart_listener_node)
    builder.add_node("irrelevant", irrelevant_node)

    builder.set_entry_point("router")

    # Adding Edges
    builder.add_conditional_edges(
        "router",
        lambda state: state.get('intent', 'irrelevant'),
        {
            "casual": "chitchat",
            "crisis": "crisis",
            "listener": "smart_listener",
            "solution": "solution",
            "irrelevant": "irrelevant"
        }
    )

    builder.add_edge('chitchat', END)
    builder.add_edge('crisis', END)
    builder.add_edge('smart_listener', END)
    builder.add_edge('solution', END)
    builder.add_edge('irrelevant', END)

    graph = builder.compile()
    graph.get_graph().print_ascii()
    return graph