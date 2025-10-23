import logging
import operator
import warnings
from dotenv import load_dotenv
from functools import lru_cache
# LangChain & Gemini
from langgraph.graph import StateGraph, END
from typing import Annotated, Dict, List, Literal, Optional, TypedDict, Any
from langchain_core.prompts import PromptTemplate
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
            model='gemini-2.5-flash-lite', # Using a more recent model
            temperature=0.5,
            max_retries=3,
            request_timeout=30
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
    messages: List[Dict[str,str]]
    current_step: Optional[str]

class RouterOutput(BaseModel):
    intent: Literal['casual', 'crisis', 'support', 'solution']
    severity_level: Literal['low', 'medium', 'high']

# LangGraph Nodes ------------------

# Router Node
def router_node(state: AgentState) -> dict:
    logger.info("---Router Node---")
    # Get the last user message from the messages list
    user_message = ""
    if state.get('messages'):
        user_message = state['messages'][-1].get('content', '').strip()

    logger.info(f"Routing message: {user_message}")

    if not user_message:
        logger.error("Router Node: Input is Empty")
        return {'answer': 'Please provide some input'}

    if not llm:
        logger.error("Router Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step':"llm_unavailable", 'messages': state['messages'] + [{'role': 'ai', 'content': response_content}], 'intent': 'irrelevant'}
    
    router_prompt = PromptTemplate(
        template="""
            You are an intelligent and capable Therapist tasked with providing help for anxiety, depression, issues and motivating the patient.
            For the given user message, first classify its primary intent, and provide the severity level (low, medium, high).

            **Intent Classification Rules:**
            - **casual** - user is chatting casually, asking general questions, or giving neutral statements not related to mental health concerns.
            - **crisis** - user expresses self-harm thoughts, suicidal ideation, or any immediate risk to their safety.
            - **support** - user expresses depression, anxiety, or a need for motivation.
            - **solution** - user explicitly asks for solution (for ex- 'help me', 'provide the solution').

            Respond strictly in JSON with keys: "intent", "severity_level".
            User Message: "{user_message}"
        """,
        input_variables=['user_message'])
    
    router_chain = router_prompt | llm | json_parser
    
    try:
        response = router_chain.invoke({"user_message": user_message})
        intent = response['intent']
        severity_level = response['severity_level']
    except Exception as e:
        logger.error(f"Router Node: Failed to parse LLM response: {e}")
        # Default to support if classification fails
        intent = 'support'
        severity_level = 'low'

    return {
        'intent': intent,
        'severity_level': severity_level,
        'current_step': 'router'
    }


# ChitChat Node
def chitchat_node(state: AgentState) -> dict:
    """Handles Casual Conversations"""
    logger.info("---ChitChat Node---")
    user_message = state['messages'][-1]['content']

    if not llm:
        logger.error("Chichat Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step':"llm_unavailable", 'messages': state['messages'] + [{'role': 'ai', 'content': response_content}], 'intent': 'casual'}
    
    formatted_messages = []
    # Get last 8 messages
    for msg in state.get("messages", [])[-8:]:
        if msg.get("role") == 'user':
            formatted_messages.append(f"Human: {msg.get('content', '')}")
        elif msg.get("role") == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
    history_string = "\n".join(formatted_messages)
    
    chitchat_prompt = PromptTemplate(
    template="""
        You are a warm, emotionally intelligent AI friend who genuinely cares about the user.
        You remember what they shared in the past and check in naturally — like a close friend would.
        You can ask how things have been since the last conversation and gently follow up if they mentioned something personal earlier.

        Conversation so far:
        {history_string}

        User says: "{user_message}"

        Your goals:
        1. Speak in a friendly, natural tone — show real interest and care.
        2. Reference past topics if relevant. For example:
        - If they talked about work, ask how work is going.
        - If they shared something they were struggling with, ask if it got better.
        3. Offer light encouragement or supportive thoughts if needed.
        4. You can give simple life advice or comforting messages, but do NOT give medical or clinical guidance.
        5. Keep your messages concise, natural, and caring — like a close friend texting back.

        Example tone:
        - “Hey, it’s nice to hear from you again! How have you been feeling since we last talked?”
        - “That sounds like a lot to handle, but I’m really proud of how you’re managing.”
        - “You’ve got this — want to tell me what’s been on your mind lately?”

        Now, respond warmly and continue the conversation like a supportive friend.
        """,
        input_variables=["user_message", "history_string"]
        )

    chitchat_chain = chitchat_prompt | llm | str_parser

    try:
        response_content = chitchat_chain.invoke({
            "user_message": user_message,
            "history_string": history_string
        })

        return {
            "intent": "casual",
            "current_step" : "chitchat_response",
            "answer": response_content,
            "messages": state['messages'] + [{'role':'ai', 'content': response_content}]
        }
    except Exception as e:
        logger.error(f"Chichat: Error in LLM Call: {e}", exc_info=True)
        error_message = "I'm having trouble with general conversation right now"
        return {
            "intent": "casual",
            "current_step" : "chitchat_response",
            "answer": error_message,
            "messages": state['messages'] + [{'role':'ai', 'content': error_message}]
        }
    

# Smart Listener Node
def smart_listener_node(state: AgentState) -> dict:
    """Listens to Patients and let them vent out their emotions"""
    logger.info("---Smart Listener Node---")
    user_message = state['messages'][-1]['content']
    severity_level = state.get('severity_level', 'low')

    if not llm:
        logger.error("Support Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step': 'llm_unavailable', 'messages': state['messages'] + [{'role': 'ai', 'content': response_content}], 'intent': 'support'}

    formatted_messages = []
    for msg in state.get('messages', [])[-8:]:
        if msg.get('role') == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
        else:
            formatted_messages.append(f"Human: {msg.get('content', '')}")
    history_string = "\n".join(formatted_messages)

    conversation_prompt = PromptTemplate(
        template="""
            You are an empathetic and intelligent AI Therapist tasked with helping men with anxiety, depression, or low motivation. 
            Your goal is to **listen actively, understand their feelings, and guide them to insights about their situation**. You Do NOT force solutions immediately.

            ### Requirements:
                1. Listen and reflect — acknowledge what the user shares, show empathy.
                2. Ask open-ended questions to help them explore their feelings or situation.
                3. React like a caring friend or therapist — show genuine interest.
                4. If appropriate, suggest very small activities to relieve stress (e.g., "take a short walk", "drink some water", "breathe slowly"), especially for anxiety or low mood.
                5. Avoid giving full solutions right away unless the user remains stuck after multiple exchanges.
                6. Keep your tone supportive, patient, and encouraging.
                7. Avoid giving full solutions immediately. Only provide micro-guidance or gentle advice.
                8. At the end, include a short guideline: "If you need a direct solution or advice, you can tell me."

            ### Constraints:
                1. Keep your response clear, friendly, and within 100-200 characters.
                2. Minimal use of Markdown.
                3. Avoid complex terms or technical language.

            Conversation context so far:
            {history_string}

            Current user message:
            "{user_message}"

            Severity Level: {severity_level}

            Generate your response in a warm, conversational, and supportive tone, engaging the user naturally while following the above rules.
        """,
        input_variables=['user_message', 'severity_level', 'history_string']
    )

    try:
        conversation_chain = conversation_prompt | llm | str_parser
        response_content = conversation_chain.invoke({
            'user_message': user_message,
            'history_string': history_string,
            'severity_level': severity_level
        })

        return {
            'intent': 'support',
            'current_step': 'support_response_successful',
            'answer': response_content,
            'messages': state['messages'] + [{'role':'ai', 'content':response_content}]
        }
    except Exception as e:
        logger.error(f"Support Node: LLM call failed: {e}", exc_info=True)
        error_message = "I'm having trouble generating a response right now, but I care about what you're going through."
        return {
            "intent": "support",
            "current_step": "support_response",
            "answer": error_message,
            'messages': state['messages'] + [{'role':'ai', 'content':error_message}]
        }

    
# Solution Node
def solution_node(state: AgentState) -> dict:
    """Helps Patient deal with situation"""
    logger.info("---Solution Node---")
    user_message = state['messages'][-1]['content']
    severity_level = state.get('severity_level', 'low')

    if not llm:
        logger.error("Solution Node: LLM is not initialized")
        response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
        return {'answer': response_content, 'current_step': 'llm_unavailable', 'messages': state['messages'] + [{'role': 'ai', 'content': response_content}], 'intent': 'solution'}

    formatted_messages = []
    for msg in state.get('messages', [])[-8:]:
        if msg.get('role') == 'ai':
            formatted_messages.append(f"AI: {msg.get('content', '')}")
        else:
            formatted_messages.append(f"Human: {msg.get('content', '')}")
    history_string = "\n".join(formatted_messages)

    solution_prompt = PromptTemplate(
        template="""
            You are a highly empathetic therapist AI helping men manage depression, anxiety, and stay motivated.
            Consider the user's current severity level, their message, and past conversation history. Respond with warmth, support, and actionable suggestions.

            Severity Level: {severity_level}
            Conversation History:
            {history_string}

            Current User Message:
            "{user_message}"

            Guidelines:
            1. For depression:
                - Low: offer encouragement and coping tips.
                - Medium: suggest supportive routines, mindfulness, and gentle guidance.
                - High: strongly recommend professional help and daily mental health support.
            2. For anxiety:
                - Low: offer relaxation tips, grounding exercises, and reassurance.
                - Medium: suggest structured calming techniques and emotional support.
                - High: emphasize immediate support and professional help if anxiety is overwhelming.
            3. For motivation:
                - Always offer actionable steps, encouragement, and empathy.
            4. Reference previous conversation context if relevant.
            5. Respond in a friendly, understanding, and caring tone.

            Respond only in natural, human-like language suitable for a supportive conversation.
        """,
        input_variables=['user_message', 'history_string', 'severity_level']
    )

    try:
        solution_chain = solution_prompt | llm | str_parser
        response_content = solution_chain.invoke({
            'user_message': user_message,
            'history_string': history_string,
            'severity_level': severity_level
        })

        return {
            'intent': 'solution',
            'current_step': 'solution_response_successful',
            'answer': response_content,
            'messages': state['messages'] + [{'role':'ai', 'content':response_content}]
        }
    except Exception as e:
        logger.error(f"Solution Node: LLM call failed: {e}", exc_info=True)
        error_message = "I'm having trouble generating a response right now, but I care about what you're going through."
        return {
            "intent": "solution",
            "current_step": "solution_response",
            "answer": error_message,
            'messages': state['messages'] + [{'role':'ai', 'content':error_message}]
        }

# Crisis Node
def crisis_node(state: AgentState) -> dict: 
    """Handles crisis situation"""
    logger.info("---Crisis Node---")
    user_message = state['messages'][-1]['content']
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
        'intent': 'crisis',
        'severity_level': severity_level,
        'answer': response_content,
        'current_step': 'crisis_situation',
        # *** FIX ***: Corrected 'messasges' to 'messages'
        'messages': state['messages'] + [{'role':'ai', 'content':response_content}]
    }

# Irrelevant Node
def irrelevant_node(state: AgentState) -> dict:
    """Handles irrelevant user queries."""
    logger.info("---Irrelevant Node---")
    user_message = state['messages'][-1]['content']
    response_content = "Sorry, I don't think this message is related to your condition, so I can't help with that"
    return {
        'intent': 'irrelevant',
        'current_step': 'irrelevant',
        'answer': response_content,
        'messages': state['messages'] + [{'role': 'ai', 'content': response_content}]
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
            "support": "smart_listener",
            "solution": "solution",
            "irrelevant": "irrelevant"
        }
    )

    builder.add_edge('chitchat', END)
    builder.add_edge('crisis', END)
    # *** FIX ***: Changed from 'solution' to 'END' for natural conversation flow
    builder.add_edge('smart_listener', END) 
    builder.add_edge('solution', END)
    builder.add_edge('irrelevant', END)

    graph = builder.compile()
    # graph.get_graph().print_ascii() # You can uncomment this to debug the graph structure
    return graph