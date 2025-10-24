# class AgentState(TypedDict):
#     user_id: Optional[str]
#     user_message: Optional[str]
#     answer: Optional[str]
#     intent: Optional[str]
#     detected_emotion: Optional[str]
#     severity_level: Optional[str]  # low, medium or high
#     messages: List[Dict[str,str]]
#     motivation_score: Optional[float]
#     current_step = Optional[str]



# ----------------------------------

# # Solution Node
# def solution_node(state: AgentState) -> dict:
#     """Helps Patient deal with situation"""
#     logger.info("---Support Node---")
#     severity_level = state.get('severity_level', 'low')

#     if not llm:
#         logger.error("Support Node: LLM is not initialized")
#         response_content = "I'm experiencing technical difficulties and cannot chat right now. Please try again later."
#         return {'answer': response_content, 'current_step': 'llm_unavailable', 'messages':[{'role': 'ai', 'content': response_content}]}

#     user_message = state.get("user_message", "").strip()
#     severity_level = state.get("severity_level", "low")

#     formatted_messages = []

#     for msg in state.get('messages', [])[:8]:
#         if msg.get('role') == 'ai':
#             formatted_messages.append(f"AI: {msg.get('content', '')}")
#         else:
#             formatted_messages.append(f"Human: {msg.get('content', '')}")

#     history_string = "\n".join(formatted_messages)

#     support_prompt = PromptTemplate(
#         template="""
#             You are a highly empathetic therapist AI helping men manage depression, anxiety, and stay motivated.
#             Consider the user's current severity level, their message, and past conversation history. Respond with warmth, support, and actionable suggestions.

#             Severity Level: {severity_level}
#             Conversation History:
#             {history_string}

#             Current User Message:
#             "{user_message}"

#             Guidelines:
#             1. For depression:
#                 - Low: offer encouragement and coping tips.
#                 - Medium: suggest supportive routines, mindfulness, and gentle guidance.
#                 - High: strongly recommend professional help and daily mental health support.
#             2. For anxiety:
#                 - Low: offer relaxation tips, grounding exercises, and reassurance.
#                 - Medium: suggest structured calming techniques and emotional support.
#                 - High: emphasize immediate support and professional help if anxiety is overwhelming.
#             3. For motivation:
#                 - Always offer actionable steps, encouragement, and empathy.
#             4. Reference previous conversation context if relevant.
#             5. Respond in a friendly, understanding, and caring tone.

#             Respond only in natural, human-like language suitable for a supportive conversation.
#         """,
#         input_variables=['user_message', 'history_string', 'severity_level']
#     )

#     try:
#         support_chain = support_prompt | llm | str_parser
#         response_content = support_chain.invoke({
#             'user_message': user_message,
#             'history_string': history_string,
#             'severity_level': severity_level
#         })

#         return {
#             'user_message': user_message,
#             'intent': 'support',
#             'current_step': 'support_response_successful',
#             'answer': response_content,
#             'messages': [{'role':'ai', 'content':response_content}]
#         }
#     except Exception as e:
#         logger.error(f"Support Node: LLM call failed: {e}", exc_info=True)
#         error_message = "I'm having trouble generating a response right now, but I care about what you're going through."
#         return {
#             'user_message': user_message,
#             "intent": "support",
#             "current_step": "support_response",
#             "answer": error_message,
#             'messages': [{'role':'ai', 'content':error_message}]
#         }