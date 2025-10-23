import streamlit as st
import json
import bcrypt  # For password hashing
import logging
from pathlib import Path
from therapist import langgraph_agent, AgentState

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USERS_FILE = Path("users.json")
CONVERSATIONS_DIR = Path("conversations")

# Ensure conversations directory exists
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Ensure users file exists
if not USERS_FILE.exists():
    with open(USERS_FILE, 'w') as f:
        json.dump([], f)

# --- Agent Initialization ---
@st.cache_resource
def load_agent():
    """Loads and returns the compiled LangGraph agent. Cached for performance."""
    logger.info("Initializing LangGraph agent...")
    try:
        agent = langgraph_agent()
        logger.info("LangGraph agent initialized successfully.")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph agent: {e}", exc_info=True)
        st.error(f"Fatal error: Could not initialize AI agent. {e}")
        return None

# agent_graph = load_agent()  # Moved inside main

# --- User Authentication Functions ---

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed_password: str) -> bool:
    """Checks if the provided password matches the stored hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def load_users() -> dict:
    """Loads the user database from users.json."""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {} # Return empty dict if file is empty or corrupt

def save_users(users: dict):
    """Saves the user database to users.json."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def authenticate_user(username, password) -> bool:
    """Authenticates a user against the users.json file."""
    users = load_users()
    if username in users and check_password(password, users[username]):
        return True
    return False

def create_user(username, password) -> bool:
    """Creates a new user. Returns True on success, False if user exists."""
    users = load_users()
    if username in users:
        return False  # User already exists
    
    users[username] = hash_password(password)
    save_users(users)
    return True

# --- Chat History Functions ---

def get_history_path(username: str) -> Path:
    """Gets the file path for a user's conversation history."""
    return CONVERSATIONS_DIR / f"{username}.json"

def load_chat_history(username: str) -> list:
    """Loads a user's chat history from their JSON file."""
    history_file = get_history_path(username)
    if not history_file.exists():
        return []  # No history yet
    
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON for user {username}. Returning empty history.")
        return []

def save_chat_history(username: str, history: list):
    """Saves a user's chat history to their JSON file."""
    history_file = get_history_path(username)
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
    except IOError as e:
        logger.error(f"Could not save chat history for user {username}: {e}")
        st.error("Error: Could not save chat history.")

# --- Streamlit Page 1: Login / Signup ---

def show_login_page():
    """Renders the login and signup page."""
    st.title("Welcome to Gentle Space Men")
    st.write("Please log in or sign up to continue.")

    login_tab = st.tabs(["Login"])[0]

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if not username or not password:
                    st.warning("Please enter both username and password.")
                elif authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Load history into session state *after* successful login
                    st.session_state.messages = load_chat_history(username)
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

# --- Streamlit Page 2: Chat Interface ---

def show_chat_page(agent_graph):
    """Renders the main chat interface."""
    
    st.title("Gentle Space Men")
    st.write(f"Logged in as: **{st.session_state.username}**")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("How are you feeling today?"):
        if not agent_graph:
            st.error("The AI agent is not available. Please check the logs.")
            return

        # 1. Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Prepare the state for the LangGraph agent
        # We pass the *entire* message history
        state: AgentState = {
            'user_id': st.session_state.username,
            'user_message': prompt, # Kept for compatibility with router's first pass
            'messages': st.session_state.messages,
            # Clear previous run's output
            'answer': None,
            'intent': None,
            'severity_level': None,
            'current_step': None
        }
        
        # 3. Invoke the agent
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    # The agent's nodes will append the AI response to the 'messages' list
                    result = agent_graph.invoke(state)
                    
                    # Extract the final AI response from the updated 'messages' list
                    if result.get('messages') and result['messages'][-1]['role'] == 'ai':
                        ai_response = result['messages'][-1]['content']
                    else:
                        # Fallback if the agent fails to update messages
                        ai_response = result.get("answer", "I'm sorry, I couldn't process that.")
                    
                    st.markdown(ai_response)
                    
                    # 4. Update session state with the new full history from the agent
                    st.session_state.messages = result.get('messages', st.session_state.messages)
                    
                    # 5. Save the complete conversation to the user's JSON file
                    save_chat_history(st.session_state.username, st.session_state.messages)

                except Exception as e:
                    logger.error(f"Error during agent invocation for user {st.session_state.username}: {e}", exc_info=True)
                    st.error(f"An error occurred while processing your message: {e}")


# --- Main Application Router ---

def main():
    logged_in = st.session_state.get("logged_in", False)
    st.set_page_config(page_title="Gentle Space Men - Login" if not logged_in else "Gentle Space Men - Chat")

    # Initialize agent after set_page_config
    agent_graph = load_agent()

    # Initialize session state variables
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Route to the correct page
    if st.session_state.logged_in:
        show_chat_page(agent_graph)
    else:
        show_login_page()

if __name__ == "__main__":
    main()