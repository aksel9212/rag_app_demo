import os
import numpy as np
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete, hf_embed
#from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from sentence_transformers import SentenceTransformer
from groq import Groq
import google.generativeai as genai

from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import streamlit as st
import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Load environment variables from .env file
#dotenv.load_dotenv()

WORKING_DIR = "rag-data"
DOCS_DIR = WORKING_DIR + "/docs"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# Groq LLM function (you might need to adapt based on your Groq client setup)
groq_api_key = st.secrets.groq_api_key #os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
gemini_api_key = st.secrets.google_api_key #os.getenv('GOOGLE_API_KEY') 
st.info(gemini_api_key)
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

async def llm_model_func_google(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = model.generate_content(combined_prompt)

    # 4. Return the response text
    return response.text


async def llm_model_func(prompt,**kwargs) -> str:
    print(kwargs)
    combined_prompt = ""
    
    if 'system_prompt' in kwargs:
        combined_prompt += kwargs['system_prompt'] + f"\n"

        
    if 'history_messages' in kwargs:
        for msg in kwargs['history_messages']:
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
        
    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"


    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or another Groq-supported model like llama3-70b
        messages=[{"role": "user", "content": combined_prompt}],
    )
    return response.choices[0].message.content

async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

CHROMADB_LOCAL_PATH = os.environ.get(
    "CHROMADB_LOCAL_PATH", os.path.join(WORKING_DIR, "chroma_data")
)
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func_google,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
        vector_storage="ChromaVectorDBStorage",
        vector_db_storage_cls_kwargs={
                "local_path": CHROMADB_LOCAL_PATH,
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def run_new_indexing():
    documents = os.listdir(DOCS_DIR)
    docs_data = []
    for doc in documents:
        print("index doc: ", doc)
        with open(DOCS_DIR+"/"+doc,'r') as f:
            docs_data.append(f.read())
        if len(docs_data) == 10:
            st.session_state.rag.insert(docs_data)
            docs_data = []
    print(docs_data)
    st.session_state.rag.insert(docs_data)
    print("documents indexed!!")

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)             



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def main():
    st.title("RAG AI Agent (demo)")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"): 
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])    
        
    with st.sidebar:
        if st.button("Update Index"):
            run_new_indexing()
        if st.button("Exit"):
            st.write("Good Bye...!")
            st.stop()      
    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({'role':'user','content':user_input})
        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Properly consume the async generator with async for
            response = st.session_state.rag.query(
                        query=user_input,
                        param=QueryParam(mode="local", top_k=1, response_type="single line"),
            )
            
            # Final response without the cursor
            response = response.replace("###",'')
            st.markdown(response)
            st.session_state.messages.append({'role':'assistant','content':response})
            print(response)


if __name__ == "__main__":
    if "rag" not in st.session_state:
        st.session_state.rag = asyncio.run(initialize_rag())
        print("rag started success!!!")
        print("run indexing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        run_new_indexing()
    main()