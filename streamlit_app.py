from PIL import Image
import pytesseract
from langchain.schema import Document
import streamlit as st
import time
import os
from core import (
    load_environment_variables,
    load_and_split_documents,
    create_vector_store,
    get_openai_llm,
    create_qa_chain,
    ask_question
)
import shutil

def load_image_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return [Document(page_content=text)]

st.set_page_config(page_title="Image2Answer", page_icon="🖼️", layout="wide")
st.title("🖼️ Image2Answer - Visual Insight QA")
st.markdown("Transform visual content into insights — ask, analyze, and get accurate answers in seconds.")

# st.set_page_config(
#     page_title="Image2Answer",
#     page_icon="./images/image.png",
#     layout="centered"
# )

st.markdown("""
<style>
.message-box {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 70%;
    word-break: break-word;
}
.user {
    background-color: #DCF8C6;
    align-self: flex-end;
    color: #000;
    margin-left: auto;
}
.assistant {
    background-color: #E8EAF6;
    align-self: flex-start;
    color: #000;
    margin-right: auto;
}
.message-container {
    display: flex;
    flex-direction: column;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Image2Answer: I'm just waiting for your questions !"
    }]

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# with st.sidebar:
#     st.image("images/image.png", width=250)
#     st.title("Document Upload")
#     st.divider()
with st.sidebar:
    st.image("images/image.png", width=250)
    st.markdown("##  🔍 Welcome to Image2Answer")
    st.markdown("Upload a *text* or *image* file to start.")
    st.markdown("---")
    st.markdown("### Supported formats:")
    st.markdown("- 📄 PDF")
    st.markdown("- 🖼️ PNG / JPG / JPEG")
    st.markdown("---")
    uploaded_file = st.file_uploader("Choose a PDF, PNG or JPG file", type=["pdf", "png", "jpg"])
    
    fileName = None
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        fileName = "tmp"+file_extension
        if file_extension == ".pdf":
            with open(fileName, "wb") as f:
                f.write(uploaded_file.getbuffer())
            fileName = "tmp.pdf"
        if file_extension == ".png":
            with open(fileName, "wb") as f:
                f.write(uploaded_file.getbuffer())
            fileName = "tmp.png"
        if file_extension == ".jpg":
            with open(fileName, "wb") as f:
                f.write(uploaded_file.getbuffer())
            fileName = "tmp.jpg"
        
        if file_extension == ".pdf":
            if st.button("Process Document"):
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                
                with st.spinner("Processing document..."):
                    try:
                        openai_api_key, openai_api_base, model_name = load_environment_variables()
                        embedding_model = "text-embedding-3-small"
                        
                        chunks = load_and_split_documents(fileName)
                        vectorstore = create_vector_store(chunks, embedding_model, openai_api_key)
                        llm = get_openai_llm(model_name, openai_api_key, openai_api_base)
                        qa_chain = create_qa_chain(llm, vectorstore)
                        
                        st.session_state.uploaded_file = uploaded_file.name
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        
                        st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"I've read *{uploaded_file.name}*. You can now ask questions about it!"
                        })
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            if st.button("Process Document"):
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                with st.spinner("Processing document..."):
                    try:
                        openai_api_key, openai_api_base, model_name = load_environment_variables()
                        embedding_model = "text-embedding-3-small"
                        
                        chunks = load_image_text(fileName)
                        print (chunks)
                        vectorstore = create_vector_store(chunks, embedding_model)
                        print (vectorstore)
                        llm = get_openai_llm(model_name, openai_api_key, openai_api_base)
                        qa_chain = create_qa_chain(llm, vectorstore)
                        
                        st.session_state.uploaded_file = uploaded_file.name
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        
                        st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"I've processed *{uploaded_file.name}*. You can now ask questions about it!"
                        })
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            
        else:
            st.error("Unsupported file type. Please upload a PDF file.")

    if st.session_state.uploaded_file:
        st.info(f"Current document: {st.session_state.uploaded_file}")
    
    st.caption("⚡ Powered by LangChain + OpenAI")
    
# st.title("Image2Answer")
# st.title("🖼️ Image2Answer - Visual Insight QA")

# st.markdown('<div class="message-container">', unsafe_allow_html=True)
# for message in st.session_state.messages:
#     role_class = "assistant" if message["role"] == "assistant" else "user"
#     st.markdown(
#         f'<div class="message-box {role_class}">{message["content"]}</div>',
#         unsafe_allow_html=True
#     )
# st.markdown('</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])


def pipeline(prompt):
    if st.session_state.qa_chain is None:
        return "Please upload and process a document first."
    
    response = ask_question(st.session_state.qa_chain, prompt)
    return response["result"]

def pipeline_stream(prompt):
    if st.session_state.qa_chain is None:
        yield "Please upload and process a document first."
    else:
        for chunk in ask_question(st.session_state.qa_chain, prompt)["result"]:
            yield chunk

if prompt := st.chat_input("Ask about the uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Affiche la question de l'utilisateur dans une bulle personnalisée
    st.markdown(
        f'<div class="message-box user">{prompt}</div>',
        unsafe_allow_html=True
    )

    # Affiche la réponse de l'assistant dans une bulle personnalisée (streaming)
    with st.spinner("Searching knowledge base..."):
        collected_response = [""]

        def stream_and_collect():
            for chunk in pipeline_stream(prompt):
                collected_response[0] += chunk
                time.sleep(0.01)
                yield chunk

        # Utilise st.write_stream pour afficher la réponse en streaming dans une bulle assistant
        response_placeholder = st.empty()
        response_chunks = []

        for chunk in stream_and_collect():
            response_chunks.append(chunk)
            response_placeholder.markdown(
                f'<div class="message-box assistant">{"".join(response_chunks)}</div>',
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": collected_response[0]
    })

# if prompt := st.chat_input("Ask about the uploaded document..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     with st.chat_message("user", avatar="😃"):
#         st.write(prompt)

#     with st.chat_message("assistant", avatar="🤓"):
#         with st.spinner("Searching knowledge base..."):
#             collected_response = [""]

#             def stream_and_collect():
#                 for chunk in pipeline_stream(prompt):
#                     collected_response[0] += chunk
#                     time.sleep(0.01)
#                     yield chunk

#             st.write_stream(stream_and_collect())

#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": collected_response[0]
#         })

if fileName is not None:
    if os.path.exists(fileName):
        try:
            os.remove(fileName)
        except:
            pass