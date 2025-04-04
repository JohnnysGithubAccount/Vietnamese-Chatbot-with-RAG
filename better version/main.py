import streamlit as st
from streamlit_chat import message
from utils import load_data, split_documents, initialize_chain, conversational_chat
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os


def main():
    st.markdown("<h1 style='text-align: center;'>Vietnamese Q&A Chatbot</h1>",
                unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Document Research Assistance!</h3>",
                unsafe_allow_html=True)

    # Create sidebar for user inputs
    st.sidebar.title('Settings')
    option = st.sidebar.selectbox('Select Data Source', ('Web URL', 'Upload Document'))

    data = []
    if option == 'Web URL':
        url_input = st.sidebar.text_input("Enter URLs (separate by commas)", key='url_input')
        if st.sidebar.button("Load Data"):
            urls_list = [url.strip() for url in url_input.split(",")]
            data = load_data(urls_list)
    else:
        file_upload = st.sidebar.file_uploader("Upload PDF", type='pdf')
        if file_upload:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_upload.read())
                tmp_file_path = tmp_file.name

            # Load the PDF
            loader = PyPDFLoader(tmp_file_path)
            data = loader.load()

            # Clean up
            os.unlink(tmp_file_path)

    docs = split_documents(data)

    temperature = st.sidebar.slider('Creativity (Temperature)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    k_value = st.sidebar.select_slider('Number of Relevant Documents (K) ', options=[i for i in range(10)], value=3)

    qa = initialize_chain(docs, temperature, k_value)

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [
            "Xin chào! Tôi là chatbot riêng của bạn được xây dựng từ LLM được lấy từ HuggingFace🤗. "
            "Tôi có thể giúp bạn tìm hiểu được tài liệu của bạn"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Xin chào ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Enter question", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(qa, user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    main()
