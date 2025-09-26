import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ SerpAPI 검색 툴 정의
def search_web():
    search = SerpAPIWrapper()
    
    def run_with_source(query: str) -> str:
        results = search.results(query)
        organic = results.get("organic_results", [])
        formatted = []
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")  # ✅ snippet 추가
            if link:
                formatted.append(f"- [{title}]({link}) ({source})\n  {snippet}")
            else:
                formatted.append(f"- {title} (출처: {source})\n  {snippet}")
        return "\n".join(formatted) if formatted else "검색 결과가 없습니다."
    
    return Tool(
        name="web_search",
        func=run_with_source,
        description="실시간 뉴스 및 웹 정보를 검색할 때 사용합니다. 결과는 제목+출처+링크+간단요약(snippet) 형태로 반환됩니다."
    )

# ✅ 파일 업로드 → 벡터DB → 검색 툴 생성 (PDF, TXT 지원)
def load_uploaded_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            all_documents.extend(documents)

        elif file_extension in ['txt', 'text']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w+', encoding='utf-8') as tmp_file:
                content = uploaded_file.read().decode('utf-8')
                tmp_file.write(content)
                tmp_file.flush()
                tmp_file_path = tmp_file.name
            loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
            all_documents.extend(documents)

    if all_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_documents)

        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        retriever = vector.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            name="document_search",
            description="업로드된 문서(PDF 및 텍스트 파일)에서 정보를 검색할 때 사용하는 도구입니다"
        )
        return retriever_tool
    return None

# ✅ Agent 대화 실행
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    return result['output']

# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# ✅ 메인 실행
def main():
    st.set_page_config(page_title="AI 비서", layout="wide", page_icon="🤖")

    with st.container():
        st.image('./chatbot_logo.png', use_container_width=True)
        st.markdown('---')
        st.title("안녕하세요! RAG를 활용한 'AI 비서 톡톡이' 입니다")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        st.session_state["SERPAPI_API"] = st.text_input("SERPAPI_API 키", placeholder="Enter Your API Key", type="password")
        st.markdown('---')

        # LLM 모델 선택
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        selected_model = st.selectbox(
            "LLM 모델 선택",
            options=model_options,
            index=0,  # gpt-4o-mini가 기본값
            key="model_selector"
        )
        st.session_state["selected_model"] = selected_model

        st.markdown('---')
        uploaded_docs = st.file_uploader("파일 업로드 (PDF, TXT)", accept_multiple_files=True, type=['pdf', 'txt'], key="file_uploader")

    # ✅ 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        if uploaded_docs:
            document_search = load_uploaded_files(uploaded_docs)
            if document_search:
                tools.append(document_search)
        tools.append(search_web())

        # LLM 설정 - 선택된 모델 사용
        selected_model = st.session_state.get("selected_model", "gpt-4o-mini")
        llm = ChatOpenAI(model_name=selected_model, temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                "Be sure to answer in Korean. You are a helpful assistant. "
                "Make sure to use the `document_search` tool for searching information from uploaded documents (PDF or text files). "
                "If you can't find the information from the uploaded documents, use the `web_search` tool for searching information from the web. "
                "If the user’s question contains words like '최신', '현재', or '오늘', you must ALWAYS use the `web_search` tool to ensure real-time information is retrieved. "
                "Please always include emojis in your responses with a friendly tone. "
                "Your name is `AI 비서 톡톡이`. Please introduce yourself at the beginning of the conversation."),
                ("placeholder", "{chat_history}"),
                ("human", "{input} \n\n Be sure to include emoji in your responses."),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )


        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 입력창
        user_input = st.chat_input('질문이 무엇인가요?')

        if user_input:
            session_id = "default_session"
            session_history = get_session_history(session_id)

            if session_history.messages:
                prev_msgs = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
                response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(prev_msgs), agent_executor)
            else:
                response = chat_with_agent(user_input, agent_executor)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

            session_history.add_message({"role": "user", "content": user_input})
            session_history.add_message({"role": "assistant", "content": response})

        print_messages()

    else:
        st.warning("OpenAI API 키와 SerpAPI API 키를 입력하세요.")

if __name__ == "__main__":
    main()
