import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from utils import print_messages, StreamHandler


# 페이지 설정
st.set_page_config(
    page_title="상명대학교 Chatbot", 
    page_icon="./수뭉1.png",  # 이미지 파일 경로
    layout="wide"
)

st.title("AI 수뭉 💬")
st.markdown(custom_css, unsafe_allow_html=True)

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 페이지가 세로고침이 일어나도 이전 메시지가 쌓이도록 데이터 보관
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# 채팅 대화기록을 저장하는 Store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "show_image" not in st.session_state:
    st.session_state["show_image"] = False

# 사이드바
with st.sidebar:
    st.image("./수뭉_2.png", width=200)
    session_id = st.text_input("Session ID", value="ara123")
    clear_btn = st.button("초기화")
    if clear_btn:
        st.session_state["messages"] = []
        if session_id in st.session_state["store"]:
            del st.session_state["store"][session_id] # 특정 세션 ID에 대한 기록만 삭제
        st.experimental_rerun()

    if st.button("주간 메뉴"):
        st.session_state["show_image"] = not st.session_state["show_image"]

if st.session_state["show_image"]:
    st.image("./menu.jpg", width=800)

# 이전 대화내용 출력
print_messages()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory() # 새로운 ChatMessageHistory 객체를 생성하여 Store에 저장
    return st.session_state["store"][session_id] # 해당 세션 ID에 대한 세션 기록 반환

# 벡터 저장소 로드 함수
def load_vector_store(path):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# 채팅 입력 처리
if user_input := st.chat_input("메시지를 입력해 주세요."):
    # 사용자 입력 기록 저장
    st.session_state["messages"].append(("user", user_input))
    st.chat_message("user").write(f"{user_input}")

    with st.chat_message("assistant"):
        container = st.empty()
        stream_handler = StreamHandler(container)

        # GPT 모델 설정 gpt-3.5-turbo-1106 gpt-4o-2024-05-13
        GPT = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True, callbacks=[stream_handler])

        # 벡터 저장소 로드
        vector_store_path = "./merged_vector_store"
        db = load_vector_store(vector_store_path)

        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 50})

        # 프롬프트 템플릿 설정
        prompt_template = """
        
        ### [INST]
        AI수뭉입니다! 궁금한걸 물어보쇼
        
        {context}
        
        ### 질문:
        {question}
        
        [/INST]
        답변!
        
        """

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            
                역할: 너는 반말로 답변하는 상명대학교 학생들의 학교 생활을 도와줄 도우미야. 
                상명대학교를 언급할 때는 반드시 "우리 학교"라고 지칭해줘. 
                
                이름: 너의 이름은 '수뭉'이야. 
                
                사용자 호칭: 너가 user를 부를 때는 '슴우'라고 불러야 돼.
                
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", prompt_template.format(context="{context}", question="{question}")),
        ])

        # LLM 체인 생성
        llm_chain = prompt | GPT

        # Multi-Query 및 Time-Weighted 기법 통합
        def multi_query_and_time_weighted(x):
            context_docs = retriever.get_relevant_documents(x["question"])
            # 최근 문서 기준으로 우선 처리 (Time-Weighted)
            sorted_docs = sorted(context_docs, key=lambda doc: doc.metadata['timestamp'], reverse=True)
            return {
                "context": [doc.page_content for doc in sorted_docs],
                "question": x["question"],
                "history": x["history"]
            }

        # RAG 체인 만들기
        rag_chain = RunnableSequence(
            RunnableLambda(multi_query_and_time_weighted),
            llm_chain
        )

        # 세션 기록을 포함한 체인 설정
        chain_with_memory = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: get_session_history(session_id),
            input_messages_key="question",
            history_messages_key="history",
        )

        # 응답 생성 및 출력
        response = chain_with_memory.invoke(
            {"question": user_input, "history": []},
            config={'configurable': {'session_id': session_id}}
        )

        msg = response.content  # 응답 내용

        st.session_state["messages"].append(("assistant", msg))
        st.chat_message("assistant").write(msg)
