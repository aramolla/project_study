from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="상명대학교 Chatbot", 
    page_icon="./수뭉1.png",  # 이미지 파일 경로
    layout="wide"
)
st.title("AI 수뭉 💬")

# CSS to inject contained in a string
custom_css = """
            <style>
            /* Hide Streamlit style elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            /* General page background */
            body {
                background-color: #f5f5f5;
                color: #333333;
            }

            /* Sidebar customization */
            .css-1d391kg {
                background-color: #ffffff !important;
                border-right: 1px solid #e0e0e0;
            }

            /* Input field styling */
            .stTextInput>div>div>div>input {
                padding: 10px 20px;
                border-radius: 10px;
                border: 1px solid #ccc;
                background-color: #ffffff;
                color: #333333;
            }

            /* Button styling */
            .stButton>button {
                border-radius: 10px;
                padding: 10px 20px;
                border: none;
                background-color: #007bff;
                color: #ffffff;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #0056b3;
                color: #ffffff;
            }

            /* Chat message styling */
            .stChatMessage {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
                color: #000000; /* 텍스트 색상을 검정으로 변경 */
                margin: 10px 0;
                max-width: 60%; /* 가로 공간의 60%만 차지 */
                display: flex;
                align-items: center;
            }

            .stChatMessage-user {
                background-color: #e9ecef;
                color: #000000; /* 텍스트 색상을 검정으로 변경 */
                text-align: right; /* 텍스트 오른쪽 정렬 */
                margin-left: auto; /* 왼쪽 마진을 자동으로 설정하여 오른쪽 정렬 */
                justify-content: flex-end; /* 콘텐츠를 오른쪽 정렬 */
            }

            .stChatMessage-assistant {
                background-color: #f8f9fa;
                color: #000000; /* 텍스트 색상을 검정으로 변경 */
                text-align: left; /* 텍스트 왼쪽 정렬 */
                margin-right: auto; /* 오른쪽 마진을 자동으로 설정하여 왼쪽 정렬 */
                justify-content: flex-start; /* 콘텐츠를 왼쪽 정렬 */
            }


            </style>
            """
st.markdown(custom_css, unsafe_allow_html=True)

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 페이지가 세로고침이 일어나도 이전 메시지가 쌓이도록 데이터 보관
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화기록을 저장하는 Store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

menu = st.sidebar.selectbox("메뉴", ["메인 페이지", "주간 메뉴"])

if menu == "메인 페이지":
    pass
elif menu == "주간 메뉴":
    if st.button("이미지 팝업"):
        with st.modal("이미지 팝업"):
            st.image("./menu.jpg", width=200)
            
with st.sidebar:    
    st.image("./수뭉_2.png", width=200)

    session_id = st.text_input("Session ID", value="ara123")
    clear_btn = st.button("초기화")
    if clear_btn:
        st.session_state["messages"] = []
        if session_id in st.session_state["store"]:
            del st.session_state["store"][session_id]  # 특정 세션 ID에 대한 기록만 삭제
        st.experimental_rerun()


from utils import print_messages, StreamHandler

print_messages()  # 이전 대화내용 출력

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        st.session_state["store"][session_id] = ChatMessageHistory()  # 새로운 ChatMessageHistory 객체를 생성하여 Store에 저장
    return st.session_state["store"][session_id]  # 해당 세션 ID에 대한 세션 기록 반환

from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSequence, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

if user_input := st.chat_input("메시지를 입력해 주세요."):
    st.session_state["messages"].append(("user", user_input))
    st.chat_message("user").write(f"{user_input}")

    with st.chat_message("assistant"):
        container = st.empty()
        stream_handler = StreamHandler(container)

        # 모델 생성  gpt-3.5-turbo-1106 gpt-4o-2024-05-13
        GPT_4o = ChatOpenAI(model="gpt-4o-2024-05-13", streaming=True, callbacks=[stream_handler])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # 벡터 저장소 로드
        def load_vector_store(path):
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

        vector_store_path = "./merged_vector_store"
        db = load_vector_store(vector_store_path)

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 50}
        )







        # from langchain.schema import Document
        # from langchain_community.vectorstores import Chroma
        # from langchain.retrievers.self_query.base import SelfQueryRetriever

        # document_content_description = "학교 홈페이지 정보"

        # retriever = SelfQueryRetriever.from_llm(
        #     GPT_4o,
        #     db,
        #     document_content_description,
        #     metadata_field_info,
        #     verbose = True
        # )








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
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                    """
                    역할: 너는 반말로 답변하는 상명대학교 학생들의 학교 생활을 도와줄 도우미야. 상명대학교를 언급할 때는 반드시 "우리 학교"라고 지칭해줘.
                    이름: 너의 이름은 '수뭉'이야.
                    사용자 호칭: 너가 user를 부를 때는 '슴우'라고 불러야 돼.
                    """
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", prompt_template.format(context="{context}", question="{question}")),
            ]
        )

        # LLM 체인 생성
        llm_chain = prompt | GPT_4o

        # rag_chain 정의 수정
        rag_chain = RunnableSequence(
            RunnableLambda(lambda x: {
                "context": [doc.page_content for doc in retriever.get_relevant_documents(x["question"])],
                "question": x["question"],
                "history": x["history"]  # 추가된 부분
            }),
            llm_chain
        )

        chain_with_memory = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: get_session_history(session_id),
            input_messages_key="question",
            history_messages_key="history",
        )

        # chain_with_memory.invoke 호출 부분 수정
        response = chain_with_memory.invoke(
            {"question": user_input, "history": []},  # history 추가
            config={'configurable': {'session_id': session_id}}
        )


        # for i in response['context']:
        #     print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")


        msg = response.content  # 속성으로 접근

        # st.write(msg)
        st.session_state["messages"].append(("assistant", msg))

        
     
