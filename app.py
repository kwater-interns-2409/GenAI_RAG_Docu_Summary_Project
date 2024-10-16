import streamlit as st
from rag_functions import load_docs, create_vectorstore, create_rag_chain, create_github_rag_chain
import time



st.title("RAG Q&A 시스템")
st.sidebar.title("소개")
st.sidebar.info(
    "이 앱은 RAG(검색 증강 생성) 시스템을 시연합니다. "
)
on=st.sidebar.toggle("RAG 기능 키기", True)
openai_on=st.sidebar.toggle("OpenAI 사용", True)
# 데이터베이스에 문서 추가
files=st.sidebar.file_uploader("RAG에 추가할 문서를 넣어주세요.", type=["txt", "pdf", "doc"], accept_multiple_files=True)
# 사용자 입력
question = st.text_input("질문하세요:")

if question:
    if st.button("답변 받기"):
        with st.spinner("처리 중..."):
            start_time=time.time()
            # 문서 로드 및 분할
            splits = load_docs(files)
            
            # 벡터 저장소 생성
            vectorstore = create_vectorstore(splits)
            
            # RAG 체인 생성
            if openai_on:
                result = create_github_rag_chain(vectorstore, question, on)
            else :
                result = create_rag_chain(vectorstore, question, on)

            end_time=time.time()
            print("걸린 시간: ", end="")
            print(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time)))
            
            st.subheader("답변:")
            st.write(result)
