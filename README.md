## RAG시스템 기반 학교 챗봇
<img width="700" alt="image" src="https://github.com/user-attachments/assets/6e78b68e-c7b2-4192-b7d2-931e722a1524" />

- Selenium과 BeautifulSoup를 이용하여 학교 사이트,학교 학생회 자료 크롤링
- 에브리타임과 학교 앞 맛집 등 데이터 수집
- session별로 대화 내용을 기억하는 memory 기능 구현
- LangChain의 Multi-Query를 이용하여 질문에 대한 여러 방면의 답을 내놓아 답변의 질 향상시킴
- 공지 사항의 경우 LangChain의 Time-Weieghted 기법을 사용하여 가장 최근에 이용된 문서기준으로 먼저 참고하게끔 함
- streamlit을 이용하여 서버 구축
- 시간 스케줄러 cron을 이용하여 주기적으로 올라오는 내용을 업데이트

### 기술 아키텍쳐
<img width="600" alt="image" src="https://github.com/user-attachments/assets/a9ce0bcf-4ad1-4a4e-8d20-858f8983d7a9" />



### 기능 내용
<img width="600" alt="image" src="https://github.com/user-attachments/assets/8005db88-8f4d-4133-9dbd-231b892aa12c" />

프로젝트 기간 - 2024.03 ~ 2024.06

