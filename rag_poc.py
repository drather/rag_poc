import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. 검색기(Retriever) 설정 ---
embedding_model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'

# --- 2. 생성기(Generator) 설정 (로컬 GGUF LLM) ---
llm_model_path = "D:/workspace/rag_poc/models/koalpaca-polyglot-12.8b.Q4_K_M.gguf"

def rag_pipeline(qa_chain, user_query, qa_chain_prompt):
    """전체 RAG 파이프라인 (LangChain 기반)"""
    try:
        result = qa_chain.invoke({"query": user_query})
        llm_response = result.get('result', '').strip()
        source_documents = result.get('source_documents', [])
        
        # --- 프롬프트 디버그 출력 ---
        context_for_prompt = "\n\n".join([doc.page_content for doc in source_documents])
        formatted_prompt = qa_chain_prompt.format(context=context_for_prompt, question=user_query)
        print("\n--- LLM에 전달될 최종 프롬프트 --- ")
        print(formatted_prompt)
        print("-----------------------------------\n")
        # --- 프롬프트 디버그 출력 끝 ---

        source_info = ", ".join(list(set([doc.metadata.get('source', 'Unknown') for doc in source_documents])))

        if not llm_response.strip() or "참고 문서에서 답변을 찾을 수 없습니다." in llm_response:
            llm_response = "참고 문서에서 답변을 찾을 수 없습니다."
        
        final_answer = f"{llm_response}\n\n참고한 문서: {source_info}"
        return final_answer
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

if __name__ == "__main__":
    print("모델과 데이터베이스를 로드합니다...")

    # 임베딩 모델 로드
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # ChromaDB 로드
    chroma_db_path = "D:/workspace/rag_poc/chroma_db"
    print(f"ChromaDB 로드 경로: {chroma_db_path}")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"DB에서 로드된 문서 개수: {vectorstore._collection.count()}")

    # Ollama LLM 로드
    llm = Ollama(
        model="llama2",
        temperature=0.1,
        verbose=True,
    )

    # 사용자 정의 프롬프트 템플릿
    template = """주어진 참고 문서만을 사용하여 다음 질문에 답변하세요. 참고 문서에 없는 내용은 절대로 지어내지 마세요. 만약 참고 문서에서 답변을 찾을 수 없다면, '참고 문서에서 답변을 찾을 수 없습니다.'라고 답변하세요.\n\n{context}\n\n질문: {question}\n답변:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, # 소스 문서 반환 설정
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # 사용자 정의 프롬프트 전달
    )
    print("로드 완료.")

    while True:
        try:
            test_query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
            if test_query.lower() == 'exit':
                break
            print(f"질문: {test_query}")
            response_text = rag_pipeline(qa_chain, test_query, QA_CHAIN_PROMPT) # QA_CHAIN_PROMPT 전달
            print("--- 답변 ---")
            print(response_text)
            "\n" + "="*50 + "\n"
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break