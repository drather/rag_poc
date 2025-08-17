import chromadb
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import argparse

# --- FastAPI & Uvicorn (API 모드용) ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# =======================================================================
# 1. 모델 및 QA 체인 로드 (공통 로직)
# 모드에 상관없이 시작 시 한 번만 로드되도록 전역으로 이동
# =======================================================================
print("모델과 데이터베이스를 로드합니다...")

embedding_model = SentenceTransformerEmbeddings(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')
vectorstore = Chroma(
    persist_directory="D:/workspace/rag_poc/chroma_db", 
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama2", temperature=0.1, verbose=True)

template = """주어진 참고 문서만을 사용하여 다음 질문에 답변하세요. 참고 문서에 없는 내용은 절대로 지어내지 마세요. 만약 참고 문서에서 답변을 찾을 수 없다면, '참고 문서에서 답변을 찾을 수 없습니다.'라고 답변하세요.\n\n{context}\n\n질문: {question}\n답변:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

print("로드 완료.")

def rag_pipeline(user_query: str):
    """RAG 파이프라인 실행 함수 (기존 코드와 동일, qa_chain을 인자로 받지 않음)"""
    try:
        result = qa_chain.invoke({"query": user_query})
        llm_response = result.get('result', '').strip()
        source_documents = result.get('source_documents', [])
        source_info = ", ".join(list(set([doc.metadata.get('source', 'Unknown') for doc in source_documents])))
        
        if not llm_response.strip() or "참고 문서에서 답변을 찾을 수 없습니다." in llm_response:
            llm_response = "참고 문서에서 답변을 찾을 수 없습니다."
        
        final_answer = f"{llm_response}\n\n참고한 문서: {source_info}"
        return final_answer
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

# =======================================================================
# 2. 실행 모드별 함수 정의
# =======================================================================

# --- API 모드 ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag_pipeline(request.query)
    return {"answer": answer}

def run_api_mode():
    """FastAPI 서버를 실행합니다."""
    uvicorn.run(app, host="0.0.0.0", port=8008)

# --- CLI 모드 ---
def run_cli_mode():
    """대화형 CLI를 실행합니다."""
    while True:
        try:
            test_query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
            if test_query.lower() == 'exit':
                break
            print(f"질문: {test_query}")
            response_text = rag_pipeline(test_query)
            print("--- 답변 ---")
            print(response_text)
            print("\n" + "="*50 + "\n")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break

# =======================================================================
# 3. 메인 실행 블록: 모드에 따라 분기
# =======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG PoC 실행 모드 선택")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cli", "api"],
        default="cli",
        help="실행 모드: 'cli' (대화형) 또는 'api' (서버)"
    )
    args = parser.parse_args()

    if args.mode == "api":
        print("API 모드로 실행합니다. (http://0.0.0.0:8008)")
        run_api_mode()
    else:
        print("대화형 CLI 모드로 실행합니다.")
        run_cli_mode()
