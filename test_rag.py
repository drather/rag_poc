import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # Import PromptTemplate

# RAG 파이프라인 함수 (rag_poc.py와 동일하게 LangChain 기반으로 변경)
def rag_pipeline(qa_chain, user_query, qa_chain_prompt):
    """전체 RAG 파이프라인 (LangChain 기반)"""
    try:
        result = qa_chain.invoke({"query": user_query})
        llm_response = result.get('result', '').strip()
        source_documents = result.get('source_documents', [])
        
        # --- 프롬프트 디버그 출력 ---
        context_for_prompt = "\n\n".join([doc.page_content for doc in source_documents])
        formatted_prompt = qa_chain_prompt.format(context=context_for_prompt, question=user_query)
        print("\n--- LLM에 전달될 최종 프롬프트 ---")
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

# LLM 자체의 추론 능력 테스트 함수
def test_direct_llm_reasoning(llm_instance):
    print("\n\n=== LLM 직접 추론 능력 테스트 ===")
    direct_query = "다음 모의해킹 이력을 보고, 가장 최근 모의해킹 완료일자를 말해줘. 20250801, 20240701, 20241101, 20250101"
    direct_prompt = f"""주어진 정보에서 가장 최근 날짜를 찾아 답변하세요. 다른 설명은 포함하지 마세요.\n\n정보: {direct_query}\n답변:"""
    
    print(f"질문: {direct_query}")
    print("--- LLM에 전달될 직접 프롬프트 ---")
    print(direct_prompt)
    print("-----------------------------------\n")

    try:
        direct_response = llm_instance.invoke(direct_prompt).strip()
        print("--- LLM 직접 답변 ---")
        print(direct_response)
    except Exception as e:
        print(f"LLM 직접 답변 생성 중 오류가 발생했습니다: {e}")
    print("===================================")

def main():
    # --- 모델 및 DB 로드 ---
    print("모델과 데이터베이스를 로드합니다...")

    # 임베딩 모델 로드
    embedding_model = SentenceTransformerEmbeddings(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # ChromaDB 로드
    chroma_db_path = "D:\\workspace\\rag_poc\\chroma_db"
    print(f"ChromaDB 로드 경로: {chroma_db_path}")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"DB에서 로드된 문서 개수: {vectorstore._collection.count()}")

    # LLM 로드
    llm_model_id = "beomi/KoAlpaca-Polyglot-5.8B"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

    llm_pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=64,
        temperature=0.1,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=llm_pipe)

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

    # --- RAG 테스트 실행 ---
    test_queries = [
        "AA 시스템의 주요 구성 요소는 무엇인가요?",
        "SQL Injection 취약점은 어떻게 조치되었나요?",
        "Kafka 클러스터는 언제 확장되었나요?",
        "B법인 API 연동 시 인증 방식은 무엇인가요?",
        "AI 기반 챗봇 시스템 개발 일정은 어떻게 되나요?",
        "주문 처리 시스템의 재고 차감 방식이 어떻게 변경되었나요?",
        "모의해킹 조치완료일이 언제야?",
        "AA 시스템의 DB 버전은 무엇인가요?",
        "시스템 개요가 뭐야?"
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        response_text = rag_pipeline(qa_chain, query, QA_CHAIN_PROMPT) # QA_CHAIN_PROMPT 전달
        print("--- 답변 ---")
        print(response_text)
        print("="*50)

    # --- LLM 직접 추론 능력 테스트 실행 ---
    test_direct_llm_reasoning(llm)

if __name__ == "__main__":
    main()