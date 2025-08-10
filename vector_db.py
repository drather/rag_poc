import pickle

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# 이전 단계에서 저장한 데이터 로드
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    chunk_metadata = data['metadata']

# ChromaDB 경로 설정
chroma_db_path = "D:\\workspace\\rag_poc\\chroma_db"
print(f"ChromaDB 경로: {chroma_db_path}")

# 임베딩 모델 로드
embedding_model = SentenceTransformerEmbeddings(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Document 객체 생성
documents = []
for i, chunk in enumerate(chunks):
    doc_metadata = chunk_metadata[i]
    documents.append(Document(page_content=chunk, metadata=doc_metadata))

# ChromaDB 생성 및 영속화
# 기존 DB가 있다면 삭제하고 새로 생성하는 로직은 Chroma.from_documents가 내부적으로 처리하지 않으므로,
# 외부에서 디렉토리를 삭제하는 것이 가장 확실합니다.
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=chroma_db_path
)

# 영속화된 데이터 확인
vectorstore.persist()
print(f"ChromaDB에 {vectorstore._collection.count()}개의 문서 조각이 인덱싱되었습니다. (추가 후)")
