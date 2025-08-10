
import os
import pickle
from sentence_transformers import SentenceTransformer

# 라이브러리 설치: pip install sentence-transformers

# 한국어 임베딩 모델 로드 (로컬에 자동 다운로드)
model_name = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(model_name)

chunk_dir = './chunks/'
chunks = []
chunk_metadata = []

# 텍스트 파일 읽기 및 메타데이터 저장
for filename in os.listdir(chunk_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(chunk_dir, filename), 'r', encoding='utf-8') as f:
            content = f.read()
            # 파일 내용 전체를 하나의 덩어리로 처리
            chunks.append(content)
            chunk_metadata.append({'source': filename.replace('.txt', '')})

# 문서 조각들을 벡터로 변환
print(f"총 {len(chunks)}개의 문서 조각을 임베딩합니다...")
chunk_embeddings = model.encode(chunks)
print(f"임베딩 벡터 차원: {chunk_embeddings.shape[1]}")

# 다음 단계에서 사용할 수 있도록 데이터 저장
with open('data.pkl', 'wb') as f:
    pickle.dump({'chunks': chunks, 'metadata': chunk_metadata, 'embeddings': chunk_embeddings}, f)

print("임베딩 및 데이터 저장이 완료되었습니다.")
