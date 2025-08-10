# 2025년 8월 11일: 13B 모델 도입 및 Ollama 전환 성공 보고서

## 1. 프로젝트 목표
기존 RAG(Retrieval Augmented Generation) 시스템에서 사용하던 5.8B 모델의 환각(Hallucination) 현상을 개선하고, 더 정확하고 풍부한 답변을 제공하기 위해 13B 규모의 LLM(Large Language Model)을 도입하는 것을 목표로 했습니다.

## 2. 초기 시도 및 어려움 (2025년 8월 10일)

### 2.1. `llama-cpp-python` 및 로컬 GGUF 모델 로드 시도
*   **계획**: `beomi/KoAlpaca-Polyglot-12.8b`의 GGUF 버전을 `llama-cpp-python` 라이브러리를 통해 로컬에서 직접 로드하여 사용하려 했습니다.
*   **어려움**:
    *   `llama-cpp-python` 설치 과정에서 `WIN 193` 오류 (DLL 문제)가 지속적으로 발생했습니다. 이는 주로 Python 버전(3.13.6)과 `llama.dll` 간의 호환성 문제로 추정되었습니다.
    *   Visual C++ Redistributables 확인, `llama-cpp-python` 재설치, 시스템 전체 Python 환경 정리, Python 3.10으로의 다운그레이드 및 가상 환경 재구축 등 다양한 시도를 했으나 `WIN 193` 오류는 해결되지 않았습니다.
    *   `auto-gptq`를 이용한 GPTQ 모델 로드 시도 또한 `torch` 및 `CUDA_HOME` 환경 변수 문제, 모델 파일 로드 오류 등 수많은 기술적 난관에 부딪혔습니다.

### 2.2. 결론
`llama-cpp-python`을 통한 로컬 모델 로드 방식은 환경 설정의 복잡성과 지속적인 오류로 인해 현재 환경에서는 적합하지 않다고 판단했습니다.

## 3. Ollama로의 전략 전환 (2025년 8월 11일)

### 3.1. 전환 결정
기존 방식의 한계를 인식하고, LLM 모델 관리 및 실행을 간소화할 수 있는 Ollama를 활용하는 방향으로 전략을 변경했습니다. Ollama는 복잡한 환경 설정 없이 다양한 LLM을 쉽게 다운로드하고 실행할 수 있는 장점이 있습니다.

### 3.2. Ollama 모델 준비 및 연동
*   **Ollama 설치**: Ollama를 시스템에 설치했습니다.
*   **모델 Pull 시도**:
    *   처음에는 `koalpaca-13b` 또는 `eeve-korean-10.8b:latest`, `llama-2-koen-13b`와 같은 한국어 특화 모델을 `ollama pull` 명령으로 다운로드하려 했습니다.
    *   하지만 이 모델들은 Ollama의 기본 레지스트리에서 직접 `pull`할 수 없다는 오류(`pull model manifest: file does not exist`)가 발생했습니다.
    *   **해결**: Ollama의 기본 레지스트리에 있는 범용 13B 모델인 `llama2`를 `ollama pull llama2` 명령으로 성공적으로 다운로드했습니다.
*   **코드 수정**:
    *   `rag_poc.py` 및 `test_rag.py` 파일에서 기존 `langchain_community.llms.LlamaCpp` (또는 `HuggingFacePipeline` in test)를 `langchain_community.llms.Ollama`로 변경했습니다.
    *   `llm = Ollama(model="llama2", ...)` 형태로 모델을 로드하도록 수정했습니다.
*   **추가 오류 해결**: `Ollama` 클래스 인스턴스화 시 `max_tokens` 매개변수가 허용되지 않는 `ValidationError`가 발생하여, 해당 매개변수를 코드에서 제거했습니다.
*   **의존성 업데이트**: `requirements.txt`에서 더 이상 사용하지 않는 `llama-cpp-python`을 제거하고, `ollama` 라이브러리를 추가했습니다.

## 4. 결과 및 현재 RAG 모델 구성

### 4.1. RAG 파이프라인 테스트 결과
*   수정된 `rag_poc.py` 및 `test_rag.py` 스크립트를 실행한 결과, Ollama와 `llama2` 모델을 사용한 RAG 파이프라인이 **성공적으로 작동함**을 확인했습니다.
*   `test_rag.py`를 통해 수행된 다양한 질의응답 테스트에서 `llama2` 모델은 **매우 높은 정확도**를 보여주었습니다. 특히 문서 기반의 사실 질문과 간단한 추론 질문에 대해 정확한 답변을 제공했습니다.

### 4.2. 현재 RAG 모델 구성
*   **LLM (Large Language Model)**:
    *   **모델**: `llama2` (Ollama를 통해 다운로드 및 실행)
    *   **Ollama 사용법**: `ollama pull llama2` 명령으로 모델을 다운로드한 후, Python 코드에서는 `langchain_community.llms.Ollama(model="llama2", ...)` 형태로 연동합니다. Ollama 서버가 백그라운드에서 실행 중이어야 합니다.
    *   **GPU 사용 여부**: Ollama는 시스템에 GPU가 있을 경우 자동으로 이를 활용합니다. 현재 설정된 `rag_poc.py` 및 `test_rag.py` 코드에서는 `Ollama` 인스턴스화 시 GPU 관련 파라미터를 명시적으로 설정하지 않았지만, Ollama 자체의 GPU 지원 기능에 따라 성능이 결정됩니다. `llama2` 모델은 CPU 환경에서도 실행 가능합니다.
*   **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (한국어 임베딩)
*   **벡터 데이터베이스**: ChromaDB
*   **프레임워크**: LangChain (RetrievalQA 체인 사용)

## 5. 향후 과제
*   현재 `llama2` 모델은 한국어에 특화된 모델이 아니므로, 한국어 질의응답의 미묘한 뉘앙스나 복잡한 문맥 처리에는 한계가 있을 수 있습니다.
*   향후 Ollama에서 직접 `pull` 가능한 한국어 특화 13B 모델이 등장하거나, GGUF + Modelfile 방식을 통해 한국어 모델을 Ollama에 연동하는 방법을 추가적으로 탐색할 수 있습니다.
*   LangChain의 DeprecationWarning을 해결하기 위해 최신 패키지 버전으로 업데이트를 고려할 수 있습니다.