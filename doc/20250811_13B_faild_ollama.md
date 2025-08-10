# 13B LLM 모델 실행 문제 해결 기록 (WIN 193 오류 및 Ollama 전환)

## 1. 초기 문제 발생 (WIN 193 오류)

- **오류**: 13B LLM 모델을 양자화하여 실행하려 할 때 `WIN 193` 오류 발생.
- **환경**: Windows 10/11, Python 3.13.6, `llama-cpp-python` 사용.
- **오류 메시지**: `%1은(는) 올바른 Win32 응용 프로그램이 아닙니다` (특히 `llama.dll` 로드 시).

## 2. `llama-cpp-python` 문제 해결 시도

### 2.1. Python 버전 및 아키텍처 확인
- `python -c "import platform; print(platform.architecture())"` 명령 실행 시 따옴표 문제로 실패.
- 임시 스크립트 `check_arch.py`를 통해 Python 인터프리터가 64비트임을 확인 (`('64bit', 'WindowsPE')`).

### 2.2. Visual C++ Redistributables 확인
- x86 및 x64 버전의 Visual C++ Redistributables (2015-2022) 모두 설치되어 있음을 확인.

### 2.3. `llama-cpp-python` 재설치 및 환경 정리
- `rag_poc.py`에서 `llama-cpp-python`을 사용하여 GGUF 모델을 로드하는 것을 확인.
- `llama.dll`에서 `WIN 193` 오류가 발생하는 것을 확인.
- `pip uninstall llama-cpp-python` 시 설치되지 않았다고 보고되어, 수동으로 `.venv\Lib\site-packages\llama_cpp` 디렉토리 삭제.
- `pip cache purge`로 pip 캐시 정리.
- `CMAKE_ARGS` 및 `FORCE_CMAKE` 환경 변수를 사용하여 `llama-cpp-python` 재설치 시도 (PowerShell 환경 변수 설정 문제로 실패).
- `--config-settings`를 사용하여 재설치 시도 (인식되지 않는 옵션 오류로 실패).
- `--no-cache-dir`만 사용하여 재설치 시도 (소스에서 빌드되었으나 `WIN 193` 오류 지속).

### 2.4. 시스템 전체 Python 환경 정리
- `nvcc --version`을 통해 CUDA 11.8 확인.
- `llama.dll` 오류가 시스템 Python 경로(`C:\python\python3.13.6\Lib\site-packages\llama_cpp\lib\llama.dll`)를 가리키는 것을 발견.
- 시스템 전체 Python에서 `llama-cpp-python` 제거 (`C:\python\python3.13.6\Scripts\pip.exe uninstall llama-cpp-python -y`).
- 시스템 전체 `llama_cpp` 디렉토리 수동 삭제 시도 (이미 존재하지 않아 성공).

### 2.5. 가상 환경 재구축 및 Python 버전 다운그레이드
- `WIN 193` 오류가 가상 환경 내 `llama.dll`을 가리키는 문제 지속.
- `pip install -r requirements.txt --verbose` 실행 시 `scipy`, `numpy` 등 일부 패키지가 Python 3.13.6과 호환되지 않는다는 오류 메시지 확인.
- **결론**: Python 버전 비호환성이 근본 원인임을 파악.
- **해결책**: Python 3.10으로 다운그레이드 결정.
- 기존 `.venv` 디렉토리 삭제.
- Python 3.10으로 새 가상 환경 생성.
- `pip install -r requirements.txt` 재실행 (사용자가 수동으로 수행).
- `rag_poc.py` 실행 시 여전히 `WIN 193` 오류 발생.

## 3. Ollama로 전환 결정

- `llama-cpp-python`의 지속적인 문제로 인해 Ollama로 전환하기로 결정.
- **Ollama 설치 완료.**

## 4. 향후 계획 (Ollama 연동)

1.  **Ollama 모델 준비**:
 
2. **`rag_poc.py` 수정**:
    - `langchain_community.llms.LlamaCpp` 대신 `langchain_community.llms.Ollama`를 사용하도록 코드 변경.
    - `llm = Ollama(model="koalpaca-13b")`와 같이 모델 로딩 부분 수정.

3.  **RAG 파이프라인 테스트**:
    - 수정된 `rag_poc.py`를 실행하여 Ollama와 연동이 잘 되는지 확인.
    - `WIN 193` 오류가 해결되고 RAG 파이프라인이 정상 작동하는지 검증.

---

