# ProcessGPT Agent Utils

ProcessGPT Agent Utilities - 도구 로더, 지식 관리, 이벤트 로깅, 데이터베이스 유틸리티

## 🚀 주요 기능

### 🛠️ 도구 관리 (`tools/`)
- **SafeToolLoader**: MCP 서버 기반 도구 로딩 및 관리
- **KnowledgeManager**: 개인지식(mem0) 및 사내문서(memento) 검색
- **HumanQueryTool**: 사용자 확인/추가정보 요청 도구

### 📊 유틸리티 (`utils/`)
- **Database**: Supabase 기반 데이터베이스 작업 (재시도, 알림 저장)
- **ContextManager**: 컨텍스트 변수 관리
- **CrewEventLogger**: CrewAI 이벤트 로깅 및 전송

## 📦 설치

```bash
pip install process-gpt-agent-utils
```

## 🔧 사용법

### 도구 로더 사용
```python
from processgpt_agent_utils import SafeToolLoader

# MCP 설정 예시
mcp_config = {
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token"}
        }
    }
}

loader = SafeToolLoader(
    tenant_id="your_tenant",
    user_id="your_user",
    agent_name="your_agent",
    mcp_config=mcp_config
)

tools = loader.create_tools_from_names([
    "mem0", "memento", "human_asked", "github"
])
```

### 지식 검색 사용
```python
from tools.knowledge_manager import Mem0Tool, MementoTool

# 개인지식 검색
mem0_tool = Mem0Tool(tenant_id="tenant", user_id="user")
result = mem0_tool._run("과거 실패 사례")

# 사내문서 검색
memento_tool = MementoTool(tenant_id="tenant")
result = memento_tool._run("회사 정책")
```

### 데이터베이스 작업
```python
from utils.database import initialize_db, save_notification

# DB 초기화
initialize_db()

# 알림 저장
await save_notification(
    title="작업 완료",
    notif_type="workitem_bpm",
    user_ids_csv="user1,user2",
    tenant_id="tenant"
)
```

## 🎯 이모지 로깅

모든 유틸리티는 이모지를 활용한 직관적인 로깅을 제공합니다:

- 🔧 초기화 완료
- 🛠️ 도구 로딩
- 🔍 검색 시작
- ✅ 성공
- ❌ 실패
- ⚠️ 경고
- 📨 이벤트 전송

## 📋 의존성

- `supabase>=2.0.0` - 데이터베이스 연결
- `crewai>=0.152.0,<=0.175.0` - AI 에이전트 프레임워크
- `mem0ai>=0.1.94` - 개인지식 저장소
- `mcp>=1.6.0` - Model Context Protocol
- `pydantic>=2.0.0` - 데이터 검증
- `a2a-sdk>=0.3.0` - A2A 통신

## 🔄 개발

### 개발 의존성 설치
```bash
pip install -e ".[dev]"
```

### 릴리스
```bash
# Linux/Mac
./release.sh 0.1.4
python -m ensurepip --upgrade
# Windows
.\release.ps1 -Version 0.1.1
```

python -m ensurepip --upgrade

## 📄 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!
