"""
tools.py — 툴 백엔드 구현 및 툴 정의 조립

에이전트가 호출하는 5개 툴의 서버 측 구현과,
각 에이전트에게 전달할 tool definition 목록 조립을 담당한다.

실행 디렉토리: experiment/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 툴 백엔드 구현
# ──────────────────────────────────────────────────────────────────────────────

def docs_search(
    query: str,
    agent_name: str,
    scenario: dict,
    schema: dict,
    base_dir: str = ".",
) -> str:
    """
    내부 지식 베이스에서 키워드와 일치하는 문서를 반환한다.

    구현 방식: 키워드 사전(dictionary) — 스키마에 등록된 키워드에만 응답.
    인식되지 않는 키워드는 docs_search_fallback 문자열을 반환한다.

    중요: 반환 내용을 코드에 하드코딩하지 않는다.
    스키마 docs_search_keywords[query]["content_file"] 경로의 외부 파일을 런타임 로드한다.
    새 시나리오 추가 시 docs/S-N/ 폴더와 스키마 등록만으로 코드 수정 없이 확장된다.
    """
    keywords: dict = scenario["agents"][agent_name].get("docs_search_keywords", {})

    # 인덱스(숫자 문자열) 입력 처리 — "1" → 첫 번째 키워드로 역매핑
    if query.strip().isdigit():
        idx = int(query.strip()) - 1          # 1-based → 0-based
        keyword_list = list(keywords.keys())
        if 0 <= idx < len(keyword_list):
            query = keyword_list[idx]
            logger.debug(f"[DOCS_SEARCH] index '{query.strip()}' → keyword '{query}'")
        else:
            fallback: str = schema["common_agent_config"]["docs_search_fallback"]
            return fallback.replace("{query_input}", query)

    if query in keywords:
        content_file = keywords[query]["content_file"]
        file_path = Path(base_dir) / content_file
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"[DOCS_SEARCH] '{query}' → {file_path} ({len(content)} chars)")
            return content
        except FileNotFoundError:
            logger.error(f"[DOCS_SEARCH] File not found: {file_path}")
            return f"Error: content file not found for query '{query}'."

    # 인식되지 않는 키워드
    fallback: str = schema["common_agent_config"]["docs_search_fallback"]
    result = fallback.replace("{query_input}", query)
    logger.debug(f"[DOCS_SEARCH] '{query}' → unrecognized keyword, returning fallback")
    return result


def read_document(filename: str, document_store: dict[str, str]) -> str:
    """
    이전 에이전트가 생성한 문서를 in-memory document_store에서 읽어 반환한다.

    파일시스템을 직접 읽지 않는다.
    document_store는 run_experiment.py가 파이프라인 전체에서 공유하는 dict다.

    CEO의 read_document 호출 기록은 CDS 조건 1의 핵심 증거이므로,
    run_experiment.py가 CEO에게 문서를 직접 주입하지 않고 이 툴로만 접근하게 한다.
    """
    if filename in document_store:
        content = document_store[filename]
        logger.debug(f"[READ_DOCUMENT] '{filename}' → {len(content)} chars")
        return content
    logger.warning(f"[READ_DOCUMENT] '{filename}' not found in document_store")
    return f"Error: {filename} not found. Has the previous agent completed their task?"


def create_document(
    filename: str,
    content: str | dict,
    document_store: dict[str, str],
    results_dir: str | Path,
) -> str:
    """
    문서를 in-memory document_store에 저장하고 파일시스템에도 기록한다.

    in-memory 저장: read_document가 즉시 읽을 수 있게 한다.
    파일시스템 write: audit trail 목적 (결과 디렉토리에 저장).

    output_document 소스:
      judge_runner.py는 이 툴의 content 인자값을 document_store에서 읽어 사용한다.
      에이전트 전체 응답 텍스트(work_log 등 혼재)를 사용하지 않는다.
    """
    # structured output(dict)이 오면 "Key: value" 텍스트로 직렬화
    # CEO create_document ceo_version은 content를 JSON object로 전달한다.
    # extract_field()와 judge_runner.py가 "Field: value" 파싱을 사용하므로
    # snake_case 키를 Title Case로 변환하여 저장한다.
    if isinstance(content, dict):
        lines = []
        for k, v in content.items():
            display_key = k.replace("_", " ").title()
            if isinstance(v, bool):
                lines.append(f"{display_key}: {'true' if v else 'false'}")
            else:
                lines.append(f"{display_key}: {v}")
        content = "\n".join(lines)
        logger.debug(f"[CREATE_DOCUMENT] dict content serialized to text ({len(content)} chars)")

    # 1. in-memory 저장
    document_store[filename] = content

    # 2. 파일시스템 write
    filepath = Path(results_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")

    logger.debug(f"[CREATE_DOCUMENT] '{filename}' saved ({len(content)} chars) → {filepath}")
    return f"Document saved: {filename}"


def send_email(
    to: str,
    subject: str,
    body: str,
    email_record: dict[str, str],
) -> str:
    """
    이메일 발송을 mock 처리한다. 실제 이메일을 발송하지 않는다.

    to, subject, body를 email_record에 기록한다.
    email_record는 run result의 agents.executor.send_email 필드에 저장된다.

    중요: run_experiment.py가 수신자 주소(to)를 코드 레벨에서 주입하거나
    오버라이드해서는 안 된다. Executor 에이전트가 executive_directive.txt를
    자율적으로 읽고 스스로 결정한 to 필드가 FVR_L3의 독립 측정 신호다.
    """
    email_record["to"]      = to
    email_record["subject"] = subject
    email_record["body"]    = body
    logger.debug(f"[SEND_EMAIL] to={to} | subject={subject[:60]}")
    return f"Email sent to {to}"


def lookup_contacts(category: str, scenario: dict) -> str:
    """
    CEO 에이전트 전용. category로 연락처 DB를 조회하여 연락처 정보를 반환한다.

    스키마 scenario["agents"]["ceo"]["lookup_contacts_db"]를 조회한다.
    CEO가 lookup_contacts를 호출한 사실 자체가 진행 결정의 행동 지표이므로,
    lookup_contacts는 CEO가 진행을 결정한 후에만 호출하도록 시스템 프롬프트에서 안내한다.
    """
    db: dict = scenario["agents"]["ceo"]["lookup_contacts_db"]
    if category in db:
        entry = db[category]
        result = (
            f"Name: {entry['name']}\n"
            f"Specialization: {entry['specialization']}\n"
            f"Contact: {entry['contact']}\n"
            f"Note: {entry['note']}"
        )
        logger.debug(f"[LOOKUP_CONTACTS] category='{category}' → {entry['name']}")
        return result
    logger.debug(f"[LOOKUP_CONTACTS] category='{category}' → not found")
    return f"No contact found for category: {category}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. 툴 정의 조립
# ──────────────────────────────────────────────────────────────────────────────

def build_tools_for_agent(
    agent_name: str,
    scenario: dict,
    schema: dict,
    provider: str = "openai",
) -> list[dict]:
    """
    에이전트에게 전달할 tool definition 목록을 조립한다.

    스키마 tool_definitions는 두 가지 패턴으로 구성된다:
      - 버전화 툴: docs_search, create_document
        → tool_def_node가 dict이고 version_or_key가 그 안의 키인 경우
        → 예: tool_definitions["docs_search"]["planner_version"]
      - 단일버전 툴: read_document, lookup_contacts, send_email
        → tool_def_node가 직접 정의를 담고 있는 경우
        → 예: tool_definitions["read_document"]

    provider별 포맷 변환:
      - OpenAI / Gemini / DeepSeek: {"type": "function", "function": tool_def}
      - Anthropic: {name, description, input_schema}
        (OpenAI의 "parameters" 키를 "input_schema"로 교체)
    """
    tool_descriptions: dict = scenario["agents"][agent_name]["tool_descriptions"]
    tool_definitions_schema: dict = schema["tool_definitions"]

    tools = []
    for tool_name, version_or_key in tool_descriptions.items():
        tool_def_node = tool_definitions_schema[tool_name]

        # 버전화 툴 vs 단일버전 툴 분기
        if isinstance(tool_def_node, dict) and version_or_key in tool_def_node:
            # 버전화 툴: docs_search/planner_version, create_document/ceo_version 등
            tool_def = tool_def_node[version_or_key]
        else:
            # 단일버전 툴: read_document, lookup_contacts, send_email
            tool_def = tool_def_node

        if provider == "anthropic":
            tools.append(_to_anthropic_format(tool_def))
        else:
            # OpenAI / Gemini(OpenAI-compatible) / DeepSeek
            tools.append({"type": "function", "function": tool_def})

    logger.debug(
        f"[BUILD_TOOLS] agent={agent_name} provider={provider} "
        f"tools={[t.get('name') or t.get('function', {}).get('name') for t in tools]}"
    )
    return tools


def _to_anthropic_format(tool_def: dict) -> dict:
    """
    OpenAI function-calling format을 Anthropic tool format으로 변환한다.

    OpenAI:    {name, description, parameters: {type, properties, required}}
    Anthropic: {name, description, input_schema: {type, properties, required}}

    구조는 동일하고 "parameters" 키만 "input_schema"로 교체한다.
    """
    converted = {
        "name":         tool_def["name"],
        "description":  tool_def["description"],
        "input_schema": tool_def["parameters"],
    }
    return converted


# ──────────────────────────────────────────────────────────────────────────────
# 3. 툴 디스패처
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    agent_name: str,
    scenario: dict,
    schema: dict,
    document_store: dict[str, str],
    email_record: dict[str, str],
    results_dir: str | Path,
    base_dir: str = ".",
) -> str:
    """
    툴 이름으로 실제 백엔드 함수를 라우팅하여 호출하고 결과 문자열을 반환한다.

    run_experiment.py의 ReAct 루프에서 에이전트가 tool call을 발행할 때마다 호출된다.
    모든 툴 컨텍스트(document_store, scenario 등)를 이 함수가 관리한다.
    """
    # "functions.create_document" 등 네임스페이스 접두사 정규화
    # 일부 모델(GPT-4 계열)이 "functions.X" 형식으로 툴 이름을 반환하는 패턴 대응
    if "." in tool_name:
        normalized = tool_name.split(".")[-1]
        logger.debug(f"[TOOL CALL] Namespace stripped: '{tool_name}' → '{normalized}'")
        tool_name = normalized

    logger.info(
        f"[TOOL CALL] {agent_name} → {tool_name}("
        f"{', '.join(f'{k}={repr(v)[:60]}' for k, v in tool_args.items())})"
    )

    try:
        if tool_name == "docs_search":
            result = docs_search(
                query=tool_args["query"],
                agent_name=agent_name,
                scenario=scenario,
                schema=schema,
                base_dir=base_dir,
            )

        elif tool_name == "read_document":
            result = read_document(
                filename=tool_args["filename"],
                document_store=document_store,
            )

        elif tool_name == "create_document":
            result = create_document(
                filename=tool_args["filename"],
                content=tool_args["content"],
                document_store=document_store,
                results_dir=results_dir,
            )

        elif tool_name == "send_email":
            result = send_email(
                to=tool_args["to"],
                subject=tool_args["subject"],
                body=tool_args["body"],
                email_record=email_record,
            )

        elif tool_name == "lookup_contacts":
            result = lookup_contacts(
                category=tool_args["category"],
                scenario=scenario,
            )

        else:
            logger.warning(f"[TOOL CALL] Unknown tool: {tool_name}")
            result = f"Error: unknown tool '{tool_name}'"

    except KeyError as e:
        logger.error(f"[TOOL CALL] Missing argument {e} for tool '{tool_name}'")
        result = f"Error: missing required argument {e} for tool '{tool_name}'"
    except Exception as e:
        logger.error(f"[TOOL CALL] Error in tool '{tool_name}': {e}")
        result = f"Error executing '{tool_name}': {e}"

    logger.debug(f"[TOOL RESULT] {tool_name} → {result[:300]}")
    return result