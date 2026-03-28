"""
shared.py — 공통 인프라
run_experiment.py와 judge_runner.py가 공유하는 유틸리티 모음.

실행 디렉토리: experiment/
모든 파일 경로는 experiment/ 아래 상대경로를 사용한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# 1. 스키마 로드
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA_PATH = "scenarios_schema_v4_6.json"


def load_schema(path: str = SCHEMA_PATH) -> dict:
    """메인 스키마 파일을 로드한다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 모델 클라이언트 팩토리
# ──────────────────────────────────────────────────────────────────────────────

# SUPPORTED_MODELS: model_id → provider
# 코드에 하드코딩하지 않는다 — 스키마 experiment_config.models를 기반으로 사용한다.
# 단, provider 매핑은 구현에 필요한 고정값이므로 여기에 정의한다.
SUPPORTED_MODELS: dict[str, str] = {
    "gpt-4o-mini":       "openai",
    "gpt-4o":            "openai",
    "gemini-2.5-flash":  "google",
    "claude-sonnet-4-5": "anthropic",
    "deepseek-v3":       "openai",   # OpenAI-compatible endpoint
}

# API 호출 시 실제 model 파라미터 매핑 (model_id와 다른 경우만)
API_MODEL_NAME: dict[str, str] = {
    "deepseek-v3": "deepseek-chat",  # DeepSeek API 공식 모델명
}

GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# seed 지원 provider
SEED_SUPPORTED_PROVIDERS = {"openai"}


def get_provider(model_id: str) -> str:
    """model_id에 해당하는 provider를 반환한다."""
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_id}. "
                         f"Supported: {list(SUPPORTED_MODELS)}")
    return SUPPORTED_MODELS[model_id]


def get_api_model_name(model_id: str) -> str:
    """API 호출 시 사용할 실제 model 파라미터를 반환한다."""
    return API_MODEL_NAME.get(model_id, model_id)


def get_model_params(provider: str, schema: dict) -> dict:
    """
    API 호출 파라미터(temperature, seed)를 반환한다.
    seed는 OpenAI만 지원 — Anthropic/Gemini/DeepSeek는 생략.
    """
    temperature = schema["experiment_config"]["temperature"]
    params: dict[str, Any] = {"temperature": temperature}
    if provider in SEED_SUPPORTED_PROVIDERS:
        params["seed"] = schema["experiment_config"]["random_seed"]
    return params


def build_model_client(model_id: str):
    """
    model_id에 맞는 SDK 클라이언트를 생성하여 반환한다.

    반환 타입:
      - openai (gpt-4o, gpt-4o-mini, deepseek-v3, gemini-2.5-flash):
            openai.OpenAI 인스턴스
      - anthropic (claude-sonnet-4-5):
            anthropic.Anthropic 인스턴스

    Note: Gemini는 Google 공식 OpenAI-compatible endpoint를 사용하므로
          openai.OpenAI 클라이언트로 처리한다.
    """
    provider = get_provider(model_id)

    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    elif provider == "google":
        import openai
        return openai.OpenAI(
            api_key=os.environ["GOOGLE_API_KEY"],
            base_url=GOOGLE_OPENAI_BASE_URL,
        )

    elif provider == "openai":
        import openai
        if model_id == "deepseek-v3":
            return openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=DEEPSEEK_BASE_URL,
            )
        else:
            return openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_judge_client():
    """
    Judge 모델(claude-opus-4-6)용 Anthropic 클라이언트를 반환한다.
    judge_runner.py에서 사용한다.
    """
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLI 인자 파서
# ──────────────────────────────────────────────────────────────────────────────

def parse_cli_args(description: str = "") -> argparse.Namespace:
    """
    run_experiment.py와 judge_runner.py가 공유하는 CLI 인터페이스.
    모든 인자는 선택적이며 독립적으로 조합 가능하다.
    미지정 시 해당 차원의 전체를 실행한다.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="시나리오 ID (예: S-2). 미지정 시 전체 시나리오 실행.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 ID (예: gpt-4o). 미지정 시 전체 모델 실행.",
    )
    parser.add_argument(
        "--pressure",
        type=str,
        default=None,
        help="Pressure 조건 (예: financial_pressure). 미지정 시 전체 조건 실행.",
    )
    parser.add_argument(
        "--exp-type",
        type=str,
        choices=["MAS", "Baseline"],
        default=None,
        dest="exp_type",
        help="실험 유형 (MAS | Baseline). 미지정 시 둘 다 실행.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=1,
        dest="run_number",
        help="Run 번호 (기본값: 1). 실패한 특정 run 재실행 시 사용.",
    )
    return parser.parse_args()


def build_run_matrix(args: argparse.Namespace, schema: dict) -> list[dict]:
    """
    CLI 인자를 바탕으로 실행할 (scenario, model, pressure, exp_type, run_number) 조합을 반환한다.
    미지정 차원은 스키마에서 전체 목록을 읽어 확장한다.

    중요: pressure 조건 목록은 schema_notes.pressure_conditions에서 읽는다 — 하드코딩 금지.
    """
    scenarios = (
        [args.scenario]
        if args.scenario
        else [sc["scenario_id"] for sc in schema["scenarios"]]
    )
    models = (
        [args.model]
        if args.model
        else schema["experiment_config"]["models"]
    )
    pressures = (
        [args.pressure]
        if args.pressure
        else schema["schema_notes"]["pressure_conditions"]
    )
    exp_types = (
        [args.exp_type]
        if args.exp_type
        else ["MAS", "Baseline"]
    )

    matrix = []
    for scenario_id in scenarios:
        for model in models:
            for pressure in pressures:
                for exp_type in exp_types:
                    matrix.append({
                        "scenario_id": scenario_id,
                        "model":       model,
                        "pressure":    pressure,
                        "exp_type":    exp_type,
                        "run_number":  args.run_number,
                    })
    return matrix


# ──────────────────────────────────────────────────────────────────────────────
# 4. run_id 생성
# ──────────────────────────────────────────────────────────────────────────────

def build_run_id(
    scenario_id: str,
    model: str,
    pressure: str,
    exp_type: str,
    run_number: int,
) -> str:
    """
    run_id를 생성한다.
    포맷: {scenario_id}_{model}_{pressure_condition}_{experiment_type}_run{N}
    예:   S-2_gpt-4o_financial_pressure_MAS_run1

    experiment_type(MAS/Baseline)은 반드시 포함해야 한다.
    동일한 scenario × model × pressure 조합이 MAS와 Baseline 양쪽에 존재하므로,
    없으면 results/failures/ 및 raw_texts 파일이 충돌한다.
    """
    return f"{scenario_id}_{model}_{pressure}_{exp_type}_run{run_number}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. 로깅
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: str, name: str = __name__) -> logging.Logger:
    """
    콘솔 + 파일 동시 출력 로거를 설정한다.
    log_file: "logs/experiment.log" 또는 "logs/judge.log"
    """
    ensure_dirs([Path(log_file).parent])

    logger = logging.getLogger(name)
    if logger.handlers:
        # 이미 핸들러가 있으면 중복 설정하지 않는다
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 콘솔 핸들러 (INFO 이상)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # 파일 핸들러 (DEBUG 이상)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# 6. 필드 추출
# ──────────────────────────────────────────────────────────────────────────────

def extract_field(text: str, field_name: str) -> str:
    """
    "field_name: value" 패턴에서 value를 추출한다.

    실패 시 빈 문자열("")을 반환한다 — RuntimeError를 raise하지 않는다.
    필드 추출 실패는 에이전트가 비표준 형식으로 작성한 데이터 품질 문제이며,
    이 경우 judge_runner.py에서 LLM judge에 전체 문서를 전달한다.

    지원 패턴:
      - "Authorized Method: Option B — Deferred..."
      - "reviewer_warning_acknowledged: true"
      - "Legal Counsel: ops@greencorechad.td"
    """
    # 필드명에서 정규식 특수문자 이스케이프
    escaped = re.escape(field_name)
    # 콜론 이후 같은 줄의 값 추출 (앞뒤 공백 제거)
    pattern = rf"(?i){escaped}\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 7. 실패 run 기록
# ──────────────────────────────────────────────────────────────────────────────

def save_failure_record(
    run_id: str,
    failed_at: str,
    error_type: str,
    error_message: str,
    run_meta: dict,
    logger: logging.Logger | None = None,
) -> None:
    """
    실패한 run의 기록을 results/failures/{run_id}.json에 저장한다.
    실패 후에도 다음 run은 계속 진행한다.

    run_meta: {"scenario_id", "model", "pressure", "exp_type", "run_number"}
    """
    ensure_dirs([Path("results/failures")])

    record = {
        "run_id":        run_id,
        "scenario_id":   run_meta.get("scenario_id", ""),
        "model":         run_meta.get("model", ""),
        "pressure":      run_meta.get("pressure", ""),
        "exp_type":      run_meta.get("exp_type", ""),
        "run_number":    run_meta.get("run_number", 1),
        "failed_at":     failed_at,
        "error_type":    error_type,
        "error_message": error_message,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    path = Path(f"results/failures/{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if logger:
        logger.error(f"[RUN FAILED] {run_id} → saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. 재시도 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def with_retry(
    fn,
    *args,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    logger: logging.Logger | None = None,
    **kwargs,
):
    """
    fn(*args, **kwargs)을 실행하며 실패 시 exponential backoff 재시도한다.

    오류 유형별 처리:
      - rate limit (429): exponential backoff 후 재시도
      - network timeout: 3회 재시도 후 skip
      - 그 외 예외: 즉시 raise

    반환: fn의 반환값
    raise: 최대 재시도 초과 시 마지막 예외를 raise
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            is_retryable = any(
                sig in err_str for sig in
                ["rate limit", "429", "timeout", "connection", "service unavailable", "503"]
            )

            if not is_retryable or attempt == max_retries:
                raise

            wait = backoff_base ** attempt
            if logger:
                logger.warning(
                    f"[RETRY] attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Waiting {wait:.1f}s..."
                )
            time.sleep(wait)

    raise last_exc  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# 9. 디렉토리 생성
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(paths: list[Path]) -> None:
    """필요한 디렉토리를 미리 생성한다. 이미 존재하면 무시한다."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def ensure_result_dirs(scenario_id: str, model: str, pressure: str) -> Path:
    """
    run result 저장 경로 results/{scenario_id}/{model}/{pressure}/를 생성하고 반환한다.
    """
    path = Path(f"results/{scenario_id}/{model}/{pressure}")
    ensure_dirs([path])
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 10. 시나리오 조회 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def get_scenario(scenario_id: str, schema: dict) -> dict:
    """schema에서 scenario_id에 해당하는 시나리오 블록을 반환한다."""
    for sc in schema["scenarios"]:
        if sc["scenario_id"] == scenario_id:
            return sc
    raise ValueError(f"Scenario '{scenario_id}' not found in schema.")


def get_compliant_option_label(scenario: dict) -> str:
    """
    scenario["ceo_decision_options"]에서 legal_status == "COMPLIANT"인
    옵션의 label을 반환한다. judge_runner.py의 Prompt 6 플레이스홀더 조립에 사용한다.
    """
    for opt in scenario["ceo_decision_options"]:
        if opt.get("legal_status") == "COMPLIANT":
            return opt["label"]
    raise ValueError(
        f"No COMPLIANT option found in scenario {scenario['scenario_id']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11. 결과 저장 경로 생성
# ──────────────────────────────────────────────────────────────────────────────

def get_run_result_path(
    scenario_id: str,
    model: str,
    pressure: str,
    exp_type: str,
    run_number: int,
) -> Path:
    """
    run result JSON 저장 경로를 반환한다.
    패턴: results/{scenario_id}/{model}/{pressure}/{exp_type}_run{N}.json
    """
    dir_path = ensure_result_dirs(scenario_id, model, pressure)
    return dir_path / f"{exp_type}_run{run_number}.json"


def get_judge_result_path(
    scenario_id: str,
    model: str,
    pressure: str,
    exp_type: str,
    run_number: int,
) -> Path:
    """
    judge result JSON 저장 경로를 반환한다.
    패턴: results/{scenario_id}/{model}/{pressure}/{exp_type}_run{N}_judge.json
    """
    dir_path = ensure_result_dirs(scenario_id, model, pressure)
    return dir_path / f"{exp_type}_run{run_number}_judge.json"


def get_raw_texts_path(run_id: str) -> Path:
    """
    raw_texts JSON 저장 경로를 반환한다.
    패턴: results/{run_id}_raw_texts.json
    """
    ensure_dirs([Path("results")])
    return Path(f"results/{run_id}_raw_texts.json")


# ──────────────────────────────────────────────────────────────────────────────
# 12. 외부 파일 로드 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def load_lds_dictionary(scenario_id: str, schema: dict, base_dir: str = ".") -> set[str]:
    """
    공통 LDS 딕셔너리와 시나리오별 additions 파일을 로드하여 합집합을 반환한다.
    코드에 어휘를 하드코딩하지 않는다 — 외부 .txt 파일에서 런타임 로드.

    파일 형식: 줄바꿈으로 구분된 단어/구문 목록
    """
    # 공통 딕셔너리
    common_file = schema["experiment_config"]["lds_common_dictionary_file"]
    common_path = Path(base_dir) / common_file
    terms: set[str] = set()
    with open(common_path, encoding="utf-8") as f:
        terms = {line.strip().lower() for line in f if line.strip()}

    # 시나리오별 additions
    scenario = get_scenario(scenario_id, schema)
    additions_file = scenario.get("lds_scenario_dictionary_file")
    if additions_file:
        additions_path = Path(base_dir) / additions_file
        if additions_path.exists():
            with open(additions_path, encoding="utf-8") as f:
                terms.update(line.strip().lower() for line in f if line.strip())

    return terms


def load_required_fields(template_name: str, schema: dict, base_dir: str = ".") -> list[str]:
    """
    document_templates의 fields_file 경로에서 required_fields 목록을 로드한다.
    코드에 필드 목록을 하드코딩하지 않는다.
    """
    fields_file = schema["document_templates"][template_name]["fields_file"]
    file_path = Path(base_dir) / fields_file
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# 13. 진행 상황 출력
# ──────────────────────────────────────────────────────────────────────────────

def log_progress(
    completed: int,
    total: int,
    failures: list,
    start_time: float,
    logger: logging.Logger,
) -> None:
    """각 run 완료 후 진행 상황 요약을 콘솔에 출력한다."""
    elapsed = time.time() - start_time
    if completed > 0:
        eta = (elapsed / completed) * (total - completed)
    else:
        eta = 0.0
    logger.info(
        f"[PROGRESS] {completed}/{total} runs done | "
        f"failures={len(failures)} | "
        f"elapsed={elapsed:.0f}s | "
        f"eta={eta:.0f}s"
    )
