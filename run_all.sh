#!/usr/bin/env bash
# run_all.sh — 전체 실험 매트릭스 순차 실행
#
# 실행 위치: experiment/ 디렉토리
# 사용법:
#   bash run_all.sh                  # 전체 120개 조합 실행
#   bash run_all.sh --dry-run        # 실행 없이 명령어만 출력
#   bash run_all.sh --run-number 2   # 모든 조합을 run_number=2로 실행
#
# 특정 조합만 재실행:
#   python run_experiment.py --scenario S-2 --model gpt-4o
#   python judge_runner.py   --scenario S-2 --model gpt-4o
#
#   python run_experiment.py --scenario S-2 --model gpt-4o --pressure financial_pressure --exp-type MAS --run-number 1
#   python judge_runner.py   --scenario S-2 --model gpt-4o --pressure financial_pressure --exp-type MAS --run-number 1
#
# 동작:
#   각 (model × scenario × pressure × exp_type) 조합에 대해
#   run_experiment.py → judge_runner.py 순으로 즉시 연속 실행한다.
#   한 run의 실험이 완료되면 바로 그 run의 판정까지 완료한 뒤 다음 조합으로 넘어간다.
#   실패한 run은 results/failures/ 에 기록되며 다음 조합은 계속 진행한다.

set -uo pipefail
# 주의: set -e 사용하지 않음.
# run_experiment/judge_runner 실패 시 다음 조합으로 계속 진행해야 하므로
# 각 명령의 종료 코드를 명시적으로 처리한다.

# ── 설정 ──────────────────────────────────────────────────────────────────────
SCHEMA="scenarios_schema_v4_6.json"
MODELS=(gpt-4o-mini gpt-4o gemini-2.5-flash claude-sonnet-4-5 deepseek-v3)
SCENARIOS=(S-2 S-3 S-4 S-5)
PRESSURES=(baseline financial_pressure executive_override)
EXP_TYPES=(MAS Baseline)
RUN_NUMBER=1
DRY_RUN=false

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --run-number)
      RUN_NUMBER="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      echo "Usage: bash run_all.sh [--dry-run] [--run-number N]"
      exit 1
      ;;
  esac
done

# ── 실행 환경 확인 ────────────────────────────────────────────────────────────
if [[ ! -f "$SCHEMA" ]]; then
  echo "[ERROR] Schema file not found: $SCHEMA"
  echo "Run from the experiment/ directory."
  exit 1
fi

if [[ ! -f "run_experiment.py" ]]; then
  echo "[ERROR] run_experiment.py not found. Run from the experiment/ directory."
  exit 1
fi

# ── 집계 초기화 ───────────────────────────────────────────────────────────────
total=$(( ${#MODELS[@]} * ${#SCENARIOS[@]} * ${#PRESSURES[@]} * ${#EXP_TYPES[@]} ))
count=0
exp_ok=0
exp_fail=0
judge_ok=0
judge_fail=0
start_time=$(date +%s)

echo "========================================"
echo "  Corporate MAS Experiment — run_all.sh"
echo "========================================"
echo "  Schema:      $SCHEMA"
echo "  Models:      ${MODELS[*]}"
echo "  Scenarios:   ${SCENARIOS[*]}"
echo "  Pressures:   ${PRESSURES[*]}"
echo "  Exp types:   ${EXP_TYPES[*]}"
echo "  Run number:  $RUN_NUMBER"
echo "  Total runs:  $total"
echo "  Dry run:     $DRY_RUN"
echo "========================================"
echo ""

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    for pressure in "${PRESSURES[@]}"; do
      for exp_type in "${EXP_TYPES[@]}"; do
        count=$((count + 1))

        CMD_ARGS="--model ${model} --scenario ${scenario} --pressure ${pressure} --exp-type ${exp_type} --run-number ${RUN_NUMBER}"

        echo ""
        echo "========================================"
        echo "[${count}/${total}] model=${model} scenario=${scenario} pressure=${pressure} exp_type=${exp_type} run=${RUN_NUMBER}"
        echo "========================================"

        if [ "$DRY_RUN" = true ]; then
          echo "[DRY-RUN] python run_experiment.py ${CMD_ARGS}"
          echo "[DRY-RUN] python judge_runner.py ${CMD_ARGS}"
          exp_ok=$((exp_ok + 1))
          judge_ok=$((judge_ok + 1))
          continue
        fi

        # ── run_experiment.py 실행 ────────────────────────────────────
        if python run_experiment.py ${CMD_ARGS}; then
          echo "[OK] run_experiment 완료"
          exp_ok=$((exp_ok + 1))

          # ── judge_runner.py 실행 ──────────────────────────────────
          if python judge_runner.py ${CMD_ARGS}; then
            echo "[OK] judge_runner 완료"
            judge_ok=$((judge_ok + 1))
          else
            echo "[WARN] judge_runner 실패 — 다음 조합으로 계속"
            judge_fail=$((judge_fail + 1))
          fi

        else
          echo "[WARN] run_experiment 실패 — judge_runner 스킵, 다음 조합으로 계속"
          exp_fail=$((exp_fail + 1))
          judge_fail=$((judge_fail + 1))
        fi

        # ── 진행 상황 요약 ────────────────────────────────────────────
        elapsed=$(( $(date +%s) - start_time ))
        if [[ $count -gt 0 ]]; then
          eta=$(( elapsed * (total - count) / count ))
        else
          eta=0
        fi
        echo "[PROGRESS] ${count}/${total} done | exp_ok=${exp_ok} exp_fail=${exp_fail} | judge_ok=${judge_ok} judge_fail=${judge_fail} | elapsed=${elapsed}s eta=${eta}s"

      done
    done
  done
done

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
elapsed_total=$(( $(date +%s) - start_time ))
echo ""
echo "========================================"
echo "  실험 완료"
echo "========================================"
echo "  전체 조합:        $total"
echo "  run_experiment 성공: $exp_ok / 실패: $exp_fail"
echo "  judge_runner 성공:   $judge_ok / 실패: $judge_fail"
echo "  총 소요 시간:     ${elapsed_total}s"
echo ""

if [[ $exp_fail -gt 0 || $judge_fail -gt 0 ]]; then
  echo "  [!] 실패한 run이 있습니다. results/failures/ 를 확인하세요."
  echo "      재실행 예시:"
  echo "        python run_experiment.py --scenario S-X --model MODEL --pressure PRESSURE --exp-type MAS --run-number ${RUN_NUMBER}"
  echo "        python judge_runner.py   --scenario S-X --model MODEL --pressure PRESSURE --exp-type MAS --run-number ${RUN_NUMBER}"
fi

echo "========================================"
