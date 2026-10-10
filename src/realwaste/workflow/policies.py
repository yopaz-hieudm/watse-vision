from __future__ import annotations

from dataclasses import dataclass

from realwaste.ui.job_manager import JobRecord
from realwaste.workflow.actions import WorkflowAction
from realwaste.workflow.state import ExperimentState


@dataclass(frozen=True, slots=True)
class WorkflowFacts:
    data_ready: bool
    eda_ready: bool
    split_ready: bool
    selected_experiment: ExperimentState
    experiments: tuple[ExperimentState, ...]
    active_job: JobRecord | None
    comparison_ready: bool


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed_actions: frozenset[WorkflowAction]
    blockers: dict[WorkflowAction, tuple[str, ...]]
    next_action: WorkflowAction | None


def evaluate_policy(facts: WorkflowFacts) -> PolicyDecision:
    blockers = {action: _blockers_for(action, facts) for action in WorkflowAction}
    allowed = frozenset(action for action, reasons in blockers.items() if not reasons)
    return PolicyDecision(
        allowed_actions=allowed,
        blockers=blockers,
        next_action=_next_action(facts, allowed),
    )


def _blockers_for(action: WorkflowAction, facts: WorkflowFacts) -> tuple[str, ...]:
    reasons: list[str] = []
    experiment = facts.selected_experiment

    if facts.active_job is not None:
        reasons.append(f"Job {facts.active_job.job_id} ({facts.active_job.kind}) đang chạy")

    if action is WorkflowAction.VERIFY_DATA and not facts.data_ready:
        reasons.append("Dataset chưa có trên máy")

    if action is WorkflowAction.RUN_EDA and not facts.data_ready:
        reasons.append("Cần chuẩn bị dataset trước khi chạy EDA")

    if action is WorkflowAction.CREATE_SPLIT:
        if not facts.data_ready:
            reasons.append("Dataset chưa sẵn sàng")
        if not facts.eda_ready:
            reasons.append("Cần hoàn thành EDA trước khi tạo split")

    if (
        action
        in {
            WorkflowAction.START_TRAINING,
            WorkflowAction.RESUME_TRAINING,
            WorkflowAction.EVALUATE_VALIDATION,
            WorkflowAction.EVALUATE_TEST,
            WorkflowAction.GENERATE_GRADCAM,
        }
        and not facts.split_ready
    ):
        reasons.append("Train/Validation/Test split chưa sẵn sàng")

    if action is WorkflowAction.START_TRAINING and experiment.has_training_artifacts:
        reasons.append("Experiment đã có artefact; dùng resume hoặc giữ kết quả hiện tại")

    if action is WorkflowAction.RESUME_TRAINING:
        if experiment.last_checkpoint is None:
            reasons.append("Không có last.pt để tiếp tục")
        elif not experiment.resume_ready:
            reasons.extend(experiment.blockers or ("last.pt không tương thích",))

    if (
        action
        in {
            WorkflowAction.EVALUATE_VALIDATION,
            WorkflowAction.EVALUATE_TEST,
            WorkflowAction.GENERATE_GRADCAM,
        }
        and experiment.compatible_checkpoint is None
    ):
        reasons.append("Không có checkpoint tương thích với config và split hiện tại")

    if action is WorkflowAction.GENERATE_GRADCAM and not experiment.test_predictions_ready:
        reasons.append("Cần kết quả dự đoán Test hiện hành trước khi tạo Grad-CAM")

    if action is WorkflowAction.COMPARE_EXPERIMENTS:
        missing = [item.experiment_id for item in facts.experiments if item.test_metrics is None]
        if missing:
            reasons.append(f"Thiếu Test metrics hiện hành của: {', '.join(missing)}")
        else:
            feedback_versions = {
                item.test_metrics.get("feedback_fingerprint")
                for item in facts.experiments
                if item.test_metrics is not None
            }
            if len(feedback_versions) != 1:
                reasons.append("E1–E4 phải dùng cùng bản chụp phản hồi để so sánh công bằng")

    return tuple(dict.fromkeys(reasons))


def _next_action(
    facts: WorkflowFacts,
    allowed: frozenset[WorkflowAction],
) -> WorkflowAction | None:
    if facts.active_job is not None:
        return None
    if not facts.data_ready and WorkflowAction.DOWNLOAD_DATA in allowed:
        return WorkflowAction.DOWNLOAD_DATA
    if not facts.eda_ready and WorkflowAction.RUN_EDA in allowed:
        return WorkflowAction.RUN_EDA
    if not facts.split_ready and WorkflowAction.CREATE_SPLIT in allowed:
        return WorkflowAction.CREATE_SPLIT

    selected = facts.selected_experiment
    if not selected.has_checkpoint and WorkflowAction.START_TRAINING in allowed:
        return WorkflowAction.START_TRAINING
    if selected.test_metrics is None and WorkflowAction.EVALUATE_TEST in allowed:
        return WorkflowAction.EVALUATE_TEST

    all_evaluated = all(item.test_metrics is not None for item in facts.experiments)
    if (
        all_evaluated
        and not facts.comparison_ready
        and WorkflowAction.COMPARE_EXPERIMENTS in allowed
    ):
        return WorkflowAction.COMPARE_EXPERIMENTS
    return None
