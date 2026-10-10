from __future__ import annotations

from pathlib import Path

from realwaste.workflow.actions import WorkflowAction
from realwaste.workflow.policies import WorkflowFacts, evaluate_policy
from realwaste.workflow.state import ExperimentState


def _experiment(
    experiment_id: str,
    *,
    checkpoint: bool = False,
    test_metrics: bool = False,
) -> ExperimentState:
    path = Path(f"/tmp/{experiment_id}/best.pt") if checkpoint else None
    return ExperimentState(
        experiment_id=experiment_id,
        experiment_name=experiment_id.lower(),
        model_name="resnet18",
        augmentation_enabled=False,
        output_dir=Path(f"/tmp/{experiment_id}"),
        best_checkpoint=path,
        last_checkpoint=None,
        history_path=None,
        best_checkpoint_compatible=checkpoint,
        last_checkpoint_compatible=False,
        validation_metrics=None,
        test_metrics={"macro_f1": 0.8} if test_metrics else None,
        validation_stale=False,
        test_stale=False,
        resume_ready=False,
        test_predictions_ready=test_metrics,
    )


def test_policy_recommends_test_after_training() -> None:
    selected = _experiment("E1", checkpoint=True)
    facts = WorkflowFacts(
        data_ready=True,
        eda_ready=True,
        split_ready=True,
        selected_experiment=selected,
        experiments=(selected,),
        active_job=None,
        comparison_ready=False,
    )

    decision = evaluate_policy(facts)

    assert WorkflowAction.EVALUATE_TEST in decision.allowed_actions
    assert decision.next_action is WorkflowAction.EVALUATE_TEST


def test_policy_recommends_compare_only_after_all_test_results() -> None:
    experiments = tuple(
        _experiment(experiment_id, checkpoint=True, test_metrics=True)
        for experiment_id in ("E1", "E2", "E3", "E4")
    )
    facts = WorkflowFacts(
        data_ready=True,
        eda_ready=True,
        split_ready=True,
        selected_experiment=experiments[0],
        experiments=experiments,
        active_job=None,
        comparison_ready=False,
    )

    decision = evaluate_policy(facts)

    assert WorkflowAction.COMPARE_EXPERIMENTS in decision.allowed_actions
    assert decision.next_action is WorkflowAction.COMPARE_EXPERIMENTS


def test_comparison_blocks_different_feedback_training_cohorts() -> None:
    from dataclasses import replace

    experiments = tuple(
        _experiment(experiment_id, checkpoint=True, test_metrics=True)
        for experiment_id in ("E1", "E2", "E3", "E4")
    )
    experiments = (
        replace(experiments[0], test_metrics={"macro_f1": 0.8, "feedback_fingerprint": "v1"}),
        *experiments[1:],
    )
    decision = evaluate_policy(
        WorkflowFacts(
            data_ready=True,
            eda_ready=True,
            split_ready=True,
            selected_experiment=experiments[0],
            experiments=experiments,
            active_job=None,
            comparison_ready=False,
        )
    )
    assert WorkflowAction.COMPARE_EXPERIMENTS not in decision.allowed_actions
    assert "bản chụp phản hồi" in decision.blockers[WorkflowAction.COMPARE_EXPERIMENTS][0]
