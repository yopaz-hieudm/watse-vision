from __future__ import annotations

from pathlib import Path

from realwaste.workflow.state import ExperimentState, WorkflowState


def _experiment(experiment_id: str, *, trained: bool, evaluated: bool) -> ExperimentState:
    checkpoint = Path(f"/tmp/{experiment_id}/best.pt") if trained else None
    metrics = {"macro_f1": 0.8} if evaluated else None
    return ExperimentState(
        experiment_id=experiment_id,
        experiment_name=experiment_id.lower(),
        model_name="resnet18",
        augmentation_enabled=False,
        output_dir=Path(f"/tmp/{experiment_id}"),
        best_checkpoint=checkpoint,
        last_checkpoint=None,
        history_path=None,
        best_checkpoint_compatible=trained,
        last_checkpoint_compatible=False,
        validation_metrics=None,
        test_metrics=metrics,
        validation_stale=False,
        test_stale=False,
        resume_ready=False,
        test_predictions_ready=False,
    )


def test_workflow_statuses_represent_partial_parallel_progress() -> None:
    experiments = {
        "E1": _experiment("E1", trained=True, evaluated=True),
        "E2": _experiment("E2", trained=True, evaluated=False),
        "E3": _experiment("E3", trained=False, evaluated=False),
        "E4": _experiment("E4", trained=False, evaluated=False),
    }
    state = WorkflowState(
        data_ready=True,
        eda_ready=True,
        split_ready=True,
        split_fingerprint="fp",
        dataset_summary={},
        split_metadata={},
        selected_experiment_id="E2",
        experiments=experiments,
        active_job=None,
        comparison_ready=False,
        inference_checkpoint=experiments["E1"].best_checkpoint,
        allowed_actions=frozenset(),
        action_blockers={},
        next_action=None,
    )

    assert state.selected_experiment.experiment_id == "E2"
    assert state.trained_experiment_count == 2
    assert state.evaluated_experiment_count == 1
    assert state.workflow_statuses() == {
        "data": "complete",
        "train": "partial",
        "evaluate": "partial",
        "compare": "pending",
        "predict": "complete",
    }
