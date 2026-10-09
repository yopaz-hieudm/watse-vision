from __future__ import annotations

from enum import StrEnum


class WorkflowAction(StrEnum):
    DOWNLOAD_DATA = "download_data"
    VERIFY_DATA = "verify_data"
    RUN_EDA = "run_eda"
    CREATE_SPLIT = "create_split"
    START_TRAINING = "start_training"
    RESUME_TRAINING = "resume_training"
    EVALUATE_VALIDATION = "evaluate_validation"
    EVALUATE_TEST = "evaluate_test"
    GENERATE_GRADCAM = "generate_gradcam"
    COMPARE_EXPERIMENTS = "compare_experiments"

    @property
    def requires_experiment(self) -> bool:
        return self in {
            self.START_TRAINING,
            self.RESUME_TRAINING,
            self.EVALUATE_VALIDATION,
            self.EVALUATE_TEST,
            self.GENERATE_GRADCAM,
        }
