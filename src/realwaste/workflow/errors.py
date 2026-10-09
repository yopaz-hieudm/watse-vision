from __future__ import annotations

from collections.abc import Sequence

from realwaste.workflow.actions import WorkflowAction


class WorkflowError(RuntimeError):
    """Base error for project-level workflow operations."""


class WorkflowStateError(WorkflowError):
    """Raised when workflow state cannot be reconstructed safely."""


class WorkflowActionBlockedError(WorkflowError):
    """Raised when an action does not satisfy its business preconditions."""

    def __init__(self, action: WorkflowAction, blockers: Sequence[str]) -> None:
        self.action = action
        self.blockers = tuple(blockers)
        detail = "; ".join(self.blockers) or "hành động không được phép ở trạng thái hiện tại"
        super().__init__(f"Không thể thực hiện {action.value}: {detail}")
