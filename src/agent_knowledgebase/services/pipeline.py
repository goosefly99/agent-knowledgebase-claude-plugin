"""Pipeline state machine for enforcing phase ordering on write operations."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import PipelinePhase, PipelineRun, RunStatus

# Ordered list of pipeline phases used to enforce sequencing.
_PHASE_ORDER: list[PipelinePhase] = list(PipelinePhase)


class PipelineManager:
    """Manages pipeline runs and enforces valid phase transitions.

    Each run progresses through the phases defined by :class:`PipelinePhase`
    in strict order: initialize -> read_source -> chunk -> embed ->
    integrate_wiki -> finalize.

    All state changes are persisted to the database immediately.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, kb_id: str, source_id: str | None = None) -> PipelineRun:
        """Start a new pipeline run.

        Creates a :class:`PipelineRun` at the ``initialize`` phase with
        ``running`` status and persists it to the database.
        """
        run = PipelineRun(
            kb_id=kb_id,
            source_id=source_id,
            phase=PipelinePhase.initialize,
            status=RunStatus.running,
        )
        self._db.insert_pipeline_run(run)
        return run

    def advance_phase(self, run_id: str) -> PipelineRun:
        """Advance to the next phase in the pipeline.

        The current phase must have ``status=completed``.  If the current
        phase is ``finalize`` and it is completed, the run is considered
        done (no further transitions are possible).

        Raises
        ------
        ValueError
            If the run does not exist, the current phase is not completed,
            or the run has already finished (finalize completed).
        """
        run = self._get_run_or_raise(run_id)

        if run.status != RunStatus.completed:
            raise ValueError(
                f"Cannot advance: current phase '{run.phase.value}' has "
                f"status '{run.status.value}', expected 'completed'"
            )

        current_index = _PHASE_ORDER.index(run.phase)

        if current_index >= len(_PHASE_ORDER) - 1:
            raise ValueError(
                "Cannot advance past finalize: the pipeline run is complete"
            )

        next_phase = _PHASE_ORDER[current_index + 1]
        run.phase = next_phase
        run.status = RunStatus.running
        run.completed_at = None
        run.error = None
        self._db.update_pipeline_run(run)
        return run

    def complete_phase(self, run_id: str) -> PipelineRun:
        """Mark the current phase as completed.

        Sets ``status=completed`` and records the completion timestamp.

        Raises
        ------
        ValueError
            If the run does not exist or the current status is not ``running``.
        """
        run = self._get_run_or_raise(run_id)

        if run.status != RunStatus.running:
            raise ValueError(
                f"Cannot complete: current phase '{run.phase.value}' has "
                f"status '{run.status.value}', expected 'running'"
            )

        run.status = RunStatus.completed
        run.completed_at = datetime.now(UTC)
        self._db.update_pipeline_run(run)
        return run

    def fail_phase(self, run_id: str, error: str) -> PipelineRun:
        """Mark the current phase as failed.

        Sets ``status=failed`` and stores the error message.

        Raises
        ------
        ValueError
            If the run does not exist or the current status is not ``running``.
        """
        run = self._get_run_or_raise(run_id)

        if run.status != RunStatus.running:
            raise ValueError(
                f"Cannot fail: current phase '{run.phase.value}' has "
                f"status '{run.status.value}', expected 'running'"
            )

        run.status = RunStatus.failed
        run.error = error
        self._db.update_pipeline_run(run)
        return run

    def retry_phase(self, run_id: str) -> PipelineRun:
        """Retry a failed phase.

        Resets a ``failed`` phase back to ``running`` so it can be
        re-attempted.

        Raises
        ------
        ValueError
            If the run does not exist or the current status is not ``failed``.
        """
        run = self._get_run_or_raise(run_id)

        if run.status != RunStatus.failed:
            raise ValueError(
                f"Cannot retry: current phase '{run.phase.value}' has "
                f"status '{run.status.value}', expected 'failed'"
            )

        run.status = RunStatus.running
        run.error = None
        self._db.update_pipeline_run(run)
        return run

    def get_run(self, run_id: str) -> PipelineRun | None:
        """Get a pipeline run by ID, or ``None`` if it does not exist."""
        return self._db.get_pipeline_run(run_id)

    def list_runs(self, kb_id: str) -> list[PipelineRun]:
        """List all runs for a knowledgebase."""
        return self._db.list_pipeline_runs(kb_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_run_or_raise(self, run_id: str) -> PipelineRun:
        """Fetch a run by ID or raise :class:`ValueError`."""
        run = self._db.get_pipeline_run(run_id)
        if run is None:
            raise ValueError(f"Pipeline run not found: {run_id}")
        return run
