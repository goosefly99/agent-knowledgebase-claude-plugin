"""Tests for the pipeline state machine (PipelineManager)."""

from __future__ import annotations

import pytest

from agent_knowledgebase.database import Database
from agent_knowledgebase.models import (
    Knowledgebase,
    PipelinePhase,
    RunStatus,
    Source,
    SourceType,
)
from agent_knowledgebase.services.pipeline import PipelineManager

KB_ID = "kb-test-001"


@pytest.fixture()
def pipeline(test_db: Database) -> PipelineManager:
    """Return a PipelineManager backed by the test database.

    Also inserts a knowledgebase record so that the foreign key on
    ``pipeline_runs.kb_id`` is satisfied.
    """
    test_db.insert_knowledgebase(Knowledgebase(id=KB_ID, name="Test KB"))
    return PipelineManager(test_db)


# ------------------------------------------------------------------
# 1. start_run creates a run at initialize/running
# ------------------------------------------------------------------


class TestStartRun:
    def test_creates_run_at_initialize_running(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        assert run.kb_id == KB_ID
        assert run.phase == PipelinePhase.initialize
        assert run.status == RunStatus.running
        assert run.source_id is None
        assert run.error is None
        assert run.completed_at is None

    def test_start_run_with_source_id(
        self, pipeline: PipelineManager, test_db: Database
    ) -> None:
        source = Source(
            id="src-123", kb_id=KB_ID, source_type=SourceType.file, uri="/tmp/f.txt"
        )
        test_db.insert_source(source)
        run = pipeline.start_run(KB_ID, source_id="src-123")
        assert run.source_id == "src-123"
        assert run.phase == PipelinePhase.initialize
        assert run.status == RunStatus.running


# ------------------------------------------------------------------
# 2. complete_phase marks phase as completed
# ------------------------------------------------------------------


class TestCompletePhase:
    def test_marks_phase_as_completed(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        run = pipeline.complete_phase(run.id)
        assert run.status == RunStatus.completed
        assert run.phase == PipelinePhase.initialize
        assert run.completed_at is not None

    def test_complete_phase_rejects_non_running(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.complete_phase(run.id)
        with pytest.raises(ValueError, match="expected 'running'"):
            pipeline.complete_phase(run.id)


# ------------------------------------------------------------------
# 3. advance_phase moves to next phase
# ------------------------------------------------------------------


class TestAdvancePhase:
    def test_initialize_to_read_source(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.complete_phase(run.id)
        run = pipeline.advance_phase(run.id)
        assert run.phase == PipelinePhase.read_source
        assert run.status == RunStatus.running
        assert run.completed_at is None
        assert run.error is None

    def test_advance_rejects_non_completed_phase(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        # status is running, not completed
        with pytest.raises(ValueError, match="expected 'completed'"):
            pipeline.advance_phase(run.id)

    def test_advance_rejects_failed_phase(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.fail_phase(run.id, "boom")
        with pytest.raises(ValueError, match="expected 'completed'"):
            pipeline.advance_phase(run.id)


# ------------------------------------------------------------------
# 4. Full pipeline walkthrough
# ------------------------------------------------------------------

_EXPECTED_PHASE_SEQUENCE = [
    PipelinePhase.initialize,
    PipelinePhase.read_source,
    PipelinePhase.chunk,
    PipelinePhase.embed,
    PipelinePhase.integrate_wiki,
    PipelinePhase.finalize,
]


class TestFullPipeline:
    def test_walk_all_phases_to_completion(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)

        for i, expected_phase in enumerate(_EXPECTED_PHASE_SEQUENCE):
            assert run.phase == expected_phase
            assert run.status == RunStatus.running

            run = pipeline.complete_phase(run.id)
            assert run.status == RunStatus.completed

            if i < len(_EXPECTED_PHASE_SEQUENCE) - 1:
                run = pipeline.advance_phase(run.id)

        # After completing finalize, the run is done
        assert run.phase == PipelinePhase.finalize
        assert run.status == RunStatus.completed


# ------------------------------------------------------------------
# 5. advance_phase raises on non-completed phase (covered above,
#    but also test with a pending/failed status)
# ------------------------------------------------------------------


class TestAdvancePhaseInvalid:
    def test_advance_on_running_raises(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        with pytest.raises(ValueError):
            pipeline.advance_phase(run.id)

    def test_advance_on_nonexistent_run_raises(self, pipeline: PipelineManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            pipeline.advance_phase("does-not-exist")


# ------------------------------------------------------------------
# 6. fail_phase sets status to failed with error
# ------------------------------------------------------------------


class TestFailPhase:
    def test_sets_failed_with_error(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        run = pipeline.fail_phase(run.id, "disk full")
        assert run.status == RunStatus.failed
        assert run.error == "disk full"
        assert run.phase == PipelinePhase.initialize

    def test_fail_rejects_completed_phase(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.complete_phase(run.id)
        with pytest.raises(ValueError, match="expected 'running'"):
            pipeline.fail_phase(run.id, "too late")


# ------------------------------------------------------------------
# 7. retry_phase resets failed phase to running
# ------------------------------------------------------------------


class TestRetryPhase:
    def test_resets_failed_to_running(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.fail_phase(run.id, "timeout")
        run = pipeline.retry_phase(run.id)
        assert run.status == RunStatus.running
        assert run.error is None
        assert run.phase == PipelinePhase.initialize

    def test_retry_then_complete_and_advance(self, pipeline: PipelineManager) -> None:
        """Retry followed by normal progression should work."""
        run = pipeline.start_run(KB_ID)
        pipeline.fail_phase(run.id, "transient error")
        run = pipeline.retry_phase(run.id)
        run = pipeline.complete_phase(run.id)
        run = pipeline.advance_phase(run.id)
        assert run.phase == PipelinePhase.read_source
        assert run.status == RunStatus.running


# ------------------------------------------------------------------
# 8. retry_phase raises on non-failed phase
# ------------------------------------------------------------------


class TestRetryPhaseInvalid:
    def test_retry_on_running_raises(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        with pytest.raises(ValueError, match="expected 'failed'"):
            pipeline.retry_phase(run.id)

    def test_retry_on_completed_raises(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.complete_phase(run.id)
        with pytest.raises(ValueError, match="expected 'failed'"):
            pipeline.retry_phase(run.id)


# ------------------------------------------------------------------
# 9. Cannot advance past finalize
# ------------------------------------------------------------------


class TestCannotAdvancePastFinalize:
    def test_advance_after_finalize_completed_raises(
        self, pipeline: PipelineManager
    ) -> None:
        run = pipeline.start_run(KB_ID)
        # Walk to finalize
        for _ in range(len(_EXPECTED_PHASE_SEQUENCE) - 1):
            pipeline.complete_phase(run.id)
            run = pipeline.advance_phase(run.id)
        # Complete finalize
        pipeline.complete_phase(run.id)
        # Attempting to advance should raise
        with pytest.raises(ValueError, match="Cannot advance past finalize"):
            pipeline.advance_phase(run.id)


# ------------------------------------------------------------------
# 10. Persistence — data survives read-back from DB
# ------------------------------------------------------------------


class TestPersistence:
    def test_run_persists_after_start(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        reloaded = pipeline.get_run(run.id)
        assert reloaded is not None
        assert reloaded.id == run.id
        assert reloaded.kb_id == KB_ID
        assert reloaded.phase == PipelinePhase.initialize
        assert reloaded.status == RunStatus.running

    def test_phase_change_persists(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.complete_phase(run.id)
        pipeline.advance_phase(run.id)
        reloaded = pipeline.get_run(run.id)
        assert reloaded is not None
        assert reloaded.phase == PipelinePhase.read_source
        assert reloaded.status == RunStatus.running

    def test_error_persists(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        pipeline.fail_phase(run.id, "connection refused")
        reloaded = pipeline.get_run(run.id)
        assert reloaded is not None
        assert reloaded.status == RunStatus.failed
        assert reloaded.error == "connection refused"

    def test_list_runs_returns_persisted_data(
        self, pipeline: PipelineManager, test_db: Database
    ) -> None:
        source = Source(
            id="src-a", kb_id=KB_ID, source_type=SourceType.file, uri="/tmp/a.txt"
        )
        test_db.insert_source(source)
        pipeline.start_run(KB_ID)
        pipeline.start_run(KB_ID, source_id="src-a")
        runs = pipeline.list_runs(KB_ID)
        assert len(runs) == 2

    def test_get_run_returns_none_for_missing(self, pipeline: PipelineManager) -> None:
        assert pipeline.get_run("nonexistent") is None

    def test_completed_at_persists(self, pipeline: PipelineManager) -> None:
        run = pipeline.start_run(KB_ID)
        run = pipeline.complete_phase(run.id)
        reloaded = pipeline.get_run(run.id)
        assert reloaded is not None
        assert reloaded.completed_at is not None
