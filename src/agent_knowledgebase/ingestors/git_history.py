"""GitHistoryIngestor -- extract commit history from a git repository."""

from __future__ import annotations

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


class GitHistoryIngestor:
    """Ingest commit log (messages, authors, dates, diffs) from a git repo.

    Uses *gitpython* to iterate over the commit history.  The URI should
    be a path to a local git repository.
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Read git log for the repository at *uri*."""
        from git import Repo

        meta = metadata or {}
        max_commits: int = int(meta.get("max_commits", 100))

        repo = Repo(uri)
        results: list[RawContent] = []

        for commit in repo.iter_commits(max_count=max_commits):
            lines: list[str] = [
                f"Commit: {commit.hexsha}",
                f"Author: {commit.author.name} <{commit.author.email}>",
                f"Date: {commit.committed_datetime.isoformat()}",
                f"Message: {commit.message.strip()}",
            ]

            # Include diff summary against parent(s).
            if commit.parents:
                parent = commit.parents[0]
                try:
                    diff = parent.diff(commit)
                    if diff:
                        lines.append("\nChanged files:")
                        for d in diff:
                            change_type = d.change_type
                            a_path = d.a_path or ""
                            b_path = d.b_path or ""
                            if change_type == "A":
                                lines.append(f"  + {b_path}")
                            elif change_type == "D":
                                lines.append(f"  - {a_path}")
                            elif change_type == "M":
                                lines.append(f"  M {b_path}")
                            elif change_type == "R":
                                lines.append(f"  R {a_path} -> {b_path}")
                            else:
                                lines.append(f"  {change_type} {b_path or a_path}")
                except Exception:  # noqa: BLE001
                    pass  # some edge cases (initial commits, etc.)

            content = "\n".join(lines)
            commit_meta = {
                "repo_path": uri,
                "commit_sha": commit.hexsha,
                "author": str(commit.author),
                "date": commit.committed_datetime.isoformat(),
                **(meta),
            }
            results.append(RawContent(text=content, metadata=commit_meta))

        return results

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk git history content using token-window splitting."""
        return token_chunk(contents, config)
