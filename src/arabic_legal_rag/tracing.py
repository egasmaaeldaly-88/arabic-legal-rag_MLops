"""Optional Langfuse tracing for the RAG pipeline.

Everything here is a safe no-op unless BOTH of these environment variables are set
(and the ``langfuse`` package is installed):

    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY      (+ LANGFUSE_HOST, e.g. http://localhost:3000)

Set LANGFUSE_ENABLED=0 to force tracing off (the tests do this). Tracing must never break
the API: every Langfuse call is wrapped, so a missing or unreachable Langfuse server only
means "no traces", never a failed /ask.

Trace layout for one question::

    ask                      (root span: input = question, output = answer + sources)
    ├── retrieve             (span:       output = cited articles)
    └── generate             (generation: model name, prompt inputs, answer)
"""
from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager

from loguru import logger

_client = None  # None = not resolved yet, False = disabled/unavailable


class _NoopSpan:
    """Stand-in used when tracing is off, so callers never need ``if tracing:``."""

    trace_id = None

    def update(self, **_kwargs):
        return self


def enabled() -> bool:
    if os.getenv("LANGFUSE_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """Return the Langfuse client, or None when tracing is off or unavailable."""
    global _client
    if _client is None:
        if not enabled():
            _client = False
        else:
            try:
                from langfuse import get_client

                _client = get_client()
                logger.info("Langfuse tracing enabled")
            except Exception as e:  # not installed, bad config, ...
                logger.warning(f"Langfuse tracing disabled: {e}")
                _client = False
    return _client or None


def reset() -> None:
    """Forget the cached client (used by tests)."""
    global _client
    _client = None


@contextmanager
def span(name: str, as_type: str = "span", tags: list[str] | None = None, **kwargs):
    """Open a Langfuse observation as the current span; yields a no-op object if off.

    Spans opened inside another ``span(...)`` become its children automatically.
    ``tags`` are applied to the whole trace (via ``propagate_attributes``, the way SDK v3+
    wants it). Other ``kwargs`` go to Langfuse (input=, metadata=, model=, ...).
    """
    lf = get_langfuse()
    cm = None
    if lf is not None:
        try:
            cm = lf.start_as_current_observation(as_type=as_type, name=name, **kwargs)
        except Exception as e:
            logger.warning(f"Langfuse span '{name}' failed to start: {e}")
    if cm is None:
        yield _NoopSpan()
        return
    with ExitStack() as stack:
        if tags:
            try:
                from langfuse import propagate_attributes

                stack.enter_context(propagate_attributes(tags=list(tags)))
            except Exception as e:  # old SDK without it, or invalid tags: trace still works
                logger.debug(f"Langfuse tags skipped: {e}")
        yield stack.enter_context(cm)


def safe_update(obs, **kwargs) -> None:
    """``obs.update(...)`` that can never raise."""
    try:
        obs.update(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse update skipped: {e}")


def trace_id_of(obs) -> str | None:
    return getattr(obs, "trace_id", None)


def score_trace(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    """Attach a numeric score (e.g. RAGAS faithfulness) to an existing trace.

    The score id is ``<trace_id>-<name>``, so re-sending the same score updates it
    instead of creating a duplicate.
    """
    lf = get_langfuse()
    if lf is None or not trace_id or value is None:
        return
    try:
        lf.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment,
            score_id=f"{trace_id}-{name}",
        )
    except Exception as e:
        logger.warning(f"Langfuse score '{name}' failed: {e}")


def flush() -> None:
    """Send buffered traces now (call before a short-lived script exits)."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed: {e}")


def record_ask_trace(
    question: str,
    sources: list[str],
    answer: str,
    *,
    model: str,
    endpoint: str,
    timings_ms: dict | None = None,
    tags: list[str] | None = None,
) -> str | None:
    """Create a finished trace after the fact; used by the streaming endpoints.

    A span cannot stay open across ``yield`` in a streaming response (the generator may be
    resumed on another thread), so streaming records the whole trace once the stream ends.
    Durations are stored in the trace metadata instead of the span timings.
    """
    if get_langfuse() is None:
        return None
    meta = {"endpoint": endpoint, **({"timings_ms": timings_ms} if timings_ms else {})}
    with span(
        "ask",
        tags=tags or [endpoint.strip("/") or "ask"],
        input={"question": question},
        metadata=meta,
    ) as root:
        with span("retrieve", input={"question": question}) as r:
            safe_update(r, output={"sources": sources})
        with span("generate", as_type="generation", model=model, input={"question": question}) as g:
            safe_update(g, output=answer)
        safe_update(root, output={"answer": answer, "sources": sources})
        return trace_id_of(root)
