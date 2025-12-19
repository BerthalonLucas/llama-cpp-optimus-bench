from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


class _DummyStatus:
    def update(self, label: str | None = None, state: str | None = None, expanded: bool | None = None) -> None:
        return


@contextmanager
def status(label: str, *, expanded: bool = True):
    """Compatibility wrapper around `st.status`.

    Streamlit added `st.status` relatively recently. For older versions, fall back to `st.spinner`.
    """

    if hasattr(st, "status"):
        with st.status(label, expanded=expanded) as s:
            yield s
    else:
        with st.spinner(label):
            yield _DummyStatus()
