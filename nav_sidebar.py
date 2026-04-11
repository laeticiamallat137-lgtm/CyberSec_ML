"""
Compact sidebar: two sections with Material icons (not emoji).
Requires streamlit>=1.33 (st.page_link, Material :material/*: icons, showSidebarNavigation).

Streamlit collapses the sidebar with max-width:0 and translateX(-width); we override
that so a slim icon rail stays visible when collapsed.
"""

from __future__ import annotations

import streamlit as st


def inject_compact_sidebar_css() -> None:
    st.markdown(
        """
<style>
  /* Align with Streamlit's own sidebar transition (≈300ms) */
  section[data-testid="stSidebar"] {
    transition:
      width 300ms cubic-bezier(0.4, 0, 0.2, 1),
      min-width 300ms cubic-bezier(0.4, 0, 0.2, 1),
      max-width 300ms cubic-bezier(0.4, 0, 0.2, 1),
      transform 300ms cubic-bezier(0.4, 0, 0.2, 1) !important;
  }

  section[data-testid="stSidebar"] > div {
    transition: padding 280ms cubic-bezier(0.4, 0, 0.2, 1) !important;
  }

  /*
   * Collapsed: undo Streamlit's off-canvas hide (min/max 0 + negative translate)
   * so page links stay visible as an icon rail.
   */
  section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(0) !important;
    width: 3.5rem !important;
    min-width: 3.5rem !important;
    max-width: 3.5rem !important;
    overflow: visible !important;
  }

  section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] {
    min-width: 100% !important;
    opacity: 1 !important;
    visibility: visible !important;
  }

  section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] {
    opacity: 1 !important;
    visibility: visible !important;
  }

  /* Page links: hover + current page (darker blue) */
  a[data-testid="stPageLink-NavLink"] {
    border-radius: 8px !important;
    transition: background-color 180ms ease, box-shadow 180ms ease !important;
  }
  a[data-testid="stPageLink-NavLink"]:hover {
    background-color: rgba(30, 58, 95, 0.12) !important;
  }
  a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background-color: rgba(30, 58, 95, 0.28) !important;
  }
  a[data-testid="stPageLink-NavLink"][aria-current="page"]:hover {
    background-color: rgba(30, 58, 95, 0.34) !important;
  }

  /*
   * Collapsed: PageLink order is [icon, label span]. Hide only the label column.
   */
  section[data-testid="stSidebar"][aria-expanded="false"] a[data-testid="stPageLink-NavLink"] {
    justify-content: center !important;
    padding-left: 0.35rem !important;
    padding-right: 0.35rem !important;
    gap: 0 !important;
  }
  section[data-testid="stSidebar"][aria-expanded="false"]
    a[data-testid="stPageLink-NavLink"] > *:nth-child(2) {
    display: none !important;
  }

  /*
   * Our CSS keeps a visible icon rail while Streamlit still thinks the sidebar is
   * "collapsed", so it also renders the main-area expand control (>>). That
   * duplicates the sidebar toggle (<<). Hide the extra >> — one control is enough.
   */
  [data-testid="stExpandSidebarButton"] {
    display: none !important;
  }

  /*
   * Sidebar toggle: replace default chevrons with a 3-line hamburger (theme blue).
   */
  [data-testid="stSidebarCollapseButton"] button {
    position: relative !important;
  }
  [data-testid="stSidebarCollapseButton"] button > * {
    opacity: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
  }
  [data-testid="stSidebarCollapseButton"] button::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 1.125rem;
    height: 2px;
    background-color: #1e3a5f;
    border-radius: 1px;
    box-shadow: 0 -5px 0 #1e3a5f, 0 5px 0 #1e3a5f;
    pointer-events: none;
  }
</style>
        """,
        unsafe_allow_html=True,
    )
    _strip_sidebar_control_tooltips()


def _strip_sidebar_control_tooltips() -> None:
    """Remove native browser tooltips (title) on sidebar chrome and nav links."""
    st.html(
        """<script>
(function () {
  function strip() {
    document.querySelectorAll(
      '[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"], ' +
      'a[data-testid="stPageLink-NavLink"]'
    ).forEach(function (root) {
      root.removeAttribute("title");
      root.querySelectorAll("button").forEach(function (el) {
        el.removeAttribute("title");
      });
    });
  }
  strip();
  new MutationObserver(strip).observe(document.body, { childList: true, subtree: true });
})();
</script>""",
        unsafe_allow_javascript=True,
        width="content",
    )


def render_minimal_sidebar_nav() -> None:
    """Main app sections with Material Symbols (professional, not emoji)."""
    with st.sidebar:
        st.page_link(
            "streamlit_app.py",
            label="Results",
            icon=":material/analytics:",
        )
        st.page_link(
            "pages/1_Interactive_Test.py",
            label="Interactive",
            icon=":material/science:",
        )
        st.page_link(
            "pages/2_Deployment_Sim.py",
            label="Deployment",
            icon=":material/cloud_upload:",
        )
