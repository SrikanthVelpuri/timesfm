/* ===========================================================
   AA Pricing Forecast Portfolio - Tab navigation
   Vanilla JS, no framework. Loaded after the DOM in index.html.
   =========================================================== */

(function () {
  "use strict";

  const tabs = document.querySelectorAll(".tab[data-tab]");
  const panels = document.querySelectorAll(".panel[id]");

  function activateTab(tabId) {
    tabs.forEach(function (tab) {
      const isActive = tab.dataset.tab === tabId;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    panels.forEach(function (panel) {
      panel.classList.toggle("active", panel.id === tabId);
    });
    // Update hash without triggering scroll jumps.
    if (history.replaceState) {
      history.replaceState(null, "", "#" + tabId);
    }
    // Re-render Chart.js charts when their tab becomes visible.
    if (typeof window.aaPortfolioRenderTab === "function") {
      window.aaPortfolioRenderTab(tabId);
    }
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      activateTab(tab.dataset.tab);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  // Activate tab from URL hash on load (e.g. /#timeline).
  function activateFromHash() {
    const hash = (window.location.hash || "").replace("#", "");
    if (hash && document.getElementById(hash)) {
      activateTab(hash);
    } else {
      activateTab("overview");
    }
  }

  // Internal tab links (e.g. <a data-tab-link="dashboard">).
  document.querySelectorAll("[data-tab-link]").forEach(function (link) {
    link.addEventListener("click", function (event) {
      event.preventDefault();
      activateTab(link.dataset.tabLink);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  document.addEventListener("DOMContentLoaded", activateFromHash);
  window.addEventListener("hashchange", activateFromHash);
})();
