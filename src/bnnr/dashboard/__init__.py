"""Dashboard package exports and namespace marker."""

__all__ = ["create_dashboard_app", "list_runs", "start_dashboard"]


def __getattr__(name: str):
    if name in {"create_dashboard_app", "list_runs"}:
        from bnnr.dashboard.backend import create_dashboard_app, list_runs

        return {"create_dashboard_app": create_dashboard_app, "list_runs": list_runs}[name]
    if name == "start_dashboard":
        from bnnr.dashboard.serve import start_dashboard

        return start_dashboard
    raise AttributeError(name)
