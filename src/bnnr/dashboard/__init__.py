"""Dashboard package exports and namespace marker."""

__all__ = ["create_dashboard_app", "list_runs", "start_dashboard"]

# Explicitly define exported names for static analysis; values are provided lazily.
create_dashboard_app = None
list_runs = None
start_dashboard = None


def __getattr__(name: str):
    if name in {"create_dashboard_app", "list_runs"}:
        from bnnr.dashboard.backend import create_dashboard_app, list_runs

        return {"create_dashboard_app": create_dashboard_app, "list_runs": list_runs}[name]
    if name == "start_dashboard":
        from bnnr.dashboard.serve import start_dashboard

        return start_dashboard
    raise AttributeError(name)
