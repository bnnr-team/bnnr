"""Dashboard package exports and namespace marker."""

__all__ = ["create_dashboard_app", "list_runs", "start_dashboard"]


def create_dashboard_app(*args, **kwargs):
    from bnnr.dashboard.backend import create_dashboard_app as _create_dashboard_app

    return _create_dashboard_app(*args, **kwargs)


def list_runs(*args, **kwargs):
    from bnnr.dashboard.backend import list_runs as _list_runs

    return _list_runs(*args, **kwargs)


def start_dashboard(*args, **kwargs):
    from bnnr.dashboard.serve import start_dashboard as _start_dashboard

    return _start_dashboard(*args, **kwargs)
