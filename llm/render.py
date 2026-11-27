"""Plain-text renderers for model tables and service status."""

NAME_WIDTH = 32
SIZE_WIDTH = 10


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return "%d B" % int(value)
            return "%.1f %s" % (value, unit)
        value /= 1024


def _status(profile_id, selected_id, running):
    if running and running.get("id") == profile_id:
        if running.get("ready"):
            return "RUNNING · ready · PID %s · up %s" % (
                running.get("pid"), running.get("uptime", "?"))
        return "RUNNING · loading · PID %s" % running.get("pid")
    if profile_id == selected_id:
        return "selected · not running"
    return "downloaded · not running"


def format_list(profiles, selected_id, local_sizes, running, selected_failure):
    lines = ["  %-*s  %*s  %s" % (NAME_WIDTH, "MODEL", SIZE_WIDTH, "SIZE", "STATUS")]
    for profile_id, profile in profiles.items():
        name = profile.get("alias") or profile_id
        size = local_sizes.get(profile_id)
        size_text = "missing" if size is None else human_size(size)
        selected_marker = "* " if profile_id == selected_id else "  "
        running_marker = "▶ " if running and running.get("id") == profile_id else ""
        lines.append("%s%s%-*s  %*s  %s" % (
            selected_marker, running_marker, NAME_WIDTH, name,
            SIZE_WIDTH, size_text, _status(profile_id, selected_id, running)))
    if running and running.get("id") != selected_id:
        lines.append("")
        lines.append("warning: selected model is not the running model")
    if selected_failure:
        lines.append("")
        lines.append(selected_failure)
    return "\n".join(lines)


def format_status(selected, running, listener, endpoint):
    lines = ["Selected: %s" % selected]
    if running:
        state = "ready" if running.get("ready") else "loading"
        lines.append("Service:  RUNNING · %s · PID %s · up %s" % (
            state, running.get("pid"), running.get("uptime", "?")))
    else:
        lines.append("Service:  stopped")
    lines.append("Endpoint: %s" % endpoint)
    if listener and (not running or running.get("pid") != listener):
        lines.append("warning: port %s is held by an unmanaged PID %s"
                     % (endpoint.rsplit(":", 1)[-1].split("/")[0], listener))
    return "\n".join(lines)
