"""Plain-text renderers for model tables and service status."""

NAME_WIDTH = 32
SIZE_WIDTH = 10
CONTEXT_WIDTH = 7
MODEL_WIDTH = 28
FILE_WIDTH = 10
NEEDED_WIDTH = 11


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return "%d B" % int(value)
            return "%.1f %s" % (value, unit)
        value /= 1024


def short_context(n):
    if n is None:
        return ""
    if n % 1024 == 0:
        return "%dK" % (n // 1024)
    return "{:,}".format(n)


def _status(profile_id, selected_id, running):
    if running and running.get("id") == profile_id:
        if running.get("ready"):
            return "RUNNING · ready · PID %s · up %s" % (
                running.get("pid"), running.get("uptime", "?"))
        return "RUNNING · loading · PID %s" % running.get("pid")
    if profile_id == selected_id:
        return "selected · not running"
    return "downloaded · not running"


def format_list(profiles, selected_id, local_sizes, running, selected_failure, contexts=None):
    lines = ["  %-*s  %*s  %*s  %s"
             % (NAME_WIDTH, "MODEL", SIZE_WIDTH, "SIZE", CONTEXT_WIDTH, "CONTEXT", "STATUS")]
    for profile_id, profile in profiles.items():
        name = profile.get("alias") or profile_id
        size = local_sizes.get(profile_id)
        size_text = "missing" if size is None else human_size(size)
        context = contexts.get(profile_id) if contexts else None
        context_text = short_context(context)
        selected_marker = "* " if profile_id == selected_id else "  "
        running_marker = "▶ " if running and running.get("id") == profile_id else ""
        lines.append("%s%s%-*s  %*s  %*s  %s" % (
            selected_marker, running_marker, NAME_WIDTH, name,
            SIZE_WIDTH, size_text, CONTEXT_WIDTH, context_text,
            _status(profile_id, selected_id, running)))
    if running and running.get("id") != selected_id:
        lines.append("")
        lines.append("warning: selected model is not the running model")
    if selected_failure:
        lines.append("")
        lines.append(selected_failure)
    return "\n".join(lines)


def format_status(selected, running, listener, endpoint, context=None, server_context=None):
    lines = ["Selected: %s" % selected]
    if context is not None:
        lines.append("Context: %s" % short_context(context))
    if running:
        state = "ready" if running.get("ready") else "loading"
        lines.append("Service:  RUNNING · %s · PID %s · up %s" % (
            state, running.get("pid"), running.get("uptime", "?")))
    else:
        lines.append("Service:  stopped")
    lines.append("Endpoint: %s" % endpoint)
    if server_context is not None:
        lines.append("server %s" % short_context(server_context))
        if context is not None and context != server_context:
            lines.append("warning: server context %s differs from configured %s"
                         % (short_context(server_context), short_context(context)))
    if listener and (not running or running.get("pid") != listener):
        lines.append("warning: port %s is held by an unmanaged PID %s"
                     % (endpoint.rsplit(":", 1)[-1].split("/")[0], listener))
    return "\n".join(lines)


def _context_cell(row):
    old = row.get("old")
    new = row.get("new")
    if old is not None and new is not None and old != new:
        cell = "%s \u2192 %s" % (short_context(old), short_context(new))
    else:
        cell = short_context(new)
    if row.get("approximate"):
        cell += " approx"
    native = row.get("native")
    if native is not None and new is not None and new > native:
        cell += " \u00b7 beyond native %s" % short_context(native)
    return cell


def runtime_summary(runtime):
    """Summarize a runtime block as `q8_0 KV · 2048/512 · fit 2048 · cache-reuse 256`."""
    runtime = runtime or {}
    kv_k = runtime.get("kv_type_k", "f16")
    kv_v = runtime.get("kv_type_v", "f16")
    kv = kv_k if kv_k == kv_v else "%s/%s" % (kv_k, kv_v)
    parts = ["%s KV" % kv]
    parts.append("%s/%s" % (runtime.get("batch", 512), runtime.get("ubatch", 256)))
    if runtime.get("fit"):
        parts.append("fit %s" % runtime.get("fit_target", 2048))
    else:
        parts.append("no-fit")
    parts.append("cache-reuse %s" % runtime.get("cache_reuse", 0))
    return " \u00b7 ".join(parts)


def _kv_cell(row):
    return row.get("kv") or ""


def _batch_cell(row):
    batch = row.get("batch")
    ubatch = row.get("ubatch")
    if batch is None or ubatch is None:
        return ""
    return "%s/%s" % (batch, ubatch)


def format_apply(rows):
    """Render the `apply` table: MODEL, FILE, CONTEXT, KV, BATCH, EST. NEEDED, ACTION."""
    rows = list(rows)
    context_cells = [_context_cell(row) for row in rows]
    kv_cells = [_kv_cell(row) for row in rows]
    batch_cells = [_batch_cell(row) for row in rows]
    context_width = max([len("CONTEXT")] + [len(cell) for cell in context_cells])
    kv_width = max([len("KV")] + [len(cell) for cell in kv_cells])
    batch_width = max([len("BATCH")] + [len(cell) for cell in batch_cells])
    lines = ["  %-*s  %*s  %-*s  %-*s  %-*s  %*s  %s" % (
        MODEL_WIDTH, "MODEL", FILE_WIDTH, "FILE", context_width, "CONTEXT",
        kv_width, "KV", batch_width, "BATCH",
        NEEDED_WIDTH, "EST. NEEDED", "ACTION")]
    for row, context_cell, kv_cell, batch_cell in zip(
            rows, context_cells, kv_cells, batch_cells):
        size = row.get("size")
        file_cell = "missing" if size is None else human_size(size)
        required = row.get("required")
        needed_cell = "" if required is None else human_size(required)
        lines.append("  %-*s  %*s  %-*s  %-*s  %-*s  %*s  %s" % (
            MODEL_WIDTH, row.get("model", ""), FILE_WIDTH, file_cell,
            context_width, context_cell, kv_width, kv_cell, batch_width,
            batch_cell, NEEDED_WIDTH, needed_cell, row.get("action", "")))
    return "\n".join(lines)


def _analysis_size_cell(value):
    return human_size(value) if value is not None else "-"


def _analysis_kv_cell(value):
    return "%.2f GiB" % (value / (1024 ** 3)) if value is not None else "-"


def _analysis_saving_cell(saving):
    return "%.1f%%" % (saving * 100) if saving is not None else "-"


def _analysis_fit_cell(fits):
    if fits is True:
        return "yes"
    if fits is False:
        return "no"
    return "-"


def _analysis_context_cell(row, context):
    cell = short_context(context)
    native = row.get("native")
    if native is not None and context is not None and context > native:
        cell += " \u00b7 beyond native %s" % short_context(native)
    return cell


def format_analysis(rows, context):
    """Render the `analyze` table: MODEL, SIZE, CONTEXT, f16, q8_0, q4_0, SAVE, FIT."""
    rows = list(rows)
    names = [row.get("alias") or row.get("model") or "" for row in rows]
    sizes = [_analysis_size_cell(row.get("size")) for row in rows]
    contexts = [_analysis_context_cell(row, context) for row in rows]
    f16s = [_analysis_kv_cell(row.get("f16")) for row in rows]
    q8s = [_analysis_kv_cell(row.get("q8_0")) for row in rows]
    q4s = [_analysis_kv_cell(row.get("q4_0")) for row in rows]
    savings = [_analysis_saving_cell(row.get("saving")) for row in rows]
    fits = [_analysis_fit_cell(row.get("fits")) for row in rows]

    def width(title, cells):
        return max([len(title)] + [len(cell) for cell in cells])

    name_width = width("MODEL", names)
    size_width = width("SIZE", sizes)
    context_width = width("CONTEXT", contexts)
    f16_width = width("f16", f16s)
    q8_width = width("q8_0", q8s)
    q4_width = width("q4_0", q4s)
    save_width = width("SAVE", savings)
    fit_width = width("FIT", fits)

    lines = ["  %-*s  %*s  %-*s  %*s  %*s  %*s  %*s  %-*s" % (
        name_width, "MODEL", size_width, "SIZE", context_width, "CONTEXT",
        f16_width, "f16", q8_width, "q8_0", q4_width, "q4_0",
        save_width, "SAVE", fit_width, "FIT")]
    for name, size, ctx, f16, q8, q4, saving, fit in zip(
            names, sizes, contexts, f16s, q8s, q4s, savings, fits):
        lines.append("  %-*s  %*s  %-*s  %*s  %*s  %*s  %*s  %-*s" % (
            name_width, name, size_width, size, context_width, ctx,
            f16_width, f16, q8_width, q8, q4_width, q4,
            save_width, saving, fit_width, fit))
    return "\n".join(lines)
