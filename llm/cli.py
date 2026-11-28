"""Command line interface for llm (lmm)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from . import (config, diagnose, hardware, hfcache, host, hub, migrate, render,
               service, update)

_subparsers = {}
# Legacy download location. `get` writes the Hugging Face cache instead, but
# keep the name so callers (and tests) can still refer to the old path.
MODELS_DIR = Path.home() / ".local" / "share" / "llm" / "models"


def _local_size(profile):
    path = profile.get("path")
    if not path:
        return None
    model = Path(path)
    parts = (hfcache.split_set(model.name)
             or hfcache.split_set(profile.get("filename") or ""))
    try:
        if parts is not None:
            stem, _index, total = parts
            shards = model.parent.glob("%s-*-of-%05d.gguf" % (stem, total))
            size = sum(shard.stat().st_size for shard in shards)
            return size if size > 0 else None
        size = model.stat().st_size
        return size if size > 0 else None
    except OSError:
        return None


def _running_info(profiles):
    pid = service.running()
    if not pid:
        return None
    actual = service.identity(pid)
    profile = next((item for item in profiles.values()
                    if item.get("alias") and item["alias"] in actual), None)
    return {
        "id": profile["id"] if profile else None,
        "ready": service.healthy(),
        "pid": pid,
        "uptime": service.uptime(pid) or "?",
    }


def cmd_list(args):
    profiles = config.PROFILES
    selected = config.selected_profile()
    sizes = {profile_id: _local_size(profile) for profile_id, profile in profiles.items()}
    running = _running_info(profiles)
    failure = None if (running and running["ready"]) else service.last_failure()
    for warning in config.PROFILE_WARNINGS:
        print("warning: " + warning, file=sys.stderr)
    print(render.format_list(profiles, selected["id"], sizes, running, failure))
    print()
    print("Service %s.   Endpoint %s/v1"
          % ("running" if running else "stopped", config.URL))
    return 0


def cmd_status(args):
    selected = config.selected_profile()
    running = _running_info(config.PROFILES)
    listener = None if running else service.listener_pid()
    print(render.format_status(selected["id"], running, listener, config.URL + "/v1"))
    if not running:
        failure = service.last_failure()
        if failure:
            print()
            print("Last failure: " + failure)
    return 0


def cmd_select(args):
    profile = config.resolve_model(args.model)
    with service.locked():
        current = config.selected_profile()
        if current["id"] != profile["id"] and service.running():
            service.stop()
        config.select_profile(profile["id"])
    print("Selected %s. Start it with: lmm start" % profile["id"])
    return 0


def cmd_use(args):
    profile = config.resolve_model(args.model)
    with service.locked():
        was_running = service.running() is not None
        config.select_profile(profile["id"])
        if was_running:
            service.stop()
            pid, created, started = service.start(profile["id"])
    if not was_running:
        print("Selected %s. Start it with: lmm start" % profile["id"])
        return 0
    print("Starting %s (PID %s) at %s/v1" % (started["alias"], pid, config.URL))
    service.wait_ready(pid, created, started)
    print("%s ready." % started["id"])
    return 0


def cmd_start(args):
    with service.locked():
        pid, created, profile = service.start(args.model)
    if args.no_wait:
        print("Starting %s (PID %s) at %s/v1" % (profile["alias"], pid, config.URL))
        return 0
    print("Starting %s (PID %s); waiting for health..." % (profile["alias"], pid))
    service.wait_ready(pid, created, profile)
    print("%s ready at %s/v1" % (profile["id"], config.URL))
    return 0


def cmd_stop(args):
    with service.locked():
        service.stop()
    print("Service stopped.")
    return 0


def cmd_restart(args):
    with service.locked():
        service.stop()
        pid, created, profile = service.start()
    print("Starting %s (PID %s); waiting for health..." % (profile["alias"], pid))
    service.wait_ready(pid, created, profile)
    print("%s ready at %s/v1" % (profile["id"], config.URL))
    return 0


def cmd_logs(args):
    if not service.LOG.exists():
        print("No log yet: %s" % service.LOG)
        return 0
    command = ["tail", "-n", str(args.lines)]
    if args.follow:
        command.append("-f")
    command.append(str(service.LOG))
    try:
        return subprocess.call(command)
    except OSError as exc:
        print("llm: could not read the log: %s" % exc, file=sys.stderr)
        return 1


def _progress(done, total, elapsed):
    speed = done / elapsed if elapsed > 0 else 0
    if total:
        percent = 100.0 * done / total
        remaining = (total - done) / speed if speed else 0
        print("\r  %5.1f%%  %s / %s  %s/s  ETA %ds"
              % (percent, render.human_size(done), render.human_size(total),
                 render.human_size(int(speed)), remaining),
              end="", file=sys.stderr, flush=True)
    else:
        print("\r  %s  %s/s" % (render.human_size(done), render.human_size(int(speed))),
              end="", file=sys.stderr, flush=True)


def cmd_search(args):
    results = hub.search_models(
        args.query, limit=args.limit * 3 if not args.all else args.limit)
    shown, hidden = hub.filter_search(results, include_all=args.all)
    shown = shown[:args.limit]
    if not shown:
        suffix = "" if args.all else " (try --all)"
        print("No models found for %r%s." % (args.query, suffix))
        return 0
    print("%-48s %10s %6s  %s" % ("REPO", "DOWNLOADS", "LIKES", "NOTES"))
    for result in shown:
        note = "gated" if result["gated"] else ""
        print("%-48s %10s %6d  %s"
              % (result["id"], format(result["downloads"], ","), result["likes"], note))
    if hidden:
        print("\n%d non-GGUF result(s) hidden; use --all to include them." % hidden)
    return 0


def _registered_model(profiles, repo, filename):
    target = Path(filename)
    for profile in profiles.values():
        if profile.get("repo") != repo:
            continue
        stored = profile.get("filename")
        if stored is not None:
            if Path(stored) == target:
                return profile
        elif Path(profile.get("path", "")).name == target.name:
            return profile
    return None


def cmd_get(args):
    repo, selector = hub.parse_spec(args.spec)
    if args.file and selector:
        raise hub.HubError("Use either :QUANT or --file, not both")
    commit, files = hub.repo_info(repo, revision=args.revision)
    try:
        chosen = hub.select_file(files, args.file or selector)
    except hub.HubError as exc:
        print("llm: %s" % exc, file=sys.stderr)
        for candidate in exc.candidates:
            size = (render.human_size(candidate["size"])
                    if candidate.get("size") else "unknown size")
            print("  %-52s %s" % (candidate["name"], size), file=sys.stderr)
        return 1

    existing = _registered_model(config.PROFILES, repo, chosen["name"])
    if existing:
        model_id = existing["id"]
        alias = existing["alias"]
        if args.name and config.normalize_alias(args.name) != alias:
            raise config.ModelError("Already registered as %r; run `lmm list`" % alias)
    else:
        taken_ids = set(config.PROFILES)
        taken_aliases = {str(profile.get("alias") or "")
                         for profile in config.PROFILES.values()}
        model_id, alias = config.derive_model_names(
            repo, chosen["name"], taken_ids, taken_aliases)
        if args.name:
            alias = config.normalize_alias(args.name)
            if not alias:
                raise config.ModelError("--name must contain a letter or digit")
            if alias in {value.lower() for value in taken_aliases if value}:
                raise config.ModelError("Alias %r is already in use" % alias)

    size_text = render.human_size(chosen["size"]) if chosen["size"] else "unknown size"
    if not existing or args.force:
        context = (args.context if args.context is not None
                   else (existing.get("context") if existing else 8192))
        warnings = hardware.preflight(
            chosen["size"], context, models_dir=hfcache.cache_root())
        if warnings:
            print("<!> Model %s may not fit on this machine <!>" % chosen["name"],
                  file=sys.stderr, flush=True)
            for warning in warnings:
                print("  " + warning, file=sys.stderr)
            if not args.force:
                print("Use --force to download anyway; the model may fail to load.",
                      file=sys.stderr)
                return 1
            print("Proceeding because --force was given; the model may fail to load.",
                  file=sys.stderr)
    if existing:
        print("Already registered as %s; verifying %s"
              % (alias, chosen["name"]), file=sys.stderr, flush=True)
    else:
        print("Downloading %s (%s) from %s" % (chosen["name"], size_text, repo),
              file=sys.stderr, flush=True)
    blob_dir = hfcache.repo_dir(repo) / "blobs"
    try:
        blob = hub.download_blob(
            repo, chosen, blob_dir, revision=args.revision,
            force=args.force, progress=_progress)
    finally:
        print("", file=sys.stderr, flush=True)

    snapshot = hfcache.link_snapshot(repo, chosen["name"], commit, blob)
    hfcache.write_ref(repo, args.revision, commit)

    size = chosen["size"] or blob.stat().st_size
    sha256 = chosen["sha256"] or hub.sha256_file(blob)
    if existing:
        profile = dict(existing)
        profile.pop("directory", None)
        profile.update({"repo": repo, "filename": chosen["name"],
                        "revision": args.revision, "path": str(snapshot)})
        if args.context is not None:
            profile["context"] = args.context
    else:
        profile = hfcache.downloaded_profile(
            repo, chosen["name"], args.revision, commit, model_id, alias,
            context=(args.context if args.context is not None else 8192),
            sampling=config.DEFAULT_SAMPLING)
    if args.requires_llama_cpp:
        profile["min_llama_cpp"] = args.requires_llama_cpp
    if existing and existing["id"] in config.BUILTIN_PROFILES:
        manifest = hub.write_manifest(
            config.STATE / "manifests" / (model_id + ".json"), repo,
            chosen["name"], args.revision, size, sha256, commit=commit)
        print("Already present as built-in %s (%s); verified %s"
              % (alias, model_id, chosen["name"]))
        print("Manifest: %s" % manifest)
        return 0
    config.register_model(profile)
    manifest = hub.write_manifest(
        config.STATE / "manifests" / (model_id + ".json"), repo,
        chosen["name"], args.revision, size, sha256, commit=commit)
    print("Downloaded %s (%s) to %s"
          % (chosen["name"], render.human_size(size), snapshot))
    print("Registered %s (%s). Start it with: lmm use %s" % (alias, model_id, alias))
    print("Manifest: %s" % manifest)
    return 0


def cmd_migrate_cache(args):
    entries = migrate.plan()
    if not entries:
        print("No legacy model directories found.")
        return 0

    errors = 0
    migrated = 0
    already = 0
    missing = 0
    for entry in entries:
        heading = "%s \u2192 %s @ %s" % (
            entry["source"], entry["repo"], entry["commit"])
        if (entry.get("status") == "missing"
                and not migrate.cache_present(entry)):
            print(heading)
            print("  missing      %d file(s) not found" % len(entry["files"]))
            missing += 1
            continue
        result = migrate.migrate(entry, dry_run=args.dry_run,
                                 verify=args.verify, keep_old=args.keep_old)
        print(heading)
        for item in result["files"]:
            print("  %-13s %s" % (item["status"], item["filename"]))
        if result["status"] == "error":
            errors += 1
            for item in result["files"]:
                if item.get("message"):
                    print("  error: " + item["message"], file=sys.stderr)
        elif result["status"] in ("dry-run", "migrated"):
            migrated += 1
        elif result["status"] == "already present":
            already += 1
        else:
            missing += 1
        if result["leftover"]:
            print("  kept old directory; %d unrelated file(s) remain"
                  % len(result["leftover"]))
        if result.get("removal_error"):
            print("  warning: " + result["removal_error"], file=sys.stderr)

    print()
    if args.dry_run:
        summary = ("Dry run: %d model(s) would migrate, %d already present"
                   % (migrated, already))
    else:
        summary = ("Migrated %d model(s); %d already present"
                   % (migrated, already))
    if missing:
        summary += ", %d missing" % missing
    print(summary + ".")
    if errors:
        print("%d error(s)." % errors, file=sys.stderr)
        return 1
    return 0


def cmd_doctor(args):
    version = diagnose.llama_server_version(config.LLAMA_SERVER)
    print("llama-server: %s" % (version["raw"] or "not found at " + config.LLAMA_SERVER))
    package = diagnose.llama_cpp_package_status()
    if package:
        print(("Homebrew:     " if host.IS_MACOS else "Package:      ") + package)
    machine = hardware.detect_hardware(models_dir=hfcache.cache_root())
    budget = hardware.memory_budget(machine.total_ram, machine.unified,
                                    machine.available_ram)
    details = "%s, %s RAM" % (machine.chip, render.human_size(machine.total_ram))
    if machine.available_ram > 0:
        details += ", %s available" % render.human_size(machine.available_ram)
    if budget:
        details += ", usable ~%s" % render.human_size(budget)
    print("Hardware:     " + details)
    if not os.access(config.LSOF, os.X_OK):
        print("warning: lsof not found; port ownership checks are limited")
    if machine.disk_free is not None:
        print("Disk free:    %s" % render.human_size(machine.disk_free))

    print()
    print("Models:")
    installed = version["version"]
    for profile_id, profile in config.PROFILES.items():
        size = _local_size(profile)
        name = profile.get("alias") or profile_id
        state = "missing" if size is None else render.human_size(size)
        note = ""
        required = profile.get("min_llama_cpp")
        if required:
            met = installed is not None and diagnose.version_at_least(installed, required)
            note = " \u00b7 needs llama.cpp >= %s%s" % (
                required, "" if met else " (NOT MET)")
        print("  %-30s %10s%s" % (name, state, note))

    print()
    pid = service.running()
    if pid:
        print("Service:      RUNNING \u00b7 PID %s \u00b7 %s"
              % (pid, "ready" if service.healthy() else "loading"))
    else:
        print("Service:      stopped")
    listener = service.listener_pid() if not pid else None
    if listener:
        print("warning: port %d is held by an unmanaged PID %s" % (config.PORT, listener))

    try:
        config.STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        writable = os.access(config.STATE, os.W_OK)
    except OSError:
        writable = False
    print("State:        %s (%s)" % (config.STATE, "writable" if writable else "NOT writable"))
    print("Log:          %s (%s)"
          % (service.LOG, render.human_size(service.log_size())))
    failure = service.last_failure()
    if failure:
        print()
        print("Last failure: " + failure)
    return 0


def cmd_update(args):
    checkout = Path(args.dir).expanduser() if args.dir else None
    print("Updating llm from %s" % update.REPO_URL)
    path = update.update(checkout=checkout)
    print("Updated llm from %s" % path)
    return 0


def cmd_version(args):
    from . import __version__, LATEST_VERSION
    program = _program_name()
    print("%s %s" % (program, __version__))
    if __version__ == LATEST_VERSION:
        print("last version is %s. you are up to date." % LATEST_VERSION)
    else:
        print("last version is %s. run `%s update` to update."
              % (LATEST_VERSION, program))
    return 0


def cmd_help(args):
    parser = build_parser()
    if args.topic:
        sub = _subparsers.get(args.topic)
        if sub is None:
            print("Unknown command %r; try `lmm help`" % args.topic, file=sys.stderr)
            return 1
        sub.print_help()
        return 0
    parser.print_help()
    return 0


def _program_name():
    name = os.path.basename(sys.argv[0] or "")
    return name if name in ("llm", "lmm") else "llm"


def build_parser(prog=None):
    global _subparsers
    program = prog or _program_name()
    parser = argparse.ArgumentParser(
        prog=program,
        description="Manage the persistent local llama.cpp model server.",
        epilog=("examples:\n"
                "  %s list\n"
                "  %s search inkling\n"
                "  %s get unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL\n"
                "  %s use inkling\n"
                "  %s logs -f\n"
                "  %s doctor") % ((program,) * 6),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _subparsers = {}

    def add(name, handler, help_text, **kwargs):
        command = sub.add_parser(name, help=help_text, description=help_text, **kwargs)
        command.set_defaults(handler=handler)
        _subparsers[name] = command
        return command

    add("list", cmd_list, "show models and which one is running")
    add("status", cmd_status, "show selected model, service health and endpoint")
    command = add("use", cmd_use, "select a model; restart the service if it is running")
    command.add_argument("model")
    command = add("start", cmd_start, "start the selected model")
    command.add_argument("model", nargs="?")
    command.add_argument("--no-wait", action="store_true")
    add("stop", cmd_stop, "stop the model service")
    add("restart", cmd_restart, "restart the model service")
    command = add("logs", cmd_logs, "tail the server log")
    command.add_argument("-n", "--lines", type=int, default=100)
    command.add_argument("-f", "--follow", action="store_true")
    command = add("search", cmd_search, "search Hugging Face for GGUF models")
    command.add_argument("query")
    command.add_argument("--limit", type=int, default=10)
    command.add_argument("--all", action="store_true",
                         help="include non-GGUF repositories")
    command = add("get", cmd_get, "download and register a GGUF model")
    command.add_argument("spec", help="repo or repo:QUANT")
    command.add_argument("--file", help="exact GGUF filename")
    command.add_argument("--revision", default="main")
    command.add_argument("--name", help="custom alias")
    command.add_argument("--context", type=int, default=None)
    command.add_argument("--requires-llama-cpp", metavar="VERSION")
    command.add_argument("--force", action="store_true",
                         help="replace an existing mismatched download")
    command = add("update", cmd_update,
                  "fetch the latest source, build, and install")
    command.add_argument("--dir", help="checkout directory")
    add("version", cmd_version, "show the installed version and whether it is current")
    add("doctor", cmd_doctor, "check llama.cpp, models, port and last failure")
    command = add("migrate-cache", cmd_migrate_cache,
                  "move legacy model files into the Hugging Face cache")
    command.add_argument("--dry-run", action="store_true",
                         help="show the plan without changing anything")
    command.add_argument("--verify", action="store_true",
                         help="hash files before and after migrating")
    command.add_argument("--keep-old", action="store_true",
                         help="keep the old model directories")
    command = add("help", cmd_help, "show this help or help for one command")
    command.add_argument("topic", nargs="?")

    command = sub.add_parser("select",
                             help="select a model without starting it (used by graphify)",
                             description="Select a model without starting it (used by graphify).")
    command.add_argument("model")
    command.set_defaults(handler=cmd_select)
    _subparsers["select"] = command
    return parser


def main(argv=None):
    try:
        config.migrate_state()
    except OSError:
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return handler(args)
    except (config.ModelError, service.ModelLoadError, RuntimeError, ValueError, OSError) as exc:
        print("llm: %s" % exc, file=sys.stderr)
        return 1
