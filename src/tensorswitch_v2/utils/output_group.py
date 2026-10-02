"""
Which Unix group owns the files TensorSwitch writes.

- Cluster jobs: the group named like the LSF project they are billed to (when the
  user belongs to it), by running the job under ``sg <group>``. Every file the job
  creates then gets that group from the start.
- Local runs (no project): the group of the folder the output goes into.

Only groups the user already belongs to are used. ``sg`` asks for a password
when given any other group, which would hang a batch job, so those are skipped.
"""

import grp
import os
import shlex
import warnings
from typing import List, Optional

APPLIED_ENV = "TENSORSWITCH_GROUP_APPLIED"
_warned = set()


def _member_gids() -> set:
    return set(os.getgroups()) | {os.getgid(), os.getegid()}


def project_group(project: Optional[str]) -> Optional[str]:
    """Name of the Unix group matching the LSF project, if it exists and the user is in it."""
    if not project:
        return None
    try:
        entry = grp.getgrnam(project)
    except KeyError:
        return None
    return entry.gr_name if entry.gr_gid in _member_gids() else None


def wrap_for_project(project: Optional[str], argv: List[str]) -> List[str]:
    """The tail of a bsub command, run under the project's group when there is one.

    ``argv`` is the command the job runs (e.g. ``["/bin/bash", "-c", "<command>"]``).
    Without a usable group it is returned unchanged, with a one-time warning.

    LSF re-quotes each argument it is given and cannot carry one that mixes single
    and double quotes: the job then runs nothing yet is reported as done. So the
    command is passed as ONE level of shell text (``sg <group> -c "export ...; <cmd>"``)
    instead of nested ``bash -c '...'`` quoting, and if it would still mix both quote
    types the wrap is skipped with a warning.
    """
    group = project_group(project)
    if group is None:
        if project and project not in _warned:
            _warned.add(project)
            warnings.warn(
                f"No Unix group named {project!r} that you belong to; the job will write files "
                f"with its default group (or the output folder's group) instead of the project's.",
                stacklevel=2,
            )
        return argv
    inner = argv[2] if len(argv) == 3 and argv[:2] == ["/bin/bash", "-c"] else shlex.join(argv)
    command = f"export {APPLIED_ENV}=1; {inner}"
    if "'" in command and '"' in command:
        warnings.warn(
            "The job command contains both single and double quotes, which LSF cannot pass on safely; "
            "the job will use its default group instead of the project's.",
            stacklevel=2,
        )
        return argv
    return ["sg", group, "-c", command]


def wrap_script_line(project: Optional[str], argv: List[str]) -> str:
    """Same as wrap_for_project, as one shell-quoted string (for generated scripts)."""
    return shlex.join(wrap_for_project(project, argv))


def _existing_ancestor(path: str) -> str:
    path = os.path.abspath(path)
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def apply_parent_group(output_path: str) -> int:
    """Give a finished local output the group of the folder it was written into.

    Does nothing inside a cluster job that already ran under the project's group,
    when the user is not in the parent's group, or when the group already matches.
    Files the user does not own are skipped. Returns the number of entries changed.
    """
    if os.environ.get(APPLIED_ENV) or not os.path.exists(output_path):
        return 0
    parent = _existing_ancestor(os.path.dirname(os.path.abspath(output_path)))
    try:
        gid = os.stat(parent).st_gid
    except OSError:
        return 0
    if gid not in _member_gids():
        return 0

    changed = 0

    def fix(path):
        nonlocal changed
        try:
            if os.lstat(path).st_gid != gid:
                os.chown(path, -1, gid, follow_symlinks=False)
                changed += 1
        except OSError:
            pass

    fix(output_path)
    for root, dirs, files in os.walk(output_path):
        for name in dirs + files:
            fix(os.path.join(root, name))
    return changed
