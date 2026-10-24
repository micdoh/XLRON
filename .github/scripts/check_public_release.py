"""Keep the withheld DRA module out of public GitHub changes.

Check every incoming commit, including code added and then removed in a PR.
This deliberately permits only the published stub in its current form. Changing
that interface requires an explicit review of this guard when publication allows
it. DRA configuration and documentation already in the public repo are allowed.
This is an accidental-disclosure guard for known DRA source paths, not a general
detector for private code copied under unrelated names.
"""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess


STUB_PATH = "xlron/environments/gn_model/isrs_gn_model_dra.py"
STUB_SHA256 = "ca1750fb3aab7c87507b0b934f4713b8207a17b1a43e54faf6fb6392b584fb5d"
DRA_PATH = re.compile(r"(?:^|[/_.-])dra(?:[/_.-]|$)", re.IGNORECASE)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def check_release(repo: Path, head: str = "HEAD", base: str | None = None) -> list[str]:
    """Return diagnostics without printing any potentially private file content."""
    head = git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").decode().strip()
    revisions = [head]
    if base and set(base) != {"0"}:
        base = git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
        revisions = [base, *git(repo, "rev-list", "--reverse", f"{base}..{head}").decode().split()]
    errors = []
    for revision in dict.fromkeys(revisions):
        paths = git(repo, "ls-tree", "-r", "--name-only", "-z", revision).decode().split("\0")
        if STUB_PATH not in paths:
            errors.append(f"{revision[:12]}: public DRA stub is missing")
        else:
            source = git(repo, "show", f"{revision}:{STUB_PATH}")
            if hashlib.sha256(source).hexdigest() != STUB_SHA256:
                errors.append(f"{revision[:12]}: DRA module differs from the approved public stub")
        for path in paths:
            if (
                path != STUB_PATH
                and path.lower().endswith((".py", ".ipynb"))
                and DRA_PATH.search(path)
            ):
                errors.append(f"{revision[:12]}: additional DRA source path: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base or previous push revision")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        errors = check_release(Path(__file__).resolve().parents[2], args.head, args.base)
    except subprocess.CalledProcessError:
        print("Public release check failed: could not inspect the requested Git history.")
        return 1
    if errors:
        print("Public release check failed. Keep the DRA implementation on private GitLab.")
        print("\n".join(errors))
        return 1
    print("Public release check passed: only the approved DRA stub is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
