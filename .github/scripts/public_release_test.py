"""Exercise the public release guard against real commit histories."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

# unittest discovery adds this script directory to sys.path.
from check_public_release import STUB_PATH, check_release, git  # ty: ignore[unresolved-import]


class PublicReleaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        git(self.repo, "init", "--quiet")
        git(self.repo, "config", "user.name", "Release guard test")
        git(self.repo, "config", "user.email", "test@example.invalid")
        self.stub = self.repo / STUB_PATH
        self.stub.parent.mkdir(parents=True)
        self.original = (Path(__file__).resolve().parents[2] / STUB_PATH).read_bytes()
        self.stub.write_bytes(self.original)
        self.base = self.commit()

    def commit(self):
        git(self.repo, "add", ".")
        git(self.repo, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "test")
        return git(self.repo, "rev-parse", "HEAD").decode().strip()

    def test_public_stub_and_unrelated_changes_pass(self):
        (self.repo / "unrelated.py").write_text("value = 1\n")
        self.commit()
        self.assertEqual(check_release(self.repo, base=self.base), [])

    def test_changed_implementation_fails(self):
        self.stub.write_bytes(
            self.original.replace(b"raise NotImplementedError(_REMOVED_MESSAGE)", b"return 0")
        )
        self.commit()
        errors = check_release(self.repo, base=self.base)
        self.assertTrue(any("differs" in error for error in errors))

    def test_missing_stub_fails(self):
        self.stub.unlink()
        self.commit()
        self.assertTrue(
            any("missing" in error for error in check_release(self.repo, base=self.base))
        )

    def test_additional_dra_source_fails(self):
        (self.stub.parent / "new_DRA_model.py").write_text("value = 1\n")
        self.commit()
        errors = check_release(self.repo, base=self.base)
        self.assertTrue(any("additional DRA source" in error for error in errors))

    def test_leak_removed_in_later_commit_still_fails(self):
        self.stub.write_bytes(self.original + b"\nprivate_value = 1\n")
        leaked = self.commit()
        self.stub.write_bytes(self.original)
        self.commit()
        errors = check_release(self.repo, base=self.base)
        self.assertTrue(any(error.startswith(leaked[:12]) for error in errors))
        self.assertEqual(check_release(self.repo), [])

    def test_unknown_base_fails_closed(self):
        with self.assertRaises(subprocess.CalledProcessError):
            check_release(self.repo, base="f" * 40)


if __name__ == "__main__":
    unittest.main()
