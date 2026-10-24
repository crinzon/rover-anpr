"""Safety checks for the project commands and version hook; no real commits."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigTest(unittest.TestCase):
    def test_purge_is_project_scoped_and_preserves_data(self):
        text = (ROOT / "Makefile").read_text()
        recipe = text.split("purge:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("$(STACK) down", recipe)
        self.assertIn("--project-name rover-anpr", text)
        for forbidden in ("--volumes", " -v", "prune", "rm -rf"):
            self.assertNotIn(forbidden, recipe)

    @unittest.skipUnless(shutil.which("make"), "GNU Make unavailable")
    def test_install_run_stop_with_fake_compose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / "Makefile", root / "Makefile")
            (root / ".env.example").write_text("VIDEO_SOURCE=example\n")
            fake = root / "compose"
            fake.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
            )
            fake.chmod(0o755)
            log = root / "commands"
            env = dict(os.environ, DOCKER_LOG=str(log))
            for target in ("install", "run", "stop"):
                if target == "run":
                    for name in ("plate_detector.pt", "plate_recognizer.pt"):
                        (root / "checkpoints" / name).write_bytes(b"fake")
                result = subprocess.run(
                    ["make", target, f"COMPOSE={fake}"],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            original = (root / ".env").read_bytes()
            subprocess.run(
                ["make", "install", f"COMPOSE={fake}"],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
            )
            self.assertEqual((root / ".env").read_bytes(), original)
            commands = log.read_text().splitlines()
            self.assertEqual(len(commands), 4)
            self.assertTrue(
                all("--project-name rover-anpr" in c for c in commands)
            )
            self.assertIn("up -d --no-build", commands[1])

    def test_hook_increment_retry_and_invalid_version_without_commits(self):
        # The fake git models index state; the real repository is never used.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "git"
            fake.write_text(
                "#!/bin/sh\n"
                'case "$*" in\n'
                ' "rev-parse --show-toplevel") pwd;;\n'
                ' "diff --cached --quiet") exit 1;;\n'
                ' "diff --cached --quiet -- VERSION") '
                "[ ! -f staged-version ];;\n"
                ' "diff --quiet -- VERSION") exit 0;;\n'
                ' "add -- VERSION") touch staged-version;;\n'
                " *) exit 2;;\n"
                "esac\n"
            )
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}")
            version = root / "VERSION"
            version.write_text("0.1.3\n")
            hook = ROOT / ".githooks" / "pre-commit"
            for _ in range(2):
                result = subprocess.run(
                    ["sh", str(hook)], cwd=root, env=env, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(version.read_text(), "0.1.4\n")
            (root / "staged-version").unlink()
            version.write_text("0.1.4.5\n")
            result = subprocess.run(
                ["sh", str(hook)], cwd=root, env=env, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(version.read_text(), "0.1.4.5\n")
