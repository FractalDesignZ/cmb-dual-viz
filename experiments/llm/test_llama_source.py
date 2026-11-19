from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


GUARD = Path(__file__).resolve().parents[2]/'bridge/llama_observer/VerifyLlamaSource.cmake'


@unittest.skipUnless(shutil.which('cmake') and shutil.which('git'), 'requires CMake and Git')
class LlamaSourceTests(unittest.TestCase):
    def test_pinned_checkout_rejects_staged_and_unstaged_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'llama'
            source.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-c', 'core.autocrlf=false', *args], cwd=source, stderr=subprocess.STDOUT).decode().strip()
            git('init')
            (source/'llama.h').write_text('original\n')
            git('add', 'llama.h')
            git('-c', 'user.name=CMB test', '-c', 'user.email=cmb-test@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-m', 'fixture')
            revision = git('rev-parse', 'HEAD')
            script = root/'verify.cmake'
            script.write_text(f'include("{GUARD.as_posix()}")\n'
                              'cmb_verify_llama_source("${SOURCE_DIR}" "${EXPECTED_REV}")\n')
            def verify(expected=revision):
                return subprocess.run(['cmake', f'-DSOURCE_DIR={source.as_posix()}', f'-DEXPECTED_REV={expected}',
                                       '-P', str(script)], capture_output=True, text=True)
            self.assertEqual(verify().returncode, 0)
            (source/'untracked-build-output').write_text('ignored\n')
            self.assertEqual(verify().returncode, 0)
            self.assertNotEqual(verify('0'*40).returncode, 0)
            (source/'llama.h').write_text('modified\n')
            self.assertIn('unmodified llama.cpp checkout', verify().stderr)
            self.assertNotEqual(verify().returncode, 0)
            git('add', 'llama.h')
            self.assertNotEqual(verify().returncode, 0)


if __name__ == '__main__':
    unittest.main()
