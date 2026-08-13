import io, json, subprocess, unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import Mock

class CliAntigravityTest(unittest.TestCase):
    def test_launcher_pins_threads_before_import_from_outside_cwd_antigravity(self):
        launcher=Path(__file__).resolve().parents[3]/"scratch/run_exp451_rdst_local_gap_qualification_antigravity.py"
        names=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS","AEON_NUM_THREADS")
        program=f'import os,runpy;runpy.run_path({str(launcher)!r},run_name="not_main");print("".join(os.environ[x] for x in {names!r}))'
        completed=subprocess.run([__import__("sys").executable,"-c",program],cwd="/tmp",text=True,capture_output=True,check=True)
        self.assertEqual(completed.stdout.strip(),"1111111")
    def test_exact_parser_and_lazy_dispatch_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.cli_antigravity import build_parser_antigravity
        parser=build_parser_antigravity()
        for argv in (["readiness"],["predict","--workers","2"],["inspect","--path","x"]):
            with self.assertRaises(SystemExit): parser.parse_args(argv)
        self.assertEqual(parser.parse_args(["inspect"]).command,"inspect")
        from experiments.exp451_rdst_local_gap_qualification_antigravity.cli_antigravity import dispatch_antigravity
        with self.assertRaises(ValueError): dispatch_antigravity(["predict","--reviewed-commit","A"*40],Mock())

    def test_dispatch_boundaries_and_aggregate_json_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.cli_antigravity import CliDependenciesAntigravity, dispatch_antigravity
        authority=Mock(); authority.to_payload_antigravity.return_value={"reviewed_commit":"a"*40}; result=Mock()
        public={"identity":"Exp451","qualified":True}
        deps=CliDependenciesAntigravity(Mock(return_value=authority),Mock(return_value=authority),Mock(return_value=result),Mock(return_value="predecessor"),Mock(return_value=result),Mock(return_value=public))
        for argv in (["predict","--reviewed-commit","a"*40],["evaluate","--reviewed-commit","a"*40],["inspect"]):
            stream=io.StringIO()
            with redirect_stdout(stream): dispatch_antigravity(argv,deps)
            json.loads(stream.getvalue())
        deps.evaluate.assert_called_once_with(authority,"a"*40); deps.inspect.assert_called_once(); self.assertEqual(deps.public.call_count,2)
        blocked=CliDependenciesAntigravity(Mock(),Mock(side_effect=ValueError("prediction absent")),Mock(),Mock(),Mock(),Mock())
        with self.assertRaisesRegex(ValueError,"prediction absent"): dispatch_antigravity(["evaluate","--reviewed-commit","a"*40],blocked)
        blocked.evaluate.assert_not_called(); blocked.public.assert_not_called()

if __name__=="__main__": unittest.main()
