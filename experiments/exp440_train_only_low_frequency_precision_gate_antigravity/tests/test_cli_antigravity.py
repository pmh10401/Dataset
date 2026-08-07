from __future__ import annotations
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.cli_antigravity import build_parser_antigravity
class CliTests(unittest.TestCase):
 def test_exactly_five_commands_and_no_overrides(self):
  parser=build_parser_antigravity();self.assertEqual(set(parser._subparsers._group_actions[0].choices),{'readiness','new','resume','inspect','evaluate'})
  for flag in ('--workers','--backend','--threshold','--budget'):
   with self.assertRaises(SystemExit):parser.parse_args(['new',flag,'1'])
if __name__=='__main__':unittest.main()
