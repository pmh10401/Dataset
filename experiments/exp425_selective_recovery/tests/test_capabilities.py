"""Behavioral tests for Exp425's private exact-type capability broker."""

from __future__ import annotations

import copy
import pickle
import unittest

from experiments.exp425_selective_recovery import capabilities


class CapabilityBrokerTests(unittest.TestCase):
    """A capability stops authorizing writes when its owner or claim changes."""

    def setUp(self) -> None:
        self.authority = object()
        capabilities._register_authority(self.authority, "local")

    def tearDown(self) -> None:
        capabilities._close_authority(self.authority)

    def test_exact_claim_is_consumed_once(self) -> None:
        capability = capabilities._mint(
            self.authority, kind="native", run_identity="run", dataset="dataset",
            context_sha256="a" * 64, trace_sha256="b" * 64,
        )
        capabilities._consume(
            capability, kind="native", run_identity="run", dataset="dataset",
            context_sha256="a" * 64, trace_sha256="b" * 64,
        )
        with self.assertRaises(ValueError):
            capabilities._consume(capability, kind="native", run_identity="run")

    def test_rejects_forged_copied_and_dead_capabilities(self) -> None:
        capability = capabilities._mint(self.authority, kind="native", run_identity="run")
        for forged in (object(),):
            with self.assertRaises(ValueError):
                capabilities._consume(forged, kind="native", run_identity="run")
        with self.assertRaises(TypeError):
            copy.copy(capability)
        with self.assertRaises(TypeError):
            pickle.dumps(capability)
        capabilities._close_authority(self.authority)
        with self.assertRaises(ValueError):
            capabilities._consume(capability, kind="native", run_identity="run")

    def test_claim_mismatch_and_unregistered_authority_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            capabilities._mint(object(), kind="native", run_identity="run")
        capability = capabilities._mint(self.authority, kind="native", run_identity="run")
        with self.assertRaises(ValueError):
            capabilities._consume(capability, kind="native", run_identity="other")
        capabilities._consume(capability, kind="native", run_identity="run")

    def test_runtime_and_source_authorities_cannot_cross_mint_operation_families(self) -> None:
        runtime, source = object(), object()
        capabilities._register_authority(runtime, "runtime")
        capabilities._register_authority(source, "source")
        try:
            with self.assertRaises(ValueError):
                capabilities._mint(runtime, kind="import_begin", run_identity="run", batch_identity="batch")
            with self.assertRaises(ValueError):
                capabilities._mint(source, kind="native", run_identity="run")
            token = capabilities._mint(runtime, kind="native", run_identity="run")
            capabilities._consume(token, kind="native", run_identity="run")
        finally:
            capabilities._close_authority(runtime)
            capabilities._close_authority(source)
