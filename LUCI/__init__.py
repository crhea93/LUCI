"""
LUCI: a general-purpose emission-line fitting pipeline for SITELLE IFU cubes.

This package previously worked only because the repository root happened to be
on ``sys.path`` (every example did ``sys.path.insert(0, '/path/to/LUCI/')``).
Adding this file makes ``LUCI`` a real, installable package so ``pip install``
/ ``uv sync`` resolves the imports instead.

The lowercase ``luci`` package introduced by the later refactor phases will
supersede this one; the public entry point remains ``from LuciBase import Luci``
until then.
"""
