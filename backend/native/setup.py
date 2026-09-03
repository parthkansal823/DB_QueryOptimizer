"""
Build the compiled plan-tree encoder.

    python native/setup.py build_ext --inplace

Optional, always. `app/optimizer/plan_tree.py` imports it if it is present and
runs the pure-Python encoder if it is not, so a checkout with no compiler is a
working checkout. The differential test in `tests/test_plan_tree_native.py`
skips itself when the module is absent and asserts bit-identical output when it
is there.
"""

from setuptools import Extension, setup

setup(
    name="plan_tree_native",
    version="1.0",
    ext_modules=[
        Extension(
            "plan_tree_native",
            sources=["native/plan_tree_native.cpp"],
            language="c++",
            # -O2 rather than -O3: the arithmetic here is a handful of
            # comparisons and one log1p per feature, so the aggressive
            # vectorisation -O3 adds has nothing to work on, and -ffast-math
            # would be actively wrong -- it licenses reassociation that would
            # make the sums differ from Python's in the last bits, which is
            # exactly the divergence this port must not introduce.
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
)
