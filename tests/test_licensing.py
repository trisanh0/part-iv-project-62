"""Unit tests enforcing licensing compliance across installed and imported dependencies.

Strictly enforces repository governance rule #1: no third-party libraries,
algorithms, or dependencies utilizing GNU licenses (GPL, LGPL, AGPL).
"""

from collections.abc import Sequence
import importlib.metadata
from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple
import unittest


# Prohibited license token patterns (GNU copyleft licenses)
PROHIBITED_LICENSE_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"\b(?:a?gpl|lgpl)(?:v[0-9]+(?:\.[0-9]+)?)?\b", re.IGNORECASE),
    re.compile(
        r"\b(?:general\s+public\s+license|lesser\s+general\s+public\s+license|library\s+general\s+public\s+license)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\baffero\b", re.IGNORECASE),
)

# Known blacklisted module prefixes for runtime import checks
RUNTIME_MODULE_BLACKLIST: Sequence[str] = (
    "gpl",
    "lgpl",
    "rpy2",
    "pygobject",
)


def is_prohibited_license_text(text: Optional[str]) -> bool:
    """Return True if text contains prohibited copyleft license patterns."""
    if text is None or not text.strip():
        return False
    return any(pattern.search(text) is not None for pattern in PROHIBITED_LICENSE_PATTERNS)


def inspect_distribution_license(
    dist: importlib.metadata.Distribution,
) -> Tuple[bool, str]:
    """Inspect distribution metadata for copyleft license violations.

    Parameters
    ----------
    dist : importlib.metadata.Distribution
        Installed distribution to inspect.

    Returns
    -------
    Tuple[bool, str]
        A tuple of (is_violating, license_summary_string).
    """
    lic_expr: Optional[str] = dist.metadata.get("License-Expression")
    classifiers: List[str] = dist.metadata.get_all("Classifier") or []
    lic_classifiers: List[str] = [c for c in classifiers if c.startswith("License ::")]
    lic_field: Optional[str] = dist.metadata.get("License")

    # 1. Canonical PEP 639 License-Expression check
    if lic_expr is not None and is_prohibited_license_text(lic_expr):
        return True, f"License-Expression: {lic_expr}"

    # 2. PyPI standard license classifier checks
    for classifier in lic_classifiers:
        if is_prohibited_license_text(classifier):
            return True, f"Classifier: {classifier}"

    # 3. Fallback to License field when neither expression nor classifier is present
    if lic_expr is None and not lic_classifiers and lic_field is not None:
        first_line = lic_field.strip().splitlines()[0]
        if is_prohibited_license_text(first_line):
            return True, f"License header: {first_line}"

    summary = (
        lic_expr
        or (lic_classifiers[0] if lic_classifiers else None)
        or (lic_field.strip().splitlines()[0] if lic_field else "Unknown")
    )
    return False, summary


class TestLicensing(unittest.TestCase):
    """Test suite validating open-source licensing compliance."""

    def test_detector_synthetic_cases(self) -> None:
        """Verify that license violation detector accurately identifies prohibited and permitted licenses."""
        violating_samples = [
            "GPL-2.0",
            "GPL-3.0-only",
            "GPLv3",
            "LGPL-2.1-or-later",
            "LGPLv2.1",
            "GNU General Public License v3 (GPLv3)",
            "GNU Lesser General Public License",
            "Affero General Public License",
            "AGPL-3.0",
        ]
        permissive_samples = [
            "MIT",
            "MIT License",
            "BSD-3-Clause",
            "BSD-2-Clause",
            "Apache-2.0",
            "Apache License, Version 2.0",
            "Python Software Foundation License",
            "PSF-2.0",
            "ISC License (ISCL)",
            "MPL-2.0 AND MIT",
        ]

        for sample in violating_samples:
            self.assertTrue(
                is_prohibited_license_text(sample),
                f"Expected violation for '{sample}', but detector did not flag it.",
            )

        for sample in permissive_samples:
            self.assertFalse(
                is_prohibited_license_text(sample),
                f"Expected permissive match for '{sample}', but detector flagged it as violation.",
            )

    def test_all_installed_distributions_licensing(self) -> None:
        """Inspect all installed Python distributions to ensure none violate GPL/LGPL constraints."""
        distributions = list(importlib.metadata.distributions())
        self.assertGreater(
            len(distributions),
            10,
            "Distribution scan returned an unexpectedly small environment.",
        )

        violations: List[str] = []
        for dist in distributions:
            name = dist.metadata.get("Name", "unknown")
            version = dist.metadata.get("Version", "unknown")
            is_violating, reason = inspect_distribution_license(dist)
            if is_violating:
                violations.append(f"{name}=={version} ({reason})")

        error_msg = (
            f"Discovered {len(violations)} package(s) violating GPL/LGPL licensing policy:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )
        self.assertEqual(len(violations), 0, error_msg)

    def test_declared_requirements_licensing(self) -> None:
        """Verify that all packages declared in requirements.txt comply with licensing policy."""
        req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
        self.assertTrue(req_path.exists(), f"Requirements file not found at {req_path}")

        with open(req_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        packages_to_check: List[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("."):
                continue
            pkg_name = re.split(r"[><=~!]", line)[0].strip()
            if pkg_name:
                packages_to_check.append(pkg_name)

        self.assertGreater(len(packages_to_check), 5)

        for pkg in packages_to_check:
            try:
                dist = importlib.metadata.distribution(pkg)
            except importlib.metadata.PackageNotFoundError:
                self.fail(f"Required package '{pkg}' is not installed in the environment.")

            is_violating, reason = inspect_distribution_license(dist)
            self.assertFalse(
                is_violating,
                f"Required package '{pkg}' violates licensing policy: {reason}",
            )

    def test_no_gpl_runtime_imports(self) -> None:
        """Verify that no GPL or LGPL third-party modules are currently loaded in sys.modules."""
        loaded_modules = {m.split(".")[0].lower() for m in sys.modules.keys()}
        violating = loaded_modules.intersection(set(RUNTIME_MODULE_BLACKLIST))
        self.assertFalse(
            violating,
            f"Discovered loaded GPL/LGPL third-party modules: {violating}",
        )


if __name__ == "__main__":
    unittest.main()
