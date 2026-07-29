"""Unit test enforcing licensing policy (no GPL/LGPL dependencies)."""

import unittest
import sys


class TestLicensing(unittest.TestCase):
    def test_no_gpl_imports(self):
        """Verify that no GPL or LGPL third-party packages are loaded."""
        gpl_blacklist = {"gpl", "lgpl", "rpy2", "pygobject"}
        loaded_modules = {m.split(".")[0].lower() for m in sys.modules.keys()}
        violating = loaded_modules.intersection(gpl_blacklist)

        self.assertFalse(violating, f"Discovered GPL/LGPL third-party modules: {violating}")


if __name__ == "__main__":
    unittest.main()

