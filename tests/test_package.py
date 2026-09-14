import unittest


class PackageTests(unittest.TestCase):
    def test_package_imports_and_exposes_version(self):
        import gtex_meta

        self.assertEqual(gtex_meta.__version__, "0.1.0")
