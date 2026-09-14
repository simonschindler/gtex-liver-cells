import unittest


class PackageTests(unittest.TestCase):
    def test_package_imports_and_exposes_version(self):
        import gtex_liver_cells

        self.assertEqual(gtex_liver_cells.__version__, "0.1.0")
