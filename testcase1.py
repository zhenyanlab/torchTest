# 单元测试类
import sys
import unittest
from io import StringIO

from pytorch2 import print_type_structure


class TestPrintTypeStructure(unittest.TestCase):
    def test_empty_list(self):
        test_output = StringIO()
        sys.stdout = test_output
        print_type_structure([])
        sys.stdout = sys.__stdout__
        self.assertIn('List/Tuple:', test_output.getvalue())

    def test_nested_list(self):
        test_output = StringIO()
        sys.stdout = test_output
        print_type_structure([1, [2, 3], (4, 5)])
        sys.stdout = sys.__stdout__
        self.assertIn('List/Tuple:', test_output.getvalue())
        self.assertIn('List/Tuple:', test_output.getvalue())
        self.assertIn('Type: int', test_output.getvalue())

    def test_dictionary(self):
        test_output = StringIO()
        sys.stdout = test_output
        print_type_structure({'a': 1, 'b': [2, 3]})
        sys.stdout = sys.__stdout__
        self.assertIn('Dictionary:', test_output.getvalue())
        self.assertIn('Key: str, Value:', test_output.getvalue())
        self.assertIn('Type: int', test_output.getvalue())

    def test_simple_object(self):
        test_output = StringIO()
        sys.stdout = test_output
        print_type_structure(123)
        sys.stdout = sys.__stdout__
        self.assertIn('Type: int', test_output.getvalue())


# 运行测试