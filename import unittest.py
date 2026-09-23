import unittest
from tensorflow.keras import Sequential
from sgan import SGAN  # Assuming SGAN is the class containing the build_generator method

class TestSGAN(unittest.TestCase):
    def setUp(self):
        self.sgan = SGAN()  # Initialize your class

    def test_build_generator(self):
        model = self.sgan.build_generator()
        self.assertIsInstance(model, Sequential)  # Check if the returned model is a Sequential model

        # Check the structure of the model
        self.assertEqual(len(model.layers), 21)  # Adjust this based on the expected number of layers
        # Add more assertions here to check the structure of the model

if __name__ == '__main__':
    unittest.main()