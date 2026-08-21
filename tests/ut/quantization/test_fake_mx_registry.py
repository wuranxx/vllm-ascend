from tests.ut.base import TestBase
from vllm_ascend.quantization.methods import get_scheme_class


class TestFakeMXRegistry(TestBase):
    def test_fake_mx_schemes_are_registered(self):
        for quant_type in (
            "W4A4_MXFP4_FAKE",
            "W8A8_MXFP8_FAKE",
            "W4A4_MXFP4_FLATQUANT_FAKE",
            "W8A8_MXFP8_FLATQUANT_FAKE",
            "W4A4_MXFP4_OMNIQUANT_FAKE",
            "W8A8_MXFP8_OMNIQUANT_FAKE",
            "W4A4_MXFP4_RHT_FAKE",
            "W8A8_MXFP8_RHT_FAKE",
            "W4A4_MXFP4_HADAMARD_LEARNING_FAKE",
            "W8A8_MXFP8_HADAMARD_LEARNING_FAKE",
            "W4A4_MXFP4_AUTOROUND_FAKE",
            "W8A8_MXFP8_AUTOROUND_FAKE",
        ):
            with self.subTest(quant_type=quant_type):
                self.assertIsNotNone(get_scheme_class(quant_type, "linear"))
                if "FLATQUANT" not in quant_type:
                    self.assertIsNotNone(get_scheme_class(quant_type, "moe"))
