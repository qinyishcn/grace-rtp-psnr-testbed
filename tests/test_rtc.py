import unittest
import numpy as np
from grace_rtp.blocks import compress_features,restore_features
from grace_rtp.network import Profile
from grace_rtp.rtc import transfer


class RtcTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_srtp_roundtrip_and_reliable_iframe(self):
        code=np.arange(2000,dtype=np.float32)
        got,e=await transfer({1:compress_features(code)},b'traditional-iframe-bytes',Profile(deadline_ms=300))
        np.testing.assert_array_equal(restore_features(got[1],len(code)),code)
        self.assertTrue(e['iframe_verified']);self.assertEqual(e['left_state'],'connected')
        self.assertFalse(e['nack']);self.assertFalse(e['rtx'])


if __name__=='__main__':unittest.main()
