"""Verify linked stock/fork jitter-buffer and NACK behavior."""
import inspect,unittest
from aiortc.rtp import RtpPacket
from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtcrtpreceiver import NackGenerator


class LinkedAiortcTests(unittest.TestCase):
    def setUp(self):
        self.fork='codecName' in inspect.signature(JitterBuffer.add).parameters

    def add(self,buffer,packet):
        return buffer.add(packet,'VP8') if self.fork else buffer.add(packet)

    def packet(self,seq,stamp,body,marker=0):
        p=RtpPacket(sequence_number=seq,timestamp=stamp,marker=marker);p._data=body;return p

    def test_marker_does_not_flush_but_next_timestamp_does(self):
        j=JitterBuffer(16,is_video=True)
        self.assertIsNone(self.add(j,self.packet(100,0,b'a'))[1])
        self.assertIsNone(self.add(j,self.packet(101,0,b'b',1))[1])
        self.assertEqual(self.add(j,self.packet(102,3000,b'c'))[1].data,b'ab')

    def test_video_hole_blocks_delivery(self):
        j=JitterBuffer(16,is_video=True)
        for p in [self.packet(100,0,b'a'),self.packet(102,0,b'c'),self.packet(103,3000,b'd')]:
            self.assertIsNone(self.add(j,p)[1])

    def test_token_hole_is_skipped(self):
        if not self.fork:self.skipTest('Only the previous Testbed fork has the token codec branch')
        j=JitterBuffer(16,is_video=True)
        j.add(self.packet(100,0,b'a'),'tokenId')
        j.add(self.packet(102,0,b'c'),'tokenId')
        self.assertEqual(j.add(self.packet(103,3000,b'd'),'tokenId')[1].data,b'ac')

    def test_nack_detects_sequence_gap(self):
        n=NackGenerator();n.add(self.packet(100,0,b'a'))
        self.assertTrue(n.add(self.packet(102,0,b'c')));self.assertEqual(n.missing,{101})


if __name__=='__main__':unittest.main()
