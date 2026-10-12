import unittest
import numpy as np
from grace_rtp.blocks import *
from grace_rtp.network import *


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.code=np.random.default_rng(4).integers(-20,21,10000).astype(np.float32)
        self.blocks=compress_features(self.code)
        self.packets=packetize(self.blocks,7,start_seq=65530,mtu=300)

    def assemble(self,packets):
        a=BlockAssembler(7)
        for p in packets:a.add(p)
        return a.finish()

    def test_roundtrip_wrap_reorder_duplicates(self):
        got=self.assemble(list(reversed(self.packets))+self.packets[:2])
        np.testing.assert_array_equal(restore_features(got,len(self.code)),self.code)

    def test_single_fragment_erases_exactly_one_block(self):
        got=self.assemble(self.packets[1:])
        self.assertIsNone(got[0]);self.assertTrue(all(b is not None for b in got[1:]))
        recovered=restore_features(got,len(self.code))
        np.testing.assert_array_equal(recovered[0::10],np.zeros(1000))
        np.testing.assert_array_equal(recovered[1::10],self.code[1::10])

    def test_corruption_whole_block(self):
        p=bytearray(self.packets[0]);p[-1]^=1
        self.assertIsNone(self.assemble([bytes(p)]+self.packets[1:])[0])

    def test_no_packets_produces_ten_missing(self):
        self.assertEqual(self.assemble([]),[None]*10)

    def test_marker_loss_no_frame_gate(self):
        got=self.assemble(self.packets[:-1]);self.assertEqual(sum(x is None for x in got),1)

    def test_delay_invariance_before_deadline(self):
        got,_=Network(Profile(delay_ms=99),1).transmit(self.packets,0,7)
        self.assertEqual(got,self.blocks)
        got,trace=Network(Profile(delay_ms=101),1).transmit(self.packets,0,7)
        self.assertEqual(got,[None]*10);self.assertTrue(all(t['reason']=='late' for t in trace))

    def test_queue_persists_across_frames(self):
        n=Network(Profile(rate_mbit=.1,limit_packets=2),2)
        _,t=n.transmit(self.packets,0,7)
        self.assertEqual(sum(x['reason']=='queue_drop' for x in t),len(t)-2)
        _,t=n.transmit(self.packets,1,7)
        self.assertTrue(all(x['reason']=='queue_drop' for x in t))

    def test_iid_block_amplification(self):
        packets=packetize([bytes(range(256))*20]*10,7,mtu=300)
        rates=[]
        for seed in range(400):
            got,_=Network(Profile(loss_pct=1),seed).transmit(packets,0,7)
            rates.append(sum(x is None for x in got)/10)
        n=len(packets)/10;expected=1-.99**n
        self.assertLess(abs(np.mean(rates)-expected),.025)

    def test_unbounded_queue_not_random_loss(self):
        _,trace=Network(Profile(rate_mbit=.01),1).transmit(self.packets,0,7)
        self.assertTrue(any(t['reason']=='late' for t in trace))
        self.assertFalse(any(t['reason'].endswith('drop') for t in trace))

    def test_prediction_does_not_observe_future_arrivals(self):
        trace=[dict(send_ms=0,arrival_ms=50,size=100),dict(send_ms=0,arrival_ms=150,size=100)]
        causal=capture_stats(trace,100)
        self.assertEqual(causal['capture_loss'],.5);self.assertEqual(causal['delay_mean'],50)
        self.assertEqual(capture_stats(trace)['capture_loss'],0)


if __name__=='__main__':unittest.main()
