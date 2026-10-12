"""Deterministic one-way packet emulator; virtual time, not Linux netem."""
from dataclasses import dataclass, asdict
import random
from .blocks import BlockAssembler


@dataclass
class Profile:
    delay_ms: float = 0
    jitter_ms: float = 0
    loss_pct: float = 0
    rate_mbit: float | None = None
    limit_packets: int | None = None
    deadline_ms: float = 100
    burst_mean: float = 1
    corrupt_pct: float = 0

    def __post_init__(self):
        if not 0 <= self.loss_pct <= 100 or self.burst_mean < 1:
            raise ValueError("Invalid loss parameters")
        if min(self.delay_ms, self.jitter_ms, self.deadline_ms) < 0:
            raise ValueError("Negative timing")
        if self.rate_mbit is not None and self.rate_mbit <= 0:
            raise ValueError("Non-positive rate")
        if self.limit_packets is not None and self.limit_packets < 1:
            raise ValueError("Non-positive queue")


class Network:
    def __init__(self, profile, seed):
        self.p = profile
        self.rng = random.Random(seed)
        self.finish_ms = 0.
        self.departures = []
        self.bad = self.rng.random() < profile.loss_pct / 100

    def schedule(self, packets, generated_ms, frame_id):
        """Continuous FIFO across frames. Loss applied after serialization."""
        p = self.p
        arrivals, trace = [], []
        for index, data in enumerate(packets):
            now = generated_ms
            self.departures = [t for t in self.departures if t > now]
            record = dict(frame_id=frame_id, packet_index=index, send_ms=now, size=len(data)+28,
                          arrival_ms=None, reason="ok")
            if p.limit_packets is not None and len(self.departures) >= p.limit_packets:
                record["reason"] = "queue_drop"
                trace.append(record)
                continue
            service = (len(data)+28)*8/(p.rate_mbit*1000) if p.rate_mbit else 0
            departure = max(now, self.finish_ms) + service
            self.finish_ms = departure
            self.departures.append(departure)
            loss = p.loss_pct / 100
            if p.burst_mean == 1:
                lost = self.rng.random() < loss
            else:
                lost = self.bad
                # Stationary Markov loss, E[bad run] = burst_mean.
                if self.bad:
                    self.bad = self.rng.random() >= 1/p.burst_mean
                else:
                    self.bad = self.rng.random() < min(1., loss/max(1e-12,1-loss)/p.burst_mean)
            if lost:
                record["reason"] = "random_drop"
            else:
                arrival = departure + max(0., self.rng.gauss(p.delay_ms,p.jitter_ms))
                record["arrival_ms"] = arrival
                if arrival > generated_ms + p.deadline_ms:
                    record["reason"] = "late"
                if self.rng.random()*100 < p.corrupt_pct:
                    damaged = bytearray(data)
                    damaged[-1] ^= 1
                    data = bytes(damaged)
                    record["reason"] = "corrupt" if record["reason"] == "ok" else "late"
                arrivals.append((arrival, data))
            trace.append(record)
        return sorted(arrivals, key=lambda x:x[0]), trace

    def transmit(self, packets, generated_ms, frame_id):
        arrivals, trace = self.schedule(packets, generated_ms, frame_id)
        assembler = BlockAssembler(frame_id)
        for arrival, data in arrivals:
            if arrival <= generated_ms + self.p.deadline_ms:
                assembler.add(data)
        return assembler.finish(), trace


def capture_stats(trace, observation_ms=None):
    """Paired sender/receiver capture observables; no decoded block mask or PSNR."""
    received = sorted([t for t in trace if t["arrival_ms"] is not None and
                       (observation_ms is None or t["arrival_ms"]<=observation_ms)],key=lambda x:x["arrival_ms"])
    n = len(trace)
    delays = [t["arrival_ms"]-t["send_ms"] for t in received]
    import numpy as np
    gaps = np.diff([t["arrival_ms"] for t in received])
    return dict(packet_count=n, wire_bytes=sum(t["size"] for t in trace),
                capture_loss=1-len(received)/max(1,n),
                delay_mean=float(np.mean(delays)) if delays else 0.,
                delay_p95=float(np.percentile(delays,95)) if delays else 0.,
                delay_std=float(np.std(delays)) if delays else 0.,
                iat_std=float(np.std(gaps)) if len(gaps) else 0.,
                max_size=max([t["size"] for t in trace],default=0))
