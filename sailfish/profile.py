"""Code for profiling a simulation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import time
from sailfish import util

class TimeProfile(object):
    """Maintains statistics about time spent in different parts of the
    simulation."""

    # GPU events.
    BULK = 0
    BOUNDARY = 1
    COLLECTION = 2
    DISTRIB = 3
    MACRO_BULK = 4
    MACRO_BOUNDARY = 5
    MACRO_COLLECTION = 6
    MACRO_DISTRIB = 7

    # CPU events.
    SEND_DISTS = 8
    RECV_DISTS = 9
    SEND_MACRO = 10
    RECV_MACRO = 11
    NET_RECV = 12

    STEP = 13

    # This event needs to have the highest ID.
    # Square of total calculation time. Used for standard deviation.
    STEP_SQ = 14

    def __init__(self, runner):
        self._runner = runner
        self._make_event = runner.backend.make_event
        self._events_start = {}
        self._events_end = {}
        self._times_start = [0.0] * (self.STEP_SQ + 1)
        self._timings = [0.0] * (self.STEP_SQ + 1)
        self._min_timings = [1000.0] * (self.STEP_SQ + 1)
        self._max_timings = [0.0] * (self.STEP_SQ + 1)
        self._samples = 0
        self._sample_sum = 0.0
        self._is_benchmark = runner.config.mode == 'benchmark'

    def record_start(self):
        self.t_start = time.time()

    def record_end(self):
        self.t_end = time.time()
        if not self._is_benchmark:
            return
        mi = self._runner.config.max_iters - self._runner.config.benchmark_sample_from
        assert mi > 0

        # Final minibatch might be incomplete, but we still need to take it into
        # account.
        if self._samples > 0:
            self._timings[self.STEP_SQ] += self._samples * (self._sample_sum /
                                                            self._samples)**2

        ti = util.TimingInfo(
                comp=(self._timings[self.BULK] + self._timings[self.BOUNDARY]) / mi,
                bulk=self._timings[self.BULK] / mi,
                bnd =self._timings[self.BOUNDARY] / mi,
                coll=self._timings[self.COLLECTION] / mi,
                net_wait=self._timings[self.NET_RECV] / mi,
                recv=self._timings[self.RECV_DISTS] / mi,
                send=self._timings[self.SEND_DISTS] / mi,
                total=self._timings[self.STEP] / mi,
                total_sq=self._timings[self.STEP_SQ] / mi,
                subdomain_id=self._runner._spec.id)

        min_ti = util.TimingInfo(
                comp=(self._min_timings[self.BULK] + self._min_timings[self.BOUNDARY]),
                bulk=self._min_timings[self.BULK],
                bnd =self._min_timings[self.BOUNDARY],
                coll=self._min_timings[self.COLLECTION],
                net_wait=self._min_timings[self.NET_RECV],
                recv=self._min_timings[self.RECV_DISTS],
                send=self._min_timings[self.SEND_DISTS],
                total=self._min_timings[self.STEP],
                total_sq=0.0,
                subdomain_id=self._runner._spec.id)

        max_ti = util.TimingInfo(
                comp=(self._max_timings[self.BULK] + self._max_timings[self.BOUNDARY]),
                bulk=self._max_timings[self.BULK],
                bnd =self._max_timings[self.BOUNDARY],
                coll=self._max_timings[self.COLLECTION],
                net_wait=self._max_timings[self.NET_RECV],
                recv=self._max_timings[self.RECV_DISTS],
                send=self._max_timings[self.SEND_DISTS],
                total=self._max_timings[self.STEP],
                total_sq=0.0,
                subdomain_id=self._runner._spec.id)

        self._runner.send_summary_info(ti, min_ti, max_ti)

    def start_step(self):
        self.record_cpu_start(self.STEP)

    def end_step(self):
        if (not self._is_benchmark or self._runner._sim.iteration <
            self._runner.config.benchmark_sample_from):
            return

        self.record_cpu_end(self.STEP)

        # Aggregate timings from GPU events.
        for i, ev_start in self._events_start.iteritems():
            duration = self._events_end[i].time_since(ev_start) / 1e3
            self._timings[i] += duration
            self._min_timings[i] = min(self._min_timings[i], duration)
            self._max_timings[i] = max(self._max_timings[i], duration)

    def record_gpu_start(self, event, stream):
        ev = self._make_event(stream, timing=self._is_benchmark)
        self._events_start[event] = ev
        return ev

    def record_gpu_end(self, event, stream, need_event=False):
        if not self._is_benchmark and not need_event:
            return
        ev = self._make_event(stream, timing=self._is_benchmark)
        self._events_end[event] = ev
        return ev

    def record_cpu_start(self, event):
        self._times_start[event] = time.time()

    def record_cpu_end(self, event):
        if (self._runner._sim.iteration <
            self._runner.config.benchmark_sample_from):
            return

        t_end = time.time()
        duration = t_end - self._times_start[event]
        self._min_timings[event] = min(self._min_timings[event], duration)
        self._max_timings[event] = max(self._max_timings[event], duration)
        self._timings[event] += duration

        minibatch = self._runner.config.benchmark_minibatch

        if event == self.STEP:
            self._samples += 1
            self._sample_sum += duration

            if self._samples == minibatch:
                self._timings[self.STEP_SQ] += minibatch * (self._sample_sum / minibatch)**2
                self._sample_sum = 0.0
                self._samples = 0


def profile(profile_event):
    def _profile(f):
        def decorate(self, *args, **kwargs):
            self._profile.record_cpu_start(profile_event)
            ret = f(self, *args, **kwargs)
            self._profile.record_cpu_end(profile_event)
            return ret
        return decorate
    return _profile

