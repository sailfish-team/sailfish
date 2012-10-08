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

    # This event needs to have the highest ID>
    STEP = 13

    def __init__(self, runner):
        self._runner = runner
        self._make_event = runner.backend.make_event
        self._events_start = {}
        self._events_end = {}
        self._times_start = [0.0] * (self.STEP + 1)
        self._timings = [0.0] * (self.STEP + 1)
        self._min_timings = [1000.0] * (self.STEP + 1)
        self._max_timings = [0.0] * (self.STEP + 1)

    def record_start(self):
        self.t_start = time.time()

    def record_end(self):
        self.t_end = time.time()
        if self._runner.config.mode != 'benchmark':
            return
        mi = self._runner.config.max_iters

        ti = util.TimingInfo(
                comp=(self._timings[self.BULK] + self._timings[self.BOUNDARY]) / mi,
                bulk=self._timings[self.BULK] / mi,
                bnd =self._timings[self.BOUNDARY] / mi,
                coll=self._timings[self.COLLECTION] / mi,
                net_wait=self._timings[self.NET_RECV] / mi,
                recv=self._timings[self.RECV_DISTS] / mi,
                send=self._timings[self.SEND_DISTS] / mi,
                total=self._timings[self.STEP] / mi,
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
                subdomain_id=self._runner._spec.id)

        self._runner.send_summary_info(ti, min_ti, max_ti)

    def start_step(self):
        self.record_cpu_start(self.STEP)

    def end_step(self):
        self.record_cpu_end(self.STEP)

        for i, ev_start in self._events_start.iteritems():
            duration = self._events_end[i].time_since(ev_start) / 1e3
            self._timings[i] += duration
            self._min_timings[i] = min(self._min_timings[i], duration)
            self._max_timings[i] = max(self._max_timings[i], duration)

    def record_gpu_start(self, event, stream):
        ev = self._make_event(stream, timing=True)
        self._events_start[event] = ev
        return ev

    def record_gpu_end(self, event, stream):
        ev = self._make_event(stream, timing=True)
        self._events_end[event] = ev
        return ev

    def record_cpu_start(self, event):
        self._times_start[event] = time.time()

    def record_cpu_end(self, event):
        t_end = time.time()
        duration = t_end - self._times_start[event]
        self._min_timings[event] = min(self._min_timings[event], duration)
        self._max_timings[event] = max(self._max_timings[event], duration)
        self._timings[event] += duration


def profile(profile_event):
    def _profile(f):
        def decorate(self, *args, **kwargs):
            self._profile.record_cpu_start(profile_event)
            ret = f(self, *args, **kwargs)
            self._profile.record_cpu_end(profile_event)
            return ret
        return decorate
    return _profile

