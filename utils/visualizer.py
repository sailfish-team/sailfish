#!/usr/bin/env python
"""
WX-based visualizer for data generated from the VisMixIn class. Run with:

    ./visualizer.py <remote_addr>

where remote_addr is the host, port and authentication token printed by the
simulation.
"""


import json
import re
import sys
import threading
import zlib
import time
from collections import deque

import zmq
import numpy as np
from zmq.eventloop import ioloop, zmqstream
import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
from matplotlib.backends.backend_wx import _load_bitmap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from wx.lib.pubsub import Publisher

def _extract_auth_token(addr):
    return re.search(r'//([^/@]+)@', addr).groups()[0]

def _remove_auth_token(addr):
    token = _extract_auth_token(addr)
    return addr.replace(token + '@', '')


class SmoothRate(object):
    """Computes data transfer rate smoothed out over several time steps."""

    def __init__(self, maxlen=5):
        self._data_sizes = deque(maxlen=maxlen)
        self._data_times = deque(maxlen=maxlen)
        self._data_size = 0

    def update(self, size, time):
        if len(self._data_sizes) == self._data_sizes.maxlen:
            self._data_size -= self._data_sizes.popleft()

        self._data_sizes.append(size)
        self._data_times.append(time)
        self._data_size += size

    @property
    def rate(self):
        if len(self._data_times) >= 2:
            return float(self._data_size) / (self._data_times[-1]  -
                                             self._data_times[0])
        else:
            return 0.0


class DataThread(threading.Thread):
    def __init__(self, window, ctx):
        threading.Thread.__init__(self)
        self._window = window
        self._ui_ctx = ctx
        self._data_rate = SmoothRate()

    def _handle_data(self, data):
        md = json.loads(data[0])
        md['fields'] = []

        # Number of bytes before/after compression.
        comp = 0
        uncomp = 0

        for i, buf in enumerate(data[1:]):
            comp += len(buf)
            buf = buffer(zlib.decompress(buf))
            uncomp += len(buf)
            t = np.frombuffer(buf, dtype=md['dtype'])
            md['fields'].append(t.reshape(md['shape']))

        self._data_rate.update(comp, time.time())

        # Save the compression ratio and data rate for the UI.
        md['compression'] = float(comp) / uncomp
        md['data_rate'] = self._data_rate.rate

        wx.CallAfter(Publisher().sendMessage, "update", md)

    def _handle_cmd_feedback(self, data):
        # Currently, all commands except 'port_info' just generate acks.
        assert data[0] == 'ack'

    def _handle_ui(self, data):
        # Add authentication token and send the command.
        data = json.loads(data[0])
        msg = [self._auth_token]
        msg.extend(data)
        self._cmd_stream.send_json(msg)

    def run(self):
        ioloop.install()

        # Command stream.
        self._ctx = zmq.Context()
        self._cmd_sock = self._ctx.socket(zmq.REQ)
        self._auth_token = _extract_auth_token(sys.argv[1])
        addr = _remove_auth_token(sys.argv[1])
        self._cmd_sock.connect(addr)
        self._cmd_sock.send_json((self._auth_token, 'port_info',))
        self._data_port, md = self._cmd_sock.recv_json()
        self._cmd_stream = zmqstream.ZMQStream(self._cmd_sock)
        self._cmd_stream.on_recv(self._handle_cmd_feedback)
        wx.CallAfter(Publisher().sendMessage, "init", md)

        # Data stream.
        addr = _remove_auth_token(sys.argv[1])
        addr = "%s:%s" % (addr.rsplit(':', 1)[0], self._data_port)
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(addr)
        self._sock.setsockopt(zmq.SUBSCRIBE, '')
        self._stream = zmqstream.ZMQStream(self._sock)
        self._stream.on_recv(self._handle_data)

        # UI stream.
        self._ui_sock = self._ui_ctx.socket(zmq.SUB)
        self._ui_sock.setsockopt(zmq.SUBSCRIBE, '')
        self._ui_sock.connect('inproc://ui')
        self._ui_stream = zmqstream.ZMQStream(self._ui_sock)
        self._ui_stream.on_recv(self._handle_ui)

        ioloop.IOLoop.instance().start()


class Toolbar(NavigationToolbar2WxAgg):
    pass


class CanvasFrame(wx.Frame):

    def OnInit(self, evt):
        # Update controls with initial values received from the server.
        data = evt.data
        self.position.SetValue(data['position'])
        self.every.SetValue(data['every'])
        self.axis.SetValue(self.axis.GetItems()[data['axis']])
        self.buckets.SetValue(data['levels'])

    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Sailfish viewer')

        self._ctx = zmq.Context()
        self._ui_sock = self._ctx.socket(zmq.PUB)
        self._ui_sock.bind('inproc://ui')
        self._last_transpose = False

        Publisher().subscribe(self.OnData, "update")
        Publisher().subscribe(self.OnInit, "init")
        DataThread(self, self._ctx).start()

        # UI setup.
        self.figure = Figure(figsize=(4,3), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Slice position control.
        self.position = wx.SpinCtrl(self, style=wx.TE_PROCESS_ENTER |
                                    wx.SP_ARROW_KEYS)
        self.position.Bind(wx.EVT_SPINCTRL, self.OnPositionChange)
        self.position.Bind(wx.EVT_TEXT_ENTER, self.OnPositionChange)
        self.position.SetToolTip(wx.ToolTip('Slice position along the selected '
                                            'axis.'))

        # Slice axis control.
        self.axis = wx.ComboBox(self, value='x', choices=['x', 'y', 'z'],
                                style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.axis.Bind(wx.EVT_COMBOBOX, self.OnAxisSelect)
        self.axis.SetToolTip(wx.ToolTip('Axis along which the slice is moved.'))

        # Refresh frequency control.
        self.every = wx.SpinCtrl(self, style=wx.TE_PROCESS_ENTER |
                                wx.SP_ARROW_KEYS)
        self.every.SetRange(1, 100000)
        self.every.Bind(wx.EVT_SPINCTRL, self.OnEveryChange)
        self.every.Bind(wx.EVT_TEXT_ENTER, self.OnEveryChange)
        self.every.SetToolTip(wx.ToolTip('Number of steps between data updates.'))

        # Field selector.
        self.field = wx.ComboBox(self, value='vx', choices=['vx'],
                                 style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.field.Bind(wx.EVT_COMBOBOX, self.OnFieldSelect)
        self.field.SetToolTip(wx.ToolTip('Field to visualize.'))

        # Buckets control.
        self.buckets = wx.SpinCtrl(self, style=wx.TE_PROCESS_ENTER |
                                wx.SP_ARROW_KEYS)
        self.buckets.SetRange(1, 100000)
        self.buckets.Bind(wx.EVT_SPINCTRL, self.OnBucketsChange)
        self.buckets.Bind(wx.EVT_TEXT_ENTER, self.OnBucketsChange)
        self.buckets.SetToolTip(wx.ToolTip(
            'Number of buckets for data discretization. Lower values '
            'will result in better compression ratios.'))

        self.transpose = wx.CheckBox(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()
        self.toolbar.update()
        self.sizer.Add(self.toolbar)
        self.SetSizer(self.sizer)

        self.sizer.Add(self.canvas, 10, wx.TOP | wx.LEFT | wx.EXPAND)

        # Status information.
        pos_txt = wx.StaticText(self, -1, 'Position: ')
        axis_txt = wx.StaticText(self, -1, 'Axis: ')
        every_txt = wx.StaticText(self, -1, 'Every: ')
        field_txt = wx.StaticText(self, -1, 'Field: ')
        buckets_txt = wx.StaticText(self, -1, 'Buckets: ')

        self.stat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.stat_sizer.Add(pos_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.position, 0, wx.LEFT)
        self.stat_sizer.AddSpacer(10)
        self.stat_sizer.Add(axis_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.axis, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.AddSpacer(10)
        self.stat_sizer.Add(every_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.every, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.AddSpacer(10)
        self.stat_sizer.Add(field_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.field, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.AddSpacer(10)
        self.stat_sizer.Add(buckets_txt, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.buckets, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(wx.StaticText(self, -1, 'Transpose: '), 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.stat_sizer.Add(self.transpose, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL)

        self.sizer.Add(self.stat_sizer, 0, wx.TOP | wx.LEFT | wx.ADJUST_MINSIZE)

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)

        self.plot = None
        wx.EVT_PAINT(self, self.OnPaint)

        # Timer to force updates of the main figure.
        self._timer = wx.Timer(self, 42)
        self._timer.Start(30)
        wx.EVT_TIMER(self, 42, self.OnTimer)

        self.Fit()
        self._reset_colorscale()

    def _reset_colorscale(self):
        self._cmin = 100000
        self._cmax = -100000

    def _cmd(self, name, args):
        self._ui_sock.send_json((name, args))

    def OnAxisSelect(self, event):
        self._cmd('position', 0)
        self._cmd('axis', event.GetSelection())
        self._reset_colorscale()
        self.figure.clear()
        self.plot = None

    def _get_int_from_event(self, event):
        if event.GetString():
            try:
                val = int(event.GetString())
            except ValueError:
                val = 0
        else:
            val = event.GetInt()
        return val

    def OnPositionChange(self, event):
        self._cmd('position', self._get_int_from_event(event))
        self._reset_colorscale()

    def OnEveryChange(self, event):
        self._cmd('every', self._get_int_from_event(event))

    def OnBucketsChange(self, event):
        self._cmd('levels', self._get_int_from_event(event))

    def OnFieldSelect(self, event):
        self._cmd('field', event.GetSelection())
        self._reset_colorscale()

    def OnPaint(self, event):
        self.canvas.draw()

    def OnTimer(self, event):
        self.figure.canvas.draw()

    def OnData(self, evt):
        data = evt.data
        f = data['fields'][0]
        if self.transpose.GetValue():
            f = f.transpose()

        if self._last_transpose != self.transpose.GetValue():
            self._last_transpose = self.transpose.GetValue()
            self.figure.clear()
            self.plot = None

        if self.field.GetItems() != data['names']:
            self.field.SetItems(data['names'])
            self.field.SetValue(self.field.GetItems()[data['field']])

        # Update the color map. Keep max/min values to prevent "oscillating"
        # colors which make the features of the flow more difficult to see.
        self._cmax = max(self._cmax, np.nanmax(f))
        self._cmin = min(self._cmin, np.nanmin(f))

        if self.plot is None:
            self.axes = self.figure.add_subplot(111)
            self.plot = self.axes.imshow(f, origin='lower',
                                         interpolation='nearest')
            self.cbar = self.figure.colorbar(self.plot)
            self.position.SetRange(0, data['axis_range'] - 1)
        else:
            self.plot.set_data(f)
            self.plot.set_clim(self._cmin, self._cmax)

        self.statusbar.SetStatusText('Iteration: %d' % data['iteration'], 0)
        self.statusbar.SetStatusText('Compression: %.2f' % data['compression'], 1)
        if 'data_rate' in data:
            self.statusbar.SetStatusText('Data rate: %2.f kB/s' % (data['data_rate'] /
                                         1024.0), 2)


class App(wx.App):
    def OnInit(self):
        frame = CanvasFrame()
        frame.Show(True)
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: visualizer.py <address>'
        sys.exit(0)

    app = App(0)
    app.MainLoop()
