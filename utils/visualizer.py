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

class DataThread(threading.Thread):
    def __init__(self, window, data_port):
        threading.Thread.__init__(self)
        self._window = window
        self._data_port = data_port

    def _handle_data(self, data):
        md = json.loads(data[0])
        md['fields'] = []

        # Number of bytes before/after compression.
        comp = 0
        uncomp = 0

        for i, buf in enumerate(data[1:]):
            comp = len(buf)
            buf = zlib.decompress(buf)
            buf = buffer(buf)
            uncomp += len(buf)

            t = np.frombuffer(buf, dtype=md['dtype'])
            md['fields'].append(t.reshape(md['shape']))

        # Save the compression ratio.
        md['compression'] = float(comp) / uncomp
        wx.CallAfter(Publisher().sendMessage, "update", md)

    def run(self):
        ioloop.install()
        addr = _remove_auth_token(sys.argv[1])
        addr = "%s:%s" % (addr.rsplit(':', 1)[0], self._data_port)

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(addr)
        self._sock.setsockopt(zmq.SUBSCRIBE, '')
        self._stream = zmqstream.ZMQStream(self._sock)
        self._stream.on_recv(self._handle_data)
        ioloop.IOLoop.instance().start()


class Toolbar(NavigationToolbar2WxAgg):
    pass


class CanvasFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Sailfish viewer')

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._auth_token = _extract_auth_token(sys.argv[1])
        addr = _remove_auth_token(sys.argv[1])
        self._sock.connect(addr)
        self._cmd('every', 25)
        self._sock.send_json((self._auth_token, 'port_info',))
        data_port = self._sock.recv_json()

        # UI setup.
        self.figure = Figure(figsize=(4,3), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Slice position control.
        self.position = wx.SpinCtrl(self)
        self.position.SetRange(0, 10)
        self.position.SetValue(0)
        self.Bind(wx.EVT_SPINCTRL, self.OnPositionChange)

        # Slice axis control.
        self.axis = wx.ComboBox(self, value='x', choices=['x', 'y', 'z'],
                                style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.OnAxisSelect)

        # Refresh frequency control.
        self.every = wx.SpinCtrl(self)
        self.every.SetRange(1, 100000)
        self.every.SetValue(25)
        self.Bind(wx.EVT_SPINCTRL, self.OnEveryChange)

        # Field selector.
        self.field = wx.ComboBox(self, value='vx', choices=['vx'],
                                 style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.Bind(wx.EVT_COMBOBOX, self.OnFieldSelect)

        # Status information.
        pos_txt = wx.StaticText(self, -1, 'Position: ')
        axis_txt = wx.StaticText(self, -1, 'Axis: ')
        every_txt = wx.StaticText(self, -1, 'Every: ')
        field_txt = wx.StaticText(self, -1, 'Field: ')

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()
        self.toolbar.update()
        self.sizer.Add(self.toolbar)
        self.SetSizer(self.sizer)

        self.info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.info_iter = wx.StaticText(self, -1, 'Iteration: NA',
                                       style=wx.ST_NO_AUTORESIZE )
        self.info_compression = wx.StaticText(self, -1, 'Compression: NA',
                                              style=wx.ST_NO_AUTORESIZE )
        self.info_sizer.Add(self.info_iter, 0, wx.ALIGN_CENTER_VERTICAL)
        self.info_sizer.AddSpacer(10)
        self.info_sizer.Add(self.info_compression, 0, wx.ALIGN_CENTER_VERTICAL)
        self.SetDoubleBuffered(True)

        self.sizer.Add(self.info_sizer)
        self.sizer.Add(self.canvas, 10, wx.TOP | wx.LEFT | wx.EXPAND)

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

        self.sizer.Add(self.stat_sizer, 0, wx.TOP | wx.LEFT | wx.ADJUST_MINSIZE)

        self.plot = None
        wx.EVT_PAINT(self, self.OnPaint)

        Publisher().subscribe(self.OnData, "update")
        DataThread(self, data_port).start()

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
        self._sock.send_json((self._auth_token, name, args))
        assert self._sock.recv_string() == 'ack'

    def OnAxisSelect(self, event):
        self._cmd('position', 0)
        self._cmd('axis', event.GetSelection())
        self._reset_colorscale()
        self.figure.clear()
        self.plot = None

    def OnPositionChange(self, event):
        self._cmd('position', event.GetInt())

    def OnEveryChange(self, event):
        self._cmd('every', event.GetInt())

    def OnFieldSelect(self, event):
        self._cmd('field', event.GetSelection())
        self._reset_colorscale()

    def OnPaint(self, event):
        self.canvas.draw()
        event.Skip()

    def OnTimer(self, event):
        self.figure.canvas.draw()
        event.Skip()

    def OnData(self, evt):
        data = evt.data
        f = data['fields'][0].transpose()

        if self.field.GetItems() != data['names']:
            self.field.Set(data['names'])

        # Update the color map. Keep max/min values to prevent "oscillating"
        # colors which make the features of the flow more difficult to see.
        self._cmax = max(self._cmax, np.nanmax(f))
        self._cmin = min(self._cmin, np.nanmin(f))

        if self.plot is None:
            self.axes = self.figure.add_subplot(111)
            self.plot = self.axes.imshow(f, origin='lower',
                                         interpolation='nearest')
            self.cbar = self.figure.colorbar(self.plot)
            self.position.SetRange(0, data['axis'] - 1)
        else:
            self.plot.set_data(f)
            self.plot.set_clim(self._cmin, self._cmax)

        self.info_iter.SetLabel('Iteration: %d' % data['iteration'])
        self.info_compression.SetLabel('Compression: %.2f' %
                                       data['compression'])

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
