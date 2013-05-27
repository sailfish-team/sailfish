#!/usr/bin/python

"""
Computes effective simulation performance based on creation time of output files.

Usage:
	  ./compute_performance.py <sample_output_file>
"""


import sys
import os.path
import pickle
import glob

arg = sys.argv[1]
base, sub_id, it, _ = arg.rsplit('.', 3)
digits = len(it)
times = []
subdomains = pickle.load(open('.'.join([base, 'subdomains']), 'r'))
n_subd = subdomains[0].num_nodes
n_subd_all = sum([x.num_nodes for x in subdomains])
n = len(subdomains)

for fn in glob.glob('.'.join([base, sub_id, ('[0-9]' * digits), 'npz'])):
	base, sub_id, it, _ = fn.rsplit('.', 3)
	times.append((os.path.getmtime(fn), int(it)))
times.sort()

times2 = []
for x in sys.argv[2:]:
	if x[0] == '-':
		beg, end = x[1:].split(':', 1)
		for i, (mtime, it) in enumerate(times):
			if int(beg) <= it <= int(end):
				del times[i]

for x in sys.argv[2:]:
	if x[0] == '+':
		beg, end = x[1:].split(':', 1)
		for i, (mtime, it) in enumerate(times):
			if int(beg) <= it <= int(end):
				times2.append(times[i])

if times2:
	times2.sort()
	times = times2


def eff(start, stop, times):
	effi = [n_subd / 1e6 * (t2[1] - t1[1]) / (t2[0] - t1[0]) for t1, t2 in zip(times[start:stop + 1], times[start + 1:stop + 1])]
	mini = min(effi)
	maxi = max(effi)
	aver = sum(effi)/len(effi)
	index = 'files: ' + str(times[start][1]) + ' - ' + str(times[stop][1])
	nn = ' (' + str(n) + ')'
	print index
	print '{0:<20} min: {1:.2f} / max: {2:.2f} / avg: {3:.2f} MLUPS'.format('one subdomain', mini, maxi, aver)
	print '{0:<15}{1:<5} min: {2:.2f} / max: {3:.2f} / avg: {4:.2f} MLUPS'.format('all subdomains', nn, mini * n_subd_all / n_subd, maxi * n_subd_all / n_subd, aver * n_subd_all / n_subd)


start = 0
stop = 1

for i in range(len(times) - 1):
	if start >= i:
		continue
	if times[i + 1][1] < times[i][1]:
		stop = i
		eff(start, stop, times)
		# i+1 is 0 or 1st of resumed sim
		start = i + 2
	if i > 0 and times[i][1] - times[i - 1][1] != times[i + 1][1] - times[i][1]:
		stop = i
		eff(start, stop, times)
		start = i + 1
		stop = i + 2
	else:
		stop = i + 1

if stop < len(times):
	eff(start, stop, times)


