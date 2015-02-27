#!/bin/bash

# Change domain orientation and cut the domain so that inlet/outlet
# is exactly at the domain boundary.
for i in geo/ushape_*.config; do
	src=${i//.config/}
	dst=${src//ushape/proc_ushape_zyx}
	python process_geometry.py ${src} ${dst} xyz 200000
	gzip ${dst}.npy
done

# Process ICA geometry.
# TODO: reorder axes to keep X longest (currently Y, Z, X in descending length)
for i in geo/c0006_*.config; do
	src=${i//.config/}
	dst=${src//c0006/proc_c0006}
	python process_geometry.py ${src} ${dst} xyz 011000
	gzip ${dst}.npy
done
