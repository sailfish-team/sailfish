#!/bin/bash

# Change domain orientation and cut the domain so that inlet/outlet
# is exactly at the domain boundary.
for i in geo/ushape_*.config; do
	src=${i//.config/}
	dst=${src//ushape/proc_ushape_zyx}
	python process_geometry.py ${src} ${dst} xyz 200000
	gzip ${dst}.npy
done
