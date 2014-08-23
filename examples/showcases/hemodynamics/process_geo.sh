#!/bin/bash

for i in geo/ushape_*.config; do
	src=${i//.config/}
	dst=${src//ushape/proc_ushape_zyx}
	python process_geometry.py ${src} ${dst} zyx 200000
	gzip ${dst}.npy
done
