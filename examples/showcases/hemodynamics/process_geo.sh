#!/bin/bash

for i in geo/ushape_*.config; do
	src=${i//.config/}
	dst=${src//ushape/proc_ushape}
	python process_geometry.py ${src} ${dst} zxy 200000
	gzip ${dst}.npy
done
