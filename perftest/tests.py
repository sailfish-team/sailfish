import git
import os
import sys
import time

import pycuda
import pycuda.autoinit
import pycuda.driver

repo = git.Repo('.')
head = repo.commits()[0]

def run_test(name, test_suite, block_sizes):
    global head

    # Default settings.
    defaults = {
        'benchmark': True,
        'quiet': True,
        'verbose': False,
        'max_iters': 10000,
        'every': 1000
    }

    print '* %s' % name

    if name not in test_suite:
        raise ValueError('Test %s not found' % name)

    if block_sizes is not None:
        basepath = os.path.join('perftest', 'results',
                pycuda.autoinit.device.name().replace(' ', '_'),
                'blocksize')
        path = os.path.join(basepath, name)

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        ddata = pycuda.tools.DeviceData()

        f = open(path, 'w')
        print >>f, "# block_size, performance, occupancy, tb_per_mp, warps_per_mp"

        for bs in block_sizes:
            defaults['block_size'] = bs
            settings = {}
            settings.update(defaults)
            settings.update(test_suite[name]['options'])
            sim = test_suite[name]['run'](settings)
            try:
                sim.run()
            except pycuda.driver.LaunchError:
                continue
            kern = sim._lb_kernel
            occ = pycuda.tools.OccupancyRecord(ddata, bs, kern.shared_size_bytes, kern.num_regs)
            print >>f, bs, sim._bench_avg, occ.occupancy, occ.tb_per_mp, occ.warps_per_mp

        f.close()
    else:
        settings = {}
        settings.update(defaults)
        settings.update(test_suite[name]['options'])
        sim = test_suite[name]['run'](settings)
        sim.run()

        basepath = os.path.join('perftest', 'results', pycuda.autoinit.device.name().replace(' ','_'))
        path = os.path.join(basepath, name)
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        f = open(path, 'a')
        print >>f, head.id, time.time(), sim._bench_avg
        f.close()

def run_suite(suite, args, block_sizes=None):
    sys.argv = sys.argv[0:1]

    if args:
        done = set()

        for name in args:
            if name in suite:
                run_test(name, suite, block_sizes)
            else:
                # Treat test name as a prefix if an exact match has not been found.
                for x in suite:
                    if len(name) < len(x) and name == x[0:len(name)] and x not in done:
                        run_test(x, suite, block_sizes)
                        done.add(x)
    else:
        for name in suite.iterkeys():
            run_test(name, suite, block_sizes)

