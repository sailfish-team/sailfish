import git
import os
import time

import pycuda
import pycuda.autoinit

repo = git.Repo('.')
head = repo.commits()[0]

def run_test(name, test_suite):
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

def run_suite(suite, args):
    if args:
        done = set()

        for name in args:
            if name in suite:
                run_test(name, suite)
            else:
                # Treat test name as a prefix if an exact match has not been found.
                for x in suite:
                    if len(name) < len(x) and name == x[0:len(name)] and x not in done:
                        run_test(x, suite)
                        done.add(x)
    else:
        for name in suite.iterkeys():
            run_test(name, suite)

