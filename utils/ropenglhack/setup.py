from distutils.core import setup, Extension

module1 = Extension('ropengl',
                    sources = ['ropengl.c'],
                    libraries = ['GL'])

setup (name = 'ropengl',
       version = '1.0',
       description = 'Remote OpenGL over NX hack.',
       ext_modules = [module1])
