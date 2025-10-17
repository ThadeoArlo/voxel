from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys
import os

# Custom build_ext to add compiler-specific flags
class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/O2', '/openmp'],
        'unix': ['-O3', '-std=c++17'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = list(self.c_opts.get(ct, []))
        link_opts = list(self.l_opts.get(ct, []))

        # Apple Silicon OpenMP configuration
        if sys.platform == 'darwin':
            # Prefer Homebrew prefix for arm64
            brew_prefix = os.environ.get('HOMEBREW_PREFIX', '/opt/homebrew')
            omp_include = os.path.join(brew_prefix, 'include')
            omp_lib = os.path.join(brew_prefix, 'lib')

            # Use clang's OpenMP via libomp
            opts += ['-Xpreprocessor', '-fopenmp']
            link_opts += ['-lomp', f'-Wl,-rpath,{omp_lib}']

            for ext in self.extensions:
                # Ensure C++17 and optimization
                ext.extra_compile_args = list(getattr(ext, 'extra_compile_args', [])) + opts
                ext.extra_link_args = list(getattr(ext, 'extra_link_args', [])) + link_opts
                # Add include/lib paths for Homebrew libomp
                ext.include_dirs = list(getattr(ext, 'include_dirs', [])) + [omp_include]
                ext.library_dirs = list(getattr(ext, 'library_dirs', [])) + [omp_lib]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = list(getattr(ext, 'extra_compile_args', [])) + opts + ['-fopenmp']
                ext.extra_link_args = list(getattr(ext, 'extra_link_args', [])) + link_opts + ['-fopenmp']

        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'process_image_cpp',
        ['process_image.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++'
    ),
]

setup(
    name='process_image_cpp',
    version='0.0.1',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
