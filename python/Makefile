PYTHON_VERSION = python2.7
PYTHON_LANG_LEVEL = 2
PYTHON_INCLUDE_PATH = /usr/include/python2.7/
PYTHON_LD_PATH = /usr/lib/
CYTHONFLAGS = -DPY_NO_DEPRECATED_API

CYTHON = cython --cplus -$(PYTHON_LANG_LEVEL)

REMOVE = rm

OPTFLAGS= -O3 -fno-strict-aliasing -DNDEBUG
CXXFLAGS = -fPIC -shared -pthread -Wall $(CYTHONFLAGS) $(OPTFLAGS)

build:
	$(CYTHON) example_cython.pyx
	$(CXX) $(CXXFLAGS) example_cython.cpp -o example_cython.so -I. -I$(PYTHON_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

clean:
	$(REMOVE) example_cython.cpp example_cython.so
