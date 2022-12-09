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
	$(CYTHON) boolpy2c.pyx
	$(CXX) $(CXXFLAGS) boolpy2c.cpp -o boolpy2c.so -I. -I$(PYTHON_INCLUDE_PATH) -L$(PYTHON_LD_PATH) -l$(PYTHON_VERSION)

clean:
	$(REMOVE) boolpy2c.cpp boolpy2c.so
