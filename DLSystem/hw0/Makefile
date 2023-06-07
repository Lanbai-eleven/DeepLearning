# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
# 	# g++ -O3 -Wall -shared -std=c++11 -fPIC -I 'F:/anaconda3/envs/f/include' -I 'F:/anaconda3/envs/f/lib/site-packages/pybind11/include' src/simple_ml_ext.cpp -o src/simple_ml_ext.so

# CXXFLAGS = -O3 -Wall -shared -std=gnu++11 -fPIC -I'F:/anaconda3/envs/f/include' -I'F:/anaconda3/envs/f/Lib/site-packages/pybind11/include'
# LDFLAGS = -L'F:/anaconda3/envs/f/libs' -L'F:/anaconda3/envs/f/Lib' 	-lpython38.lib

# all: src/simple_ml_ext.so

# src/simple_ml_ext.so: src/simple_ml_ext.cpp
# 	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)