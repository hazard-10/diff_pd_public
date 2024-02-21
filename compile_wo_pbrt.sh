# remove exisiting .py, .so files
rm -f python/py_diff_pd/core/py_diff_pd_core.py
rm -f python/py_diff_pd/core/_py_diff_pd_core.so

cd cpp/core/src
swig -c++ -python py_diff_pd_core.i

# Compile c++ code.
cd ../../
mkdir -p build
cd build
if [ $# -eq 0 ]
then
    cmake -DPARDISO_AVAILABLE=OFF ..
else
    cmake -DPARDISO_AVAILABLE=ON ..
fi
make -j20
./diff_pd_demo

# Python binding.
cd ../core/src/
mv py_diff_pd_core.py ../../../python/py_diff_pd/core
mv ../../build/libpy_diff_pd_core.so ../../../python/py_diff_pd/core/_py_diff_pd_core.so

# Log absolute path.
cd ../../../
root_path=$(pwd)
printf "root_path = '%s'\n" "$root_path" > python/py_diff_pd/common/project_path.py