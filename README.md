# libkdtree
[![Build Status](https://travis-ci.org/WiseDoge/libkdtree.svg?branch=master)](https://travis-ci.org/WiseDoge/libkdtree)   
A KD-Tree model written in C++(with C and Python interface).

## Usage
### Python
```python
import kdtree as kt
import numpy as np
  
X = np.random.rand(60000, 250)
y = np.random.rand(60000)
  
clf = kt.KNeighborsRegressor(k=5)
clf.fit(X, y)
  
ypred = clf.predict(X[:10]) 
print(ypred) 
```
### C
```c
#include "kdtree.h"
#include <stdio.h>
int main()
{
    double datas[100] = {1.3, 1.3, 1.3,
                         8.3, 8.3, 8.3,
                         2.3, 2.3, 2.3,
                         1.2, 1.2, 1.2,
                         7.3, 7.3, 7.3,
                         9.3, 9.3, 9.3,
                         15, 15, 15,
                         3, 3, 3,
                         1.1, 1.1, 1.1,
                         12, 12, 12,
                         4, 4, 4,
                         5, 5, 5};
    double labels[100];
    for(size_t i = 0; i < 12; ++i)
        labels[i] = (double)i;
    tree_model *model = build_kdtree(datas, labels, 12, 3, 2);
    double *ans = k_nearests_neighbor(model, test, 2, 5, false);
    printf("k Nearest Neighbors Regressor: \n%.2f %.2f\n", ans[0], ans[1]);
    free(ans);
    free_tree_memory(model->root);
}


```
## Build
### Linux and Mac OS
* `mkdir build`
* `cd build`
* `cmake ..`
* `make`
* `cd ..`
* `cp build/libkdtree* python/kdtree` (if you would like to build python package)
### Windows
Use *cmake-gui* to build(If your operating system is **Windows 10 X64** and you have **Visual Studio 2017** installed, you can also run `build.bat`).

### Build Python Package
* `cd python`
* `python setup.py install`
* `python ./demo.py`(run a demo)

### Build MKL Support
Intel® Math Kernel Library is a very popular library product from Intel that accelerates math processing routines to increase application performance.
If you want to build with MKL support, please add `#define USE_INTEL_MKL` to `kdtree.cpp`.  
```cpp
......
#include <queue>
#include <cstring>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

// If you need to use Intel MKL to accelerate, 
// you can cancel the next line comment.

// #define USE_INTEL_MKL

#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif
......
```