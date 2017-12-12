:: For Windows 10 X64 user

rd /s/q build
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
cd ..
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64"
msbuild /property:Configuration="Release" build\libkdtree.sln
copy /y build\Release\kdtree.dll python\kdtree