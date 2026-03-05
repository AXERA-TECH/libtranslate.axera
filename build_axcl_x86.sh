#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_axcl_x86
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}

axcl_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/axcl_3.6.2_x86.zip"
if [ ! -d "axcl_3.6.2" ]; then
    echo "Downloading axcl from ${axcl_url}"
    if [ ! -f "axcl_3.6.2_x86.zip" ]; then
        wget ${axcl_url}
    fi
    unzip axcl_3.6.2_x86.zip
fi
axcl_dir=${PWD}/axcl_3.6.2


# 开始编译
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DAXCL_DIR=${axcl_dir} \
    -DBUILD_AX650=OFF \
    -DBUILD_AXCL=ON \
    ..
cmake --build . -j16
cmake --install .

# 复制库到 Python 目录（x86_64）
mkdir -p ../pytranslate/x86_64
cp libax_translate.so ../pytranslate/x86_64/
