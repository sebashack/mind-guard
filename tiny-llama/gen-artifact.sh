#!/usr/bin/bash

dir_path=$1

parent_path=$( dirname ${dir_path} )
dirname=$( basename ${dir_path} )

cd ${parent_path}
tar cf - ${dirname} | lzip -9 -o ${dirname}.tar.lz
cd -
