#!/bin/bash

if [ ! -d $PWD/datasets ]; then
  mkdir $PWD/datasets
fi
echo ""
echo ...creating the datasets folder.
echo $PWD/datasets
echo ""
echo ...downloading the KAIST dataset.
echo ""

cd ./datasets
if [ ! -d ./images ]; then
  mkdir ./images
fi

cd ./images

for ii in $(seq -f %02g 0 1 11)
do
  wget http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set$ii.zip
  if [ ! -d ./set$ii ]; then
    mkdir ./set$ii
  fi
  unzip ./set$ii.zip -d ./set$ii
  rm set$ii.zip
done

cd ./set05
if [ ! -d ./V000 ]; then
  mkdir ./V000
fi
mv ./visible ./V000
mv ./lwir ./V000
echo ""
echo ...downloading process has been completed.
echo ""