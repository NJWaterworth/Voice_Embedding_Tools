#!/bin/bash

count="$1"+1
cwd="$(pwd)"

mkdir "embeddings"
mkdir "faces"


for i in $(seq -f "%04g" 1 $1)
do
  cd "/home/nick/voxcelebdb/wav/id1${i}/"
  find "$(pwd)" -iname \*.wav > temp.txt
  echo $i

  cd "/home/nick/voxcelebdb/"
  name="$(cat vox_meta.csv|gawk -v var="id1${i}" '{if ($1 == var) {print $2}}')"

  echo $name 
  cd $cwd
  echo NEW DIRECTORY 
  mkdir "embeddings/id1${i}"
  mkdir "faces/id1${i}"  

  cp -a "/home/nick/Face-Voice-DB/VGG_ALL_FRONTAL/${name}/." "faces/id1${i}"

  p="/home/nick/voxcelebdb/wav/id1${i}/temp.txt"
  s="./embeddings/id1${i}/"
  python generate.py $p $s

done
