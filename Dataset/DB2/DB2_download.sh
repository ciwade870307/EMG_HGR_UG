#!/bin/bash

for i in {4..40}
do
  wget https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s$i.zip
  unzip DB2_s$i.zip
  rm -rf DB2_s$i.zip
done
