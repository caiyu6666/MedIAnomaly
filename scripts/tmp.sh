#!/bin/bash

#array=("a" "b" "c")
# shellcheck disable=SC2068
# shellcheck disable=SC2034
for((i=1;i<=3;i=i+1));do
  python tmp.py
done
