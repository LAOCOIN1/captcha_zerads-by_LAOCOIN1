#!/data/data/com.termux/files/usr/bin/bash

while true
do
  git add .
  git commit -m "auto update api zerads" 2>/dev/null
  git push 2>/dev/null
  sleep 2
done
