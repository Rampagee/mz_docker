#! /bin/bash

# ===========================================================
# Copyright(C) 2015-2019 Mipsology SAS.  All Rights Reserved.
# ===========================================================

# wrapper script to download a file from Google Drive
# Syntax
# $0 https://drive.google.com/uc?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU save.dat

link="$1"
outfile="$2"

tmpfile=$( mktemp )

clean() {
  rm "$tmpfile"
}

trap clean EXIT

wget --no-check-certificate --load-cookies "$tmpfile" -O "$outfile" \
  "$link&export=download&confirm=$(
  wget --quiet --save-cookies "$tmpfile" --keep-session-cookies \
  --no-check-certificate -O - "$1&export=download" | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' )"

