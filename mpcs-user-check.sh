#!/usr/bin/env bash

#
# Use this script to check for other students on a CSIL/MPCS workstation.
# 
# Exclusive access is required for the GPU assignment.
#

echo Checking for other ssh users. You should see only 3 processes with 2 featuring your id.
pgrep -a sshd

echo Checking for xRDP users. There should be nothing following this message.
pgrep -a xrdp-chanserv

