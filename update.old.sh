#! /bin/bash

set -e

OWNER="CorneliuBoboc"
REPO="MONAD"
test -n "$OWNER"
test -n "$REPO"

cd $HOME
rm -rf $REPO
git clone https://github.com/$OWNER/$REPO.git
cd $REPO

