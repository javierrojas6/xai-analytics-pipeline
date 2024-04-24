#!/usr/bin/env bash

mkdir -p packages/postgress


# download postgres source code
curl https://ftp.postgresql.org/pub/source/v15.5/postgresql-15.5.tar.gz -k --output packages/postgress/postgresql-15.5.tar.gz

cd packages/postgress

tar -xzvf postgresql-15.5.tar.gz

cd postgresql-15.5

./configure

make