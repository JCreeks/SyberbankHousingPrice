#!/usr/bin/env bash

cd features
python run.py
echo '============ training model ==========='
cd ../model
python run.py
