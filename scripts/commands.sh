#!/bin/bash
jupyter notebook ../notebooks/Report_SER.ipynb --port=8888 --no-browser --ip=0.0.0.0 --allow-root & python app.py && fg

