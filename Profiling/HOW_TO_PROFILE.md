# How to run profiling on a python script
In this folder you can find a batch file called `profile.bat` that automatically runs a python script, saves the profiling data in a temp file, visualizes the profiling using snakeviz and deletes the temp file. 

To use it first make sure you have snakeviz installed. You can do this by running the following command in your terminal:
```bash
pip install snakeviz
```
Then you can run the batch file by running the following command in your terminal:
```bash
profile.bat <python_script.py>
```
Where `<python_script.py>` is the path to the python script you want to profile.
