To run the models first clone the repository to a device using the following command in the terminal: 
git clone https://github.com/luciaadan/nlp.git

The project was done in WSL and it requires a virtual environment to be run. The necessary files are in the repository but it also needs to be activated on the device so the following command must be executed in the terminal: 
    uv sync
    source .venv/bin/activate

Since the most important file is a python file, the code can be run in the terminal by the following commands:
    cd notebooks
    python3 assignment_1.py