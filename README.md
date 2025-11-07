
# TornadoQ
*This work is being actively developed, and is not ready for industry use*

This repository deploys a tool to help government officals and other appropriate authorities classify tornadoes on the EF scale so as to improve the quality of decisions made in response to these emergencies.

## Setting up TornadoQ
This project was set up with Python 3.10.14. Certain necessary dependencies may not be compatible with newer versions of Python.

1. Create a virtural environment
   To initialize the enviroment on Linux, MacOS, or Windows open a terminal window and run
   ```
   python -m venv .venv
   ```
   Note for certain distributions of linux and MacOS, replace ```python``` with ```python3```.

   For the appropriate OS, the enviroment is activated as follows:
   ### Windows (command prompt)
   ```
   .venv\Scripts\activate
   ```

   ### Windows (Powershell)
   ```
   .venv\Scripts\Activate.ps1

   ```

   ### MacOS
   ```
   source .venv/bin/activate

   ```

   ### Linux
   ```
   source .venv/bin/activate

   ```
2. Install the repository as a python package

```
pip install -e .
pip install -r requirements.txt
```