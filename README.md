First time using the project, do the following

1) cd to project directory
2) create the virtual environment for python -> python -m venv venv
3) activate the venv -> .\venv\Scripts\Activate  (linux -> source ./venv/bin/activate)
4) install the required libraries -> pip install -r requirements.txt
5) install the newly curated Playwright module -> playwright install
6) deactivate <- for deactivating the venv

If all are done for the first time, then we can directly run the activate_venv_win_powershell.bat (from powershell) -> if do not work do it manually by running the following commands
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

or the activate_venv_lin.sh (in linux) to activate the virtual environment. 

Running the test_playwright_crawler.py file -> python scripts/test_playwright_crawler.py