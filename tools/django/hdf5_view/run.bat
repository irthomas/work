set root=C:\Users\iant\Documents\PROGRAMS\anaconda3
call %root%\Scripts\activate.bat %root%

start "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "http://127.0.0.1:8000"

cd C:\Users\iant\Dropbox\NOMAD\Python\django\hdf5_view
python C:\Users\iant\Documents\PROGRAMS\anaconda3\Scripts\django-admin.py runserver --settings=viewer --pythonpath=.

