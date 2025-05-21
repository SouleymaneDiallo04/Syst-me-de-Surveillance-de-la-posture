@echo off
echo Compilation de la documentation Sphinx...
sphinx-build -E -W -b html docs/source docs/build/html
if errorlevel 1 (
    echo Erreur lors de la compilation !
    pause
    exit /b 1
)
echo Compilation réussie.
echo Ouverture de la documentation...
start docs\build\html\index.html