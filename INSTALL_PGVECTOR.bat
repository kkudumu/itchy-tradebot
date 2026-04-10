@echo off
:: Build and install pgvector for PostgreSQL 16
:: Right-click -> Run as administrator

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo === Building pgvector for PostgreSQL 16 ===

:: Set up VS2019 Build Tools environment for x64
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Set PostgreSQL root
set "PGROOT=C:\Program Files\PostgreSQL\16"

:: Clone pgvector
cd /d %TEMP%
if exist pgvector rd /s /q pgvector
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector

:: Build
echo Building pgvector...
nmake /F Makefile.win

:: Install
echo Installing pgvector into PostgreSQL 16...
nmake /F Makefile.win install

:: Create the extension
echo Creating vector extension in trading database...
set PGPASSWORD=postgres
"%PGROOT%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "CREATE EXTENSION IF NOT EXISTS vector;"
"%PGROOT%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

echo.
echo === Done! ===
pause
