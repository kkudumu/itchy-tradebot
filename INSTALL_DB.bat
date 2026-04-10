@echo off
:: Run as admin - right-click this file and choose "Run as administrator"
:: Or just double-click and approve the UAC prompt

:: Self-elevate
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

set PG_DIR=C:\Program Files\PostgreSQL\16
set TSDB_SRC=C:\Users\kkudu\AppData\Local\Temp\extensions\timescaledb-pg16\timescaledb

echo === Installing TimescaleDB 2.26.0 for PostgreSQL 16 ===

echo Copying extension files...
copy /Y "%TSDB_SRC%\timescaledb*.sql" "%PG_DIR%\share\extension\"
copy /Y "%TSDB_SRC%\timescaledb.control" "%PG_DIR%\share\extension\"
copy /Y "%TSDB_SRC%\timescaledb*.dll" "%PG_DIR%\lib\"

echo Updating postgresql.conf...
findstr /C:"timescaledb" "%PG_DIR%\data\postgresql.conf" >nul 2>&1
if %errorlevel% neq 0 (
    echo shared_preload_libraries = 'timescaledb' >> "%PG_DIR%\data\postgresql.conf"
    echo Added timescaledb to shared_preload_libraries
) else (
    echo Already configured
)

echo Restarting PostgreSQL 16...
net stop postgresql-x64-16
timeout /t 3 /nobreak >nul
net start postgresql-x64-16
timeout /t 5 /nobreak >nul

echo Creating database and extensions...
set PGPASSWORD=postgres
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -c "SELECT 1 FROM pg_database WHERE datname='trading'" -t | findstr "1" >nul 2>&1
if %errorlevel% neq 0 (
    "%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -c "CREATE DATABASE trading;"
)
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -c "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='trader') THEN CREATE USER trader WITH PASSWORD 'xVImEbuok65p5M3rgTzzQS5L9d3R3j6k' CREATEDB; END IF; END $$;"
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "GRANT ALL PRIVILEGES ON DATABASE trading TO trader;"
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "GRANT ALL ON SCHEMA public TO trader;"
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trader;"
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trader;"

echo.
echo === Verification ===
"%PG_DIR%\bin\psql.exe" -U postgres -h localhost -p 5433 -d trading -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('timescaledb', 'vector');"

echo.
echo === DONE ===
pause
