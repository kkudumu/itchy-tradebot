# install_extensions.ps1 — Install TimescaleDB into PostgreSQL 16
# Non-interactive, must run as Administrator
$ErrorActionPreference = "Continue"
$PG_DIR = "C:\Program Files\PostgreSQL\16"
$TSDB_SRC = "C:\Users\kkudu\AppData\Local\Temp\extensions\timescaledb-pg16\timescaledb"

Write-Host "=== Installing TimescaleDB 2.26.0 for PostgreSQL 16 ==="

# Copy extension files
Copy-Item "$TSDB_SRC\timescaledb*.sql" "$PG_DIR\share\extension\" -Force 2>&1
Copy-Item "$TSDB_SRC\timescaledb.control" "$PG_DIR\share\extension\" -Force 2>&1
Copy-Item "$TSDB_SRC\timescaledb*.dll" "$PG_DIR\lib\" -Force 2>&1
Write-Host "Files copied."

# Update postgresql.conf
$pgconf = "$PG_DIR\data\postgresql.conf"
$content = Get-Content $pgconf -Raw
if ($content -notmatch "shared_preload_libraries\s*=.*timescaledb") {
    Add-Content $pgconf "`nshared_preload_libraries = 'timescaledb'`n"
    Write-Host "Added timescaledb to shared_preload_libraries"
} else {
    Write-Host "shared_preload_libraries already set"
}

# Restart PG16
Write-Host "Restarting PostgreSQL 16..."
Restart-Service -Name "postgresql-x64-16" -Force 2>&1
Start-Sleep -Seconds 5
Write-Host "Restarted."

# Create database + extensions + user permissions
$env:PGPASSWORD = "postgres"
$psql = "$PG_DIR\bin\psql.exe"

& $psql -U postgres -h localhost -p 5433 -c "SELECT 1 FROM pg_database WHERE datname='trading'" -t 2>&1 | Out-String | ForEach-Object {
    if ($_.Trim() -ne "1") {
        & $psql -U postgres -h localhost -p 5433 -c "CREATE DATABASE trading;" 2>&1
    }
}

# Create trader role if not exists
& $psql -U postgres -h localhost -p 5433 -c "DO `$`$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='trader') THEN CREATE USER trader WITH PASSWORD 'xVImEbuok65p5M3rgTzzQS5L9d3R3j6k' CREATEDB; END IF; END `$`$;" 2>&1

# Create extensions
& $psql -U postgres -h localhost -p 5433 -d trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" 2>&1

# Grant permissions
& $psql -U postgres -h localhost -p 5433 -d trading -c "GRANT ALL PRIVILEGES ON DATABASE trading TO trader;" 2>&1
& $psql -U postgres -h localhost -p 5433 -d trading -c "GRANT ALL ON SCHEMA public TO trader;" 2>&1
& $psql -U postgres -h localhost -p 5433 -d trading -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trader;" 2>&1
& $psql -U postgres -h localhost -p 5433 -d trading -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trader;" 2>&1

# Verify
& $psql -U postgres -h localhost -p 5433 -d trading -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('timescaledb', 'vector');" 2>&1

Write-Host "Done."
