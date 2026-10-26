# Start the whole Money Plant stack on Windows: ML service :8001 -> backend :8000 -> frontend :5173.
# UNTESTED on Windows (written to mirror scripts/dev.sh).
#
#   .\scripts\dev.ps1                 set up (first run only), then start all three services
#   .\scripts\dev.ps1 -SetupOnly      create venvs / install deps / check artifacts + SHAP report, then exit
#   .\scripts\dev.ps1 -NoFrontend     start only the ML service and the backend
#
# Needs Python >= 3.12 (the pinned ML dependencies require it), Node.js >= 20 and npm.
# Ctrl-C stops every service. Logs: .dev-logs\*.log
param(
    [switch]$SetupOnly,
    [switch]$NoFrontend
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$MlDir = Join-Path $Root 'money_plant_ml_pipeline'
$BeDir = Join-Path $Root 'backend'
$FeDir = Join-Path $Root 'frontend'
$LogDir = Join-Path $Root '.dev-logs'
$MlPort = 8001
$BePort = 8000
$FePort = 5173
$MlWorkers = 4

$MlPy = Join-Path $MlDir '.venv\Scripts\python.exe'
$BePy = Join-Path $BeDir '.venv\Scripts\python.exe'

function Info($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "ok  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "warning: $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

# ---- prerequisites -------------------------------------------------------------------------
function Find-Python {
    $candidates = @()
    if ($env:PYTHON) { $candidates += , @($env:PYTHON) }
    $candidates += , @('py', '-3.13')
    $candidates += , @('py', '-3.12')
    $candidates += , @('python')
    $candidates += , @('python3')
    foreach ($c in $candidates) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        $extra = @()
        if ($c.Count -gt 1) { $extra = $c[1..($c.Count - 1)] }
        try {
            & $c[0] @extra -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return , @($c[0], $extra) }
        } catch { }
    }
    return $null
}

Info 'Checking prerequisites'
$found = Find-Python
if (-not $found) {
    Fail 'Python >= 3.12 not found. The pinned ML dependencies (numpy 2.5 / scipy 1.18 / xgboost 3.4 / shap 0.54) need Python 3.12 or newer. Install it from python.org and re-run (or set $env:PYTHON).'
}
$SysPy = $found[0]
$SysPyArgs = @($found[1])
Ok "python: $SysPy $($SysPyArgs -join ' ')"

if (-not $NoFrontend -or $SetupOnly) {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail 'Node.js >= 20 not found. Install it from https://nodejs.org.' }
    $nodeVersion = (& node --version).Trim()
    if ($nodeVersion -match '^v?(\d+)\.') {
        $nodeMajor = [int]$Matches[1]
    } else {
        Fail "Could not determine the Node.js version (found '$nodeVersion')."
    }
    if ($nodeMajor -lt 20) { Fail "Node.js >= 20 is required (found $nodeVersion)." }
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { Fail 'npm not found (it normally ships with Node.js).' }
    Ok "node: $nodeVersion"
}

# ---- helpers -------------------------------------------------------------------------------
function Get-DepsHash([string[]]$Files) {
    $text = ($Files | ForEach-Object { Get-Content -Raw -LiteralPath $_ }) -join "`n"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text + (& $SysPy @SysPyArgs -c 'import sys; print(sys.version)'))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '')
}

function Ensure-Venv([string]$Name, [string]$Dir, [string[]]$Requirements) {
    $py = Join-Path $Dir '.venv\Scripts\python.exe'
    $marker = Join-Path $Dir '.venv\.deps-hash'
    if (-not (Test-Path $py)) {
        Info "Creating $Name virtualenv"
        & $SysPy @SysPyArgs -m venv (Join-Path $Dir '.venv')
        if ($LASTEXITCODE -ne 0) { Fail "Could not create the $Name virtualenv." }
    }
    $want = Get-DepsHash $Requirements
    $have = ''
    if (Test-Path $marker) { $have = (Get-Content -Raw $marker).Trim() }
    if ($want -ne $have) {
        Info "Installing $Name dependencies (first run takes a few minutes)"
        $pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '-q')
        & $py -m pip install --disable-pip-version-check -q --upgrade pip
        foreach ($r in $Requirements) { $pipArgs += @('-r', $r) }
        & $py @pipArgs
        if ($LASTEXITCODE -ne 0) { Fail "Installing the $Name dependencies failed." }
        Set-Content -Path $marker -Value $want
    }
    Ok "$Name environment ready"
}

function Ensure-Frontend {
    $marker = Join-Path $FeDir 'node_modules\.dev-lock-hash'
    $want = (Get-FileHash -Algorithm SHA256 (Join-Path $FeDir 'package-lock.json')).Hash
    $have = ''
    if (Test-Path $marker) { $have = (Get-Content -Raw $marker).Trim() }
    if (-not (Test-Path (Join-Path $FeDir 'node_modules')) -or $want -ne $have) {
        Info 'Installing frontend dependencies (npm ci)'
        Push-Location $FeDir
        try { & npm.cmd ci --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { Fail 'npm ci failed.' } } finally { Pop-Location }
        Set-Content -Path $marker -Value $want
    }
    Ok 'frontend dependencies ready'
}

function Test-PortInUse([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(500) -and $client.Connected) { return $true }
        return $false
    } catch { return $false } finally { $client.Close() }
}

function Get-Json([string]$Url) {
    try { return Invoke-RestMethod -Uri $Url -TimeoutSec 3 } catch { return $null }
}

function Wait-Service([string]$Label, $Proc, [string]$Url, [scriptblock]$Cond, [int]$TimeoutSec, [string]$LogFile) {
    for ($i = 0; $i -lt $TimeoutSec * 2; $i++) {
        if ($Proc.HasExited) {
            Write-Host "error: the $Label process exited during startup. Last log lines:" -ForegroundColor Red
            if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
            exit 1
        }
        $d = Get-Json $Url
        if ($d -and (& $Cond $d)) { return }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "error: $Label did not become ready within ${TimeoutSec}s ($Url). Last log lines:" -ForegroundColor Red
    if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
    exit 1
}

# Start a process with stdout+stderr appended to a log file. Returns the Process object.
function Start-Svc([string]$Name, [string]$Exe, [string[]]$ArgList, [string]$WorkDir, [hashtable]$EnvVars) {
    $log = Join-Path $LogDir "$Name.log"
    $err = Join-Path $LogDir "$Name.err.log"
    foreach ($k in $EnvVars.Keys) { Set-Item -Path "Env:$k" -Value $EnvVars[$k] }
    return Start-Process -FilePath $Exe -ArgumentList $ArgList -WorkingDirectory $WorkDir `
        -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -NoNewWindow
}

# Stop a process and everything it started (uvicorn workers, node children).
function Stop-Tree($Proc) {
    if ($null -eq $Proc) { return }
    try { & taskkill /PID $Proc.Id /T /F 2>$null | Out-Null } catch { }
}

# ---- setup ---------------------------------------------------------------------------------
Ensure-Venv 'ML' $MlDir @((Join-Path $MlDir 'requirements-tested.txt'))
Ensure-Venv 'backend' $BeDir @((Join-Path $BeDir 'requirements.txt'))
if (-not $NoFrontend -or $SetupOnly) { Ensure-Frontend }

$latest = Join-Path $MlDir 'artifacts\latest.json'
$RunId = $null
if (Test-Path $latest) { $RunId = (Get-Content -Raw $latest | ConvertFrom-Json).run_id }
if (-not $RunId -or -not (Test-Path (Join-Path $MlDir "artifacts\$RunId\manifest.json"))) {
    Write-Host 'error: the ML model artifacts are missing (artifacts\latest.json or its run directory).' -ForegroundColor Red
    Write-Host 'Training is an explicit step and is never run automatically. Train once, then re-run this script:'
    Write-Host ''
    Write-Host '    cd money_plant_ml_pipeline'
    Write-Host '    .venv\Scripts\python.exe scripts\train_all.py --input data/raw/plant_sensor_data.csv'
    Write-Host ''
    exit 1
}
Ok "ML artifacts: run $RunId"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$shapGlob = Join-Path $MlDir "reports\$RunId\evaluation_*\shap_global_importance.json"
if (Get-ChildItem -Path $shapGlob -ErrorAction SilentlyContinue) {
    Ok 'global SHAP report present'
} else {
    Info "Generating the global SHAP report for $RunId (one-off, explanation only: no training)"
    Push-Location $MlDir
    try {
        & $MlPy scripts\generate_explanations.py --input data/raw/plant_sensor_data.csv --run-id $RunId *> (Join-Path $LogDir 'explain.log')
        $exit = $LASTEXITCODE
    } catch { $exit = 1 } finally { Pop-Location }
    if ($exit -eq 0 -and (Get-ChildItem -Path $shapGlob -ErrorAction SilentlyContinue)) {
        Ok 'global SHAP report written'
    } else {
        Warn 'could not generate the global SHAP report (see .dev-logs\explain.log). The Global SHAP page will show its empty state.'
    }
}

if ($SetupOnly) { Ok 'setup complete. Start everything with .\scripts\dev.ps1'; exit 0 }

# ---- ports ---------------------------------------------------------------------------------
$ports = @(@($MlPort, 'ML service'), @($BePort, 'backend'))
if (-not $NoFrontend) { $ports += , @($FePort, 'frontend') }
$busy = $false
foreach ($p in $ports) {
    if (Test-PortInUse $p[0]) { Write-Host "error: port $($p[0]) ($($p[1])) is already in use." -ForegroundColor Red; $busy = $true }
}
if ($busy) { Fail 'Free the port(s) above and re-run.' }

# ---- start ---------------------------------------------------------------------------------
$ml = $null; $be = $null; $fe = $null
try {
    Info "Starting the ML service on :$MlPort ($MlWorkers workers)"
    $ml = Start-Svc 'ml' $MlPy @('-m', 'uvicorn', 'plant_ml.api.main:app', '--host', '127.0.0.1', '--port', "$MlPort", '--workers', "$MlWorkers") $MlDir @{ PYTHONPATH = 'src' }
    Wait-Service 'ML service' $ml "http://127.0.0.1:$MlPort/health" { param($d) $d.status -eq 'ok' -and $d.models_loaded -eq $true } 120 (Join-Path $LogDir 'ml.err.log')
    Ok 'ML service healthy'

    Info "Starting the backend on :$BePort (single process)"
    $be = Start-Svc 'backend' $BePy @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$BePort") $BeDir @{}
    Wait-Service 'backend' $be "http://127.0.0.1:$BePort/api/health" { param($d) $d.status -in @('ok', 'degraded') } 90 (Join-Path $LogDir 'backend.err.log')
    Ok 'backend healthy'

    if (-not $NoFrontend) {
        Info "Starting the frontend on :$FePort"
        $fe = Start-Svc 'frontend' 'cmd.exe' @('/c', 'npm', 'run', 'dev', '--', '--host', '127.0.0.1', '--port', "$FePort", '--strictPort') $FeDir @{}
        Wait-Service 'frontend' $fe "http://127.0.0.1:$FePort/api/health" { param($d) $d.status -in @('ok', 'degraded') } 60 (Join-Path $LogDir 'frontend.err.log')
        Ok 'frontend healthy (proxying /api to the backend)'
    }

    Write-Host ''
    Write-Host 'Money Plant Monitoring is running' -ForegroundColor Green
    if (-not $NoFrontend) { Write-Host "  App:      http://127.0.0.1:$FePort    <- open this and press Run in the top bar" }
    Write-Host "  Backend:  http://127.0.0.1:$BePort/docs"
    Write-Host "  ML:       http://127.0.0.1:$MlPort/health"
    Write-Host "  Logs:     $LogDir"
    Write-Host 'Press Ctrl-C to stop everything.'

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($p in @($ml, $be, $fe)) {
            if ($null -ne $p -and $p.HasExited) { Fail "a service (pid $($p.Id)) exited unexpectedly. See $LogDir." }
        }
    }
} finally {
    Write-Host ''
    Info 'Stopping services'
    Stop-Tree $fe
    Stop-Tree $be
    Stop-Tree $ml
}
