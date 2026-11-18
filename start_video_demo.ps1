$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$existingDemo = $null
try { $existingDemo = Invoke-RestMethod "http://127.0.0.1:8000/live/demo" -TimeoutSec 5 } catch {}
if ($existingDemo) {
    if (-not $existingDemo.enabled) { throw "Port 8000 is running without VIDEO DEMO. Stop that API before starting this mode." }
    Start-Process "http://localhost:8000/live"
    Write-Host "VIDEO DEMO is already running: http://localhost:8000/live"
    return
}
$env:VIDEO_DEMO_ENABLED = "1"
$demoPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
if (-not (Test-Path -LiteralPath $demoPython)) { throw "Existing Windows Python 3.13 not found" }
Start-Process "http://localhost:8000/live"
& $demoPython -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
