# Preload the agent's model into memory at boot so the first real request pays no load cost.
# OLLAMA_KEEP_ALIVE=-1 keeps a model resident once loaded, but nothing loads it at startup -- this does.
# Register as a startup task (e.g. Windows Task Scheduler, AtStartup) after the Ollama server task.
#
# Usage: powershell -ExecutionPolicy Bypass -File ollama_warm.ps1 [-Model agent-model] [-Log path]

param(
    [string]$Model = "agent-model",
    [string]$Log   = (Join-Path $env:TEMP "ollama-warm.log")
)

$ErrorActionPreference = "SilentlyContinue"
$uri = "http://127.0.0.1:11434/api/generate"

# Wait for the server to bind the port (up to ~3 min after boot).
for ($i = 0; $i -lt 36; $i++) {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 5 | Out-Null
        break
    } catch { Start-Sleep -Seconds 5 }
}

$body = @{
    model   = $Model
    prompt  = "hi"
    stream  = $false
    options = @{ num_predict = 1 }
} | ConvertTo-Json -Depth 4

$t0 = Get-Date
try {
    Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json" -Body $body -TimeoutSec 600 | Out-Null
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Add-Content -Path $Log -Value ("[" + (Get-Date -Format s) + "] warmed " + $Model + " in " + $secs + "s")
} catch {
    Add-Content -Path $Log -Value ("[" + (Get-Date -Format s) + "] warm FAILED: " + $_.Exception.Message)
}
