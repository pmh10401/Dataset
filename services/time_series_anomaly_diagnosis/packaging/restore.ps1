$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$archive = Join-Path $PSScriptRoot 'diagnosis-windows.zip'
$destination = Join-Path $PSScriptRoot 'TimeSeriesAnomalyDiagnosis'
function Read-Sha256($path) {
    $stream = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $path))
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '') }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
try {
    if ((Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath $destination)) {
        throw 'Existing archive or application folder found. Restore in a new empty folder.'
    }
    $manifest = Get-Content -LiteralPath 'manifest.json' -Raw | ConvertFrom-Json
    if ($manifest.schema -ne 1 -or $manifest.parts.Count -lt 1) { throw 'Invalid manifest.' }
    $index = 1
    foreach ($part in $manifest.parts) {
        if ($part.name -cne ('diagnosis-windows.zip.{0:D3}' -f $index)) { throw 'Invalid part sequence.' }
        $index++
        Write-Host "Checking $($part.name)..."
        if ((Get-Item -LiteralPath $part.name).Length -ne $part.bytes -or
            (Read-Sha256 $part.name) -ne $part.sha256) {
            throw "Missing or damaged part: $($part.name)"
        }
    }
    $output = [System.IO.File]::Open($archive, [System.IO.FileMode]::CreateNew)
    try {
        foreach ($part in $manifest.parts) {
            $inputStream = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $part.name))
            try { $inputStream.CopyTo($output) } finally { $inputStream.Dispose() }
        }
    } finally { $output.Dispose() }
    if ((Read-Sha256 'diagnosis-windows.zip') -ne $manifest.archive_sha256) {
        throw 'Archive checksum mismatch. No files were extracted.'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($archive, $PSScriptRoot)
    $database = Join-Path $destination 'data\univariate_ts.db'
    if ((Read-Sha256 'TimeSeriesAnomalyDiagnosis\data\univariate_ts.db') -ne $manifest.database_sha256) {
        throw 'Database checksum mismatch. Do not run the application.'
    }
    Write-Host 'Ready. Open TimeSeriesAnomalyDiagnosis.exe in the application folder.' -ForegroundColor Green
    Start-Process explorer.exe -ArgumentList ('"' + $destination + '"')
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
