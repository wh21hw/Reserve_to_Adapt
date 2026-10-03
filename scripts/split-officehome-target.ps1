$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path $PSScriptRoot -Parent
$taskDirectory = Join-Path $taskRepo 'artifacts/officehome-pr2rw-v1'
$taskArchive = Join-Path $taskDirectory 'real_world_0-64_test.zip'
$taskPartSize = 250MB
$taskBuffer = [byte[]]::new(8MB)
$taskRecords = @()
$taskInput = [IO.File]::OpenRead($taskArchive)
try {
    $taskIndex = 0
    while ($taskInput.Position -lt $taskInput.Length) {
        $taskName = 'real_world_0-64_test.zip.part{0:d3}' -f $taskIndex
        $taskPartPath = Join-Path $taskDirectory $taskName
        $taskOutput = [IO.File]::Open($taskPartPath, [IO.FileMode]::CreateNew)
        try {
            $taskWritten = 0L
            while ($taskWritten -lt $taskPartSize) {
                $taskRead = $taskInput.Read($taskBuffer, 0, [int][math]::Min($taskBuffer.Length, $taskPartSize - $taskWritten))
                if ($taskRead -eq 0) { break }
                $taskOutput.Write($taskBuffer, 0, $taskRead)
                $taskWritten += $taskRead
            }
        } finally { $taskOutput.Dispose() }
        $taskRecords += [ordered]@{file=$taskName; index=$taskIndex; bytes=(Get-Item $taskPartPath).Length; sha256=(Get-FileHash $taskPartPath).Hash.ToLower()}
        Write-Output "Prepared $taskName ($taskWritten bytes)"
        $taskIndex++
    }
} finally { $taskInput.Dispose() }
[ordered]@{file='real_world_0-64_test.zip'; bytes=(Get-Item $taskArchive).Length; sha256=(Get-FileHash $taskArchive).Hash.ToLower(); parts=$taskRecords} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskDirectory 'parts.json') -Encoding utf8
