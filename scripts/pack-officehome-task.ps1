$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path $PSScriptRoot -Parent
$taskData = [IO.Path]::GetFullPath((Join-Path $taskRepo 'data'))
$taskOutput = Join-Path $taskRepo 'artifacts/officehome-pr2rw-v1'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
Add-Type -AssemblyName System.IO.Compression
$taskManifest = @()
foreach ($taskList in @('product_0-24_train_all.txt', 'real_world_0-64_test.txt')) {
    $taskArchivePath = Join-Path $taskOutput ($taskList.Replace('.txt', '.zip'))
    if (Test-Path -LiteralPath $taskArchivePath) { throw "Refusing to overwrite $taskArchivePath" }
    $taskImages = @()
    $taskListPath = Join-Path $taskData $taskList
    $taskStream = [IO.File]::Open($taskArchivePath, [IO.FileMode]::CreateNew)
    $taskZip = [IO.Compression.ZipArchive]::new($taskStream, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $taskRelativePaths = @($taskList) + @(Get-Content -LiteralPath $taskListPath | ForEach-Object { $_ -replace '\s+\d+\s*$', '' })
        foreach ($taskRelative in $taskRelativePaths) {
            $taskImagePath = [IO.Path]::GetFullPath((Join-Path $taskData $taskRelative))
            if (-not $taskImagePath.StartsWith($taskData + [IO.Path]::DirectorySeparatorChar)) { throw 'Path outside dataset root' }
            $taskEntry = $taskZip.CreateEntry($taskRelative.Replace('\','/'), [IO.Compression.CompressionLevel]::Fastest)
            $taskInput = [IO.File]::OpenRead($taskImagePath)
            $taskEntryStream = $taskEntry.Open()
            try { $taskInput.CopyTo($taskEntryStream) } finally { $taskInput.Dispose(); $taskEntryStream.Dispose() }
            $taskImages += [ordered]@{path=$taskRelative; bytes=(Get-Item -LiteralPath $taskImagePath).Length; sha256=(Get-FileHash -LiteralPath $taskImagePath).Hash.ToLower()}
        }
    } finally { $taskZip.Dispose(); $taskStream.Dispose() }
    $taskRecord = [ordered]@{file=(Split-Path $taskArchivePath -Leaf); bytes=(Get-Item -LiteralPath $taskArchivePath).Length; sha256=(Get-FileHash -LiteralPath $taskArchivePath).Hash.ToLower(); entries=$taskImages}
    $taskManifest += $taskRecord
    Write-Output "$($taskRecord.file): $($taskRecord.bytes) bytes, $($taskImages.Count) entries, SHA256=$($taskRecord.sha256)"
}
$taskManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $taskOutput 'manifest.json') -Encoding utf8
