param([string]$OutputDirectory = 'artifacts')
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path $PSScriptRoot -Parent
$outputPath = [IO.Path]::GetFullPath((Join-Path $repoPath $OutputDirectory))
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$officeArchive = Join-Path $outputPath 'office31_images.tar'
& tar -cf $officeArchive -C (Join-Path $repoPath 'data') domain_adaptation_images
if ($LASTEXITCODE -ne 0) { throw 'Office-31 archive creation failed' }
$weightDirectory = Get-ChildItem -LiteralPath $repoPath -Directory | Where-Object Name -Like '*model' | Select-Object -First 1
if (-not $weightDirectory) { throw 'Pretrained model directory not found' }
$weightPath = Join-Path $weightDirectory.FullName 'resnet50-19c8e357.pth'
Copy-Item -LiteralPath $weightPath -Destination $outputPath -Force
$archivePaths = @($officeArchive, (Join-Path $outputPath 'resnet50-19c8e357.pth'))
$manifest = foreach ($filePath in $archivePaths) {
    $fileItem = Get-Item -LiteralPath $filePath
    [ordered]@{file=$fileItem.Name; bytes=$fileItem.Length; sha256=(Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLower()}
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'data-manifest.json') -Encoding utf8
$manifest | Format-Table -AutoSize
