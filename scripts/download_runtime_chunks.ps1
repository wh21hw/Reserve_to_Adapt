param(
    [Parameter(Mandatory=$true)][string]$PartPrefix,
    [Parameter(Mandatory=$true)][int]$PartCount,
    [Parameter(Mandatory=$true)][string]$OutputName
)
# /content artifact transfer fallback when Drive is full. No hashes or deletion.
if ($PartPrefix -notmatch '^[a-zA-Z0-9-]+$' -or $OutputName -notmatch '^[a-zA-Z0-9_.-]+$' -or $PartCount -lt 1) {
    throw 'Use simple names and a positive part count'
}
$taskRoot = Split-Path -Parent $PSScriptRoot
$partsRoot = Join-Path $taskRoot "pipeline-results/$PartPrefix-parts"
New-Item -ItemType Directory -Force -Path $partsRoot | Out-Null
$wslProject = '/mnt/c/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt'
for ($partNumber = 0; $partNumber -lt $PartCount; $partNumber++) {
    $partName = '{0}-{1:d2}' -f $PartPrefix, $partNumber
    $localPart = Join-Path $partsRoot $partName
    if (Test-Path -LiteralPath $localPart) { continue }
    & wsl -d Ubuntu-24.04 -u wanghao21 -- /home/wanghao21/.local/bin/colab fs download "content/$partName" --output "$wslProject/pipeline-results/$PartPrefix-parts/$partName" --endpoint gpu-l4-s-kkb-ass1a1-13n8bs33o8id1
    if ($LASTEXITCODE -ne 0) { throw "Transfer failed: $partName" }
}
$checkpointPath = Join-Path $taskRoot "pipeline-results/$OutputName"
$targetStream = [System.IO.File]::Open($checkpointPath, [System.IO.FileMode]::CreateNew)
try {
    for ($partNumber = 0; $partNumber -lt $PartCount; $partNumber++) {
        $partPath = Join-Path $partsRoot ('{0}-{1:d2}' -f $PartPrefix, $partNumber)
        $inputStream = [System.IO.File]::OpenRead($partPath)
        try { $inputStream.CopyTo($targetStream) } finally { $inputStream.Dispose() }
    }
} finally { $targetStream.Dispose() }
Write-Output "RUNTIME_ARTIFACT_SAVED $checkpointPath"
