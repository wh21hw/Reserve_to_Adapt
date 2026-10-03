# Save the selected cloud best checkpoint in small transfer chunks; no hashes.
$taskRoot = Split-Path -Parent $PSScriptRoot
$partsRoot = Join-Path $taskRoot 'pipeline-results/rta-selected-weight-parts'
New-Item -ItemType Directory -Force -Path $partsRoot | Out-Null
$wslProject = '/mnt/c/Users/46025/Desktop/复现论文/osda2019/Reserve_to_Adapt'
for ($partNumber = 0; $partNumber -le 12; $partNumber++) {
    $partName = 'rta-seed3-best-part-{0:d2}' -f $partNumber
    $localPart = Join-Path $partsRoot $partName
    if (Test-Path -LiteralPath $localPart) { continue }
    & wsl -d Ubuntu-24.04 -u wanghao21 -- /home/wanghao21/.local/bin/colab fs download "content/$partName" --output "$wslProject/pipeline-results/rta-selected-weight-parts/$partName" --endpoint gpu-l4-s-kkb-ass1a1-13n8bs33o8id1
    if ($LASTEXITCODE -ne 0) { throw "Transfer failed: $partName" }
}
$checkpointPath = Join-Path $taskRoot 'pipeline-results/rta-legacy-l4-seed3-best.pt'
$targetStream = [System.IO.File]::Open($checkpointPath, [System.IO.FileMode]::CreateNew)
try {
    for ($partNumber = 0; $partNumber -le 12; $partNumber++) {
        $partPath = Join-Path $partsRoot ('rta-seed3-best-part-{0:d2}' -f $partNumber)
        $inputStream = [System.IO.File]::OpenRead($partPath)
        try { $inputStream.CopyTo($targetStream) } finally { $inputStream.Dispose() }
    }
} finally { $targetStream.Dispose() }
Write-Output "SELECTED_BASELINE_WEIGHT_SAVED $checkpointPath"
