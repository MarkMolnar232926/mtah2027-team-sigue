# Link already-downloaded data into the (gitignored) places solution/ expects. Nothing is copied
# and nothing here ends up in git.
#
#   .\wsp\setup_local_data.ps1 -Source ..\more-than-a-hack-2027-TEAM_SIGUE
#
# -Source is a folder that holds
#   data\tactical\BAR-ATH_tactical_atd.{xml,txt}         the provided match (via MTAH_DATA_DIR)
#   data_external\dfl\DFL_0*.xml                         DFL open data, 7 matches
#   data_external\skillcorner\data\matches\...           SkillCorner opendata clone
#   data_external\metrica\data\Sample_Game_{1,2,3}\*     Metrica sample data
#
# Hardlinks need source and repository on the same volume; junctions do not need admin rights.
param([Parameter(Mandatory = $true)][string]$Source)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = (Resolve-Path $Source).Path

function Link-File($from, $to) {
    if (-not (Test-Path $to)) { New-Item -ItemType HardLink -Path $to -Target $from | Out-Null }
}

# DFL -> external/raw/*.xml
$raw = Join-Path $root "external\raw"
New-Item -ItemType Directory -Force $raw | Out-Null
$dfl = Get-ChildItem (Join-Path $src "data_external\dfl") -Filter "DFL_0*.xml"
foreach ($f in $dfl) { Link-File $f.FullName (Join-Path $raw $f.Name) }
"DFL files linked: $($dfl.Count)"

# SkillCorner -> external/skillcorner (junction)
$sc = Join-Path $root "external\skillcorner"
if (-not (Test-Path $sc)) {
    New-Item -ItemType Junction -Path $sc -Target (Join-Path $src "data_external\skillcorner") | Out-Null
}
"SkillCorner matches: $((Get-ChildItem (Join-Path $sc 'data\matches') -Directory).Count)"

# Metrica -> external/metrica/* (flat)
$met = Join-Path $root "external\metrica"
New-Item -ItemType Directory -Force $met | Out-Null
$mfiles = Get-ChildItem (Join-Path $src "data_external\metrica\data") -Recurse -File
foreach ($f in $mfiles) { Link-File $f.FullName (Join-Path $met $f.Name) }
"Metrica files linked: $($mfiles.Count)"

# The provided match is read through MTAH_DATA_DIR (see games.py); set it in the shell you run from.
$data = Join-Path $src "data"
"Now run:  `$env:MTAH_DATA_DIR = `"$data`""
