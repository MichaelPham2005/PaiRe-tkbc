# Download ICEWS14, ICEWS05-15, yago15k and wikidata (PowerShell version)

# Lấy thư mục gốc (tương đương DIR trong bash)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $RootDir

# URL file
$Url = "https://dl.fbaipublicfiles.com/tkbc/data.tar.gz"
$OutputFile = "data.tar.gz"

# Tải file
Invoke-WebRequest -Uri $Url -OutFile $OutputFile

# Giải nén
tar -xvzf $OutputFile

# Xoá file nén
Remove-Item $OutputFile
