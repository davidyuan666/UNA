# PowerShell setup script for SeemingAI-backend

Write-Host "🚀 Starting Windows development environment setup..." -ForegroundColor Green

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Configure pip to use Tsinghua mirror
Write-Host "🔧 Configuring pip to use Tsinghua mirror..." -ForegroundColor Blue
if (-not (Test-Path "$HOME\pip")) {
    New-Item -ItemType Directory -Force -Path "$HOME\pip"
}
@"
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
"@ | Out-File -FilePath "$HOME\pip\pip.conf" -Encoding UTF8 -Force

# Check Python installation
Write-Host "🐍 Checking Python installation..." -ForegroundColor Blue
if (Test-Command python) {
    $pythonVersion = python --version
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found. Please install Python 3.12" -ForegroundColor Red
    exit 1
}

# Check pip installation
Write-Host "📦 Checking pip installation..." -ForegroundColor Blue
if (-not (Test-Command pip)) {
    Write-Host "Installing pip..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
    python get-pip.py
    Remove-Item get-pip.py
}

# Configure PDM to use Tsinghua mirror
Write-Host "🔧 Configuring PDM to use Tsinghua mirror..." -ForegroundColor Blue
$pdmConfig = @"
[[pypi]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
verify_ssl = true
name = "tuna"

[[pypi]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"
"@

# Create PDM config directory if it doesn't exist
$pdmConfigPath = "$env:APPDATA\pdm"
if (-not (Test-Path $pdmConfigPath)) {
    New-Item -ItemType Directory -Force -Path $pdmConfigPath
}
$pdmConfig | Out-File -FilePath "$pdmConfigPath\config.toml" -Encoding UTF8 -Force

# Install PDM if not present
Write-Host "📦 Checking PDM installation..." -ForegroundColor Blue
if (-not (Test-Command pdm)) {
    Write-Host "Installing PDM..." -ForegroundColor Yellow
    python -m pip install --user pdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # Add PDM to PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pdmPath = "$env:APPDATA\Python\Scripts"
    if ($userPath -notlike "*$pdmPath*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$pdmPath", "User")
        $env:Path = "$env:Path;$pdmPath"
    }
}

# Configure npm to use Taobao mirror
Write-Host "🔧 Configuring npm to use Taobao mirror..." -ForegroundColor Blue
if (Test-Command npm) {
    npm config set registry https://registry.npmmirror.com
    Write-Host "✅ npm configured to use Taobao mirror" -ForegroundColor Green
}

# Check Node.js installation
Write-Host "📦 Checking Node.js installation..." -ForegroundColor Blue
if (-not (Test-Command node)) {
    Write-Host "❌ Node.js not found. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Install project dependencies
Write-Host "📦 Installing project dependencies..." -ForegroundColor Blue
try {
    pdm install
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install dependencies: $_" -ForegroundColor Red
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "📝 Creating .env file..." -ForegroundColor Blue
    @"
# OpenAI API Key
OPENAI_API_KEY=your-api-key-here

# Worker URLs (comma-separated)
WORKER_URLS=http://localhost:8000,http://localhost:8001

# Other configuration
PORT=6001
"@ | Out-File -FilePath .env -Encoding UTF8
    Write-Host "✅ Created .env file. Please update with your actual values." -ForegroundColor Green
}

Write-Host @"

✨ Setup completed! ✨

Sources configured:
- pip: Tsinghua Mirror
- pdm: Tsinghua Mirror
- npm: Taobao Mirror

To start development:
1. Update the .env file with your actual values
2. Run 'make serve' to start the development server
3. Run 'make logs' to view the server logs

Additional commands:
- 'make stop' to stop the server
- 'make test' to run tests
- 'make commit' for git operations

Happy coding! 🎉
"@ -ForegroundColor Cyan