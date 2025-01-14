# Windows 兼容性设置
ifeq ($(OS),Windows_NT)
    PYTHON = py -3.10
    PIP = $(PYTHON) -m pip
    PDM = pdm
    VENV_BIN = .venv\Scripts
    VENV_PYTHON = $(VENV_BIN)\python.exe
    RM = if exist $(1) rmdir /s /q $(1)
    MKDIR = if not exist $(1) mkdir $(1)
else
    PYTHON = python3
    PIP = $(PYTHON) -m pip
    PDM = pdm
    VENV_BIN = .venv/bin
    VENV_PYTHON = $(VENV_BIN)/python
    RM = rm -rf
    MKDIR = mkdir -p
endif


# 检查虚拟环境
venv-check:
ifeq ($(OS),Windows_NT)
	@if not exist .venv ( \
		echo "Creating virtual environment..." & \
		$(PYTHON) -m venv .venv & \
		echo "Installing dependencies..." & \
		$(VENV_PYTHON) -m pip install -e . \
	)
	@if not exist $(VENV_PYTHON) ( \
		echo "Virtual environment is broken. Please run 'make install' first." & \
		exit 1 \
	)
else
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv .venv; \
		echo "Installing dependencies..."; \
		$(VENV_PYTHON) -m pip install -e .; \
	fi
	@if [ ! -f $(VENV_PYTHON) ]; then \
		echo "Virtual environment is broken. Please run 'make install' first."; \
		exit 1; \
	fi
endif


install:
	@echo "Installing dependencies..."
	@if ! command -v pdm >/dev/null 2>&1; then \
		echo "PDM not found. Installing PDM..."; \
		curl -sSL https://pdm-project.org/install-pdm.py | python3 - ; \
	fi
	@echo "Configuring PDM to use mirror..."
	@pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple
	@echo "Clearing PDM cache..."
	@pdm cache clear
	@echo "Installing project dependencies..."
	@pdm install --no-lock
	@echo "Installing system dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		echo "Detected Debian/Ubuntu system"; \
		sudo apt-get update && \
		sudo apt-get install -y ffmpeg python3-dev; \
	elif command -v yum >/dev/null 2>&1; then \
		echo "Detected CentOS/RHEL system"; \
		sudo yum install -y ffmpeg python3-devel; \
	else \
		echo "Warning: Could not detect package manager. Please install ffmpeg manually."; \
	fi
	@echo "Installation completed successfully!"


	
ifeq ($(OS),Windows_NT)
# Windows specific commit command
commit:
	@powershell -Command "$$name = git config --get user.name; $$email = git config --get user.email; \
	if (-not $$name -or -not $$email) { \
		Write-Host 'Git user identity not configured. Please configure it first:' -ForegroundColor Yellow; \
		$$name = Read-Host 'Enter your name'; \
		git config --global user.name $$name; \
		$$email = Read-Host 'Enter your email'; \
		git config --global user.email $$email; \
		Write-Host 'Git identity configured successfully!' -ForegroundColor Green; \
	} \
	if (-not (git status --porcelain)) { \
		Write-Host 'No changes to commit. Working tree clean.' -ForegroundColor Yellow; \
		exit 0; \
	} \
	Write-Host 'Committing changes...' -ForegroundColor Cyan; \
	git add .; \
	$$msg = Read-Host 'Enter commit message'; \
	git commit -m $$msg; \
	Write-Host 'Changes committed successfully' -ForegroundColor Green; \
	if (git remote | Select-String 'origin') { \
		Write-Host 'Pushing to remote...' -ForegroundColor Cyan; \
		git push origin main; \
	} else { \
		Write-Host 'No remote repository configured. Skipping push.' -ForegroundColor Yellow; \
	}"

else
# Unix/Linux specific commit command
commit:
	@if [ -z "$$(git config --get user.name)" ] || [ -z "$$(git config --get user.email)" ]; then \
		echo "\033[93mGit user identity not configured. Please configure it first:\033[0m"; \
		echo "\033[93m1. Set your name:\033[0m"; \
		read -p "Enter your name: " name; \
		git config --global user.name "$$name"; \
		echo "\033[93m2. Set your email:\033[0m"; \
		read -p "Enter your email: " email; \
		git config --global user.email "$$email"; \
		echo "\033[92mGit identity configured successfully!\033[0m"; \
	fi
	@if git diff-index --quiet HEAD --; then \
		echo "\033[93mNo changes to commit. Working tree clean.\033[0m"; \
		exit 0; \
	fi
	@echo "Committing changes..."
	@git add .
	@read -p "Enter commit message: " msg && git commit -m "$$msg"
	@echo "\033[92mChanges committed successfully\033[0m"
	@if git remote | grep -q "origin"; then \
		echo "Pushing to remote..."; \
		git push origin main || echo "\033[91mPush failed. You may need to pull first.\033[0m"; \
	else \
		echo "\033[93mNo remote repository configured. Skipping push.\033[0m"; \
	fi
endif


setup:
	powershell -ExecutionPolicy Bypass -File toolscripts/setup.ps1

.PHONY: venv-check install commit