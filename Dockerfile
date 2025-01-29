# Dockerfile for Windows Server Core (matching your host's OS version),
# with Python 3.9.13, PyQt5, ib_insync, and TWS connectivity environment.

FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["cmd", "/S", "/C"]

# 1) Download & install Python 3.9.13 (64-bit)
RUN powershell -Command ^
    $ProgressPreference = 'SilentlyContinue'; ^
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe' -OutFile 'C:\temp\python-3.9.13-amd64.exe' ; ^
    Start-Process 'C:\temp\python-3.9.13-amd64.exe' -Wait -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1'

# 2) Ensure pip is up to date
RUN python -m ensurepip --upgrade
RUN pip install --upgrade pip

# 3) Install your Python dependencies
RUN pip install pyqt5 ib_insync nest_asyncio statsmodels numpy pandas

# 4) Expose port 7497 for TWS (if TWS is inside container)
# If TWS is on the host, you may or may not need this. We'll demonstrate.
EXPOSE 7497

# 5) Set environment variables for TWS connectivity if needed:
ENV TWS_HOST=host.docker.internal
ENV TWS_PORT=7497

# 6) Create app directory, copy your code
WORKDIR C:\\app
COPY . C:\\app

# 7) By default, run your main Python script
CMD ["python", "final_app.py"]
