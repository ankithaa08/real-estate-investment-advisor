@echo off
echo ========================================================
echo Real Estate Investment Advisor - Push to GitHub
echo ========================================================
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g., https://github.com/username/repo.git): "
if "%REPO_URL%"=="" (
    echo [ERROR] No repository URL provided. Aborting.
    pause
    exit /b 1
)

git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main
echo [INFO] Pushing to %REPO_URL% ...
git push -u origin main
echo.
echo ========================================================
echo [SUCCESS] Project pushed to GitHub!
echo.
echo Deploy for free to the public on Streamlit Community Cloud:
echo  1. Visit: https://share.streamlit.io/
echo  2. Click "New app"
echo  3. Choose this repository, branch 'main', and path 'app/app.py'
echo  4. Click "Deploy" to get your global https://*.streamlit.app URL!
echo ========================================================
pause
