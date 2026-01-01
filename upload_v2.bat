@echo off
echo --- Atl4s-Forex v2.0 Uploader ---
cd ..
echo Staging v2_0 modules...
git add v2_0/
echo Committing...
git commit -m "feat(v2.0): RELEASE v2.0 - Dynamic Risk Manager & Interactive Startup"
echo Pushing...
git push
echo Done.
pause
