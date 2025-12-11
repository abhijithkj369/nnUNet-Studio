@echo off
echo ================================================
echo   Cleaning Up Duplicate Dataset001 Folders
echo ================================================
echo.
echo This will remove old Dataset001 folders to fix the preprocessing error.
echo.
echo The following folders will be DELETED:
echo   - nnUNet_raw\Dataset001_001_Knee
echo   - nnUNet_preprocessed\Dataset001_Mandibularcanal
echo.
echo Your current airway dataset will remain safe.
echo.
pause

echo.
echo Removing old datasets...

if exist "nnUNet_raw\Dataset001_001_Knee" (
    echo Deleting nnUNet_raw\Dataset001_001_Knee...
    rmdir /s /q "nnUNet_raw\Dataset001_001_Knee"
    echo Done!
)

if exist "nnUNet_preprocessed\Dataset001_Mandibularcanal" (
    echo Deleting nnUNet_preprocessed\Dataset001_Mandibularcanal...
    rmdir /s /q "nnUNet_preprocessed\Dataset001_Mandibularcanal"
    echo Done!
)

echo.
echo ================================================
echo   Cleanup Complete!
echo ================================================
echo.
echo You can now run preprocessing for your airway dataset.
echo.
pause
