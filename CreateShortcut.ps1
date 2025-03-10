$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\JoyCaption.lnk")
$Shortcut.TargetPath = "$PSScriptRoot\JoyCaption.bat"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "JoyCaption - High Quality Image Captioning"
# If you have an icon file, uncomment the next line and specify the path
# $Shortcut.IconLocation = "$PSScriptRoot\icon.ico"
$Shortcut.Save()

Write-Host "Shortcut created on your desktop!"
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
