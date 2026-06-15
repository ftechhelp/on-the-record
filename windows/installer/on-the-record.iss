; Inno Setup script for On The Record (Windows tray app).
; Build with:  iscc windows\installer\on-the-record.iss
; Requires dist\On The Record.exe (run scripts\build_windows_tray.py first).

#define AppName "On The Record"
#define AppVersion "0.1.0"
#define AppPublisher "Vincent Fontaine"
#define AppExeName "On The Record.exe"

[Setup]
; A stable AppId keeps upgrades/uninstalls consistent across versions.
AppId={{8E0B6D2C-6F4E-4C1B-9F1A-7E2B7C0A4D11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\..\dist\installer
OutputBaseFilename=OnTheRecord-Setup-{#AppVersion}
SetupIconFile=..\assets\on-the-record.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
