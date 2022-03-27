# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
hiddenimports_dns = collect_submodules('dns')
hidden_imports_rest = ['engineio.async_gevent', 'eventlet', 'eventlet.hubs.epolls','eventlet.hubs.kqueue', 'eventlet.hubs.selects', 'dnspython', 'dns', 'dns.asyncbackend', 'dns.asyncquery', 'dns.*', 'dns.asyncresolver', 'gevent-websocket', 'geventwebsocket']
all_hidden_imports = hiddenimports_dns + hidden_imports_rest

block_cipher = None


a = Analysis(['local_client.py'],
             pathex=[],
             binaries=[
                ('shape_predictor_68_face_landmarks.dat',
                '.'),
                ('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml',
                'cv2/data'),
                ('venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml',
                'cv2/data'),
                ('C:/Users/Lenovo/anaconda3/pkgs/mkl-2020.2-256/Library/bin/mkl_avx2.dll',
                '.'),
                ('C:/Users/Lenovo/anaconda3/pkgs/mkl-2020.2-256/Library/bin/mkl_def.dll',
                '.')
             ],
             datas=[('templates', 'templates'), ('static', 'static')],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='local_client',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='local_client')
