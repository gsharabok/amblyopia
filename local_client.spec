# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['local_client.py'],
             pathex=[],
             binaries=[
                 ('shape_predictor_68_face_landmarks.dat',
                '.'),
                ('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml',
                'cv2/data'),
                ('venv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml',
                'cv2/data')
             ],
             datas=[('templates', 'templates'), ('static', 'static')],
             hiddenimports=['engineio.async_gevent'],
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
