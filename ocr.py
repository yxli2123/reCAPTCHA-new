import ocrspace
import os
import shutil
import time

imgs = os.listdir('collected')
dumpdir = 'yidun'

for img in imgs:
    api = ocrspace.API(language=ocrspace.Language.Chinese_Simplified)
    name = api.ocr_file(f'collected/{img}').replace('“', '').replace('"', '').replace("'", '').strip()[-3:]
    shutil.copy(f'collected/{img}', f'{dumpdir}/{name}.png')
    time.sleep(60)
