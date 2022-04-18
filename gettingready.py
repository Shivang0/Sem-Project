import zipfile

in_dir = "wholemodel"
zipdir = 'wholemodel.zip'
with zipfile.ZipFile(zipdir, 'r') as zip_ref:
    zip_ref.extractall(in_dir)