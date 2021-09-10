import os
import gdown

if os.path.exists('./model/tale_model.tar'):
    print("== Data existed.==")
    pass
else:
    os.system("rm -rf model")
    os.system("mkdir model")
    url = "https://drive.google.com/uc?id=1OOSOHJWAmtNWrMxyLjGJX7d9NlKmf85D"
    output = './model/tale_model.tar'
    print("Download tale_model.tar")
    gdown.download(url, output, quiet=False)