import re
def cleaning(text):
    #따옴표 통일
    _filter = re.compile('[“]+')
    text = _filter.sub(' "', text)
    _filter = re.compile('[”]+')
    text = _filter.sub('" ', text)
    _filter = re.compile("[']+")
    text = _filter.sub("' ", text)
    _filter = re.compile('[`]+')
    text = _filter.sub(" '", text)   
    #자음 제거
    _filter = re.compile('[ㄱ-ㅣ]+')
    text = _filter.sub('', text)
    #괄호와 내용 제거
    _filter = re.compile('\([^)]*\)')
    text = _filter.sub('', text)
    #한영,숫자,문장부호 외 제거
    _filter = re.compile('[^가-힣 0-9 a-z A-Z \. \, \' \" \? \! \" \' \n]+')
    text = _filter.sub('', text)
    return text