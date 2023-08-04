# pip install py-hanspell
# pip install git+https://github.com/ssut/py-hanspell.git
# hanspell 버전이 1.1.0 이상인 경우
from hanspell import spell_checker
# hanspell 버전이 1.0.1 이하인 경우
# from hanspell import speller
sent01 = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지"
spelled_sent = spell_checker.check(sent01)
hanspell_sent = spelled_sent.checked
print(hanspell_sent)

# 교정된 문장 출력

#     raise JSONDecodeError("Expecting value", s, err.value) from None
# json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

# json.decoder 오류 뜨면서 안됌
